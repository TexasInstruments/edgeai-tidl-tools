# Copyright (c) {2024 - 2024} Texas Instruments Incorporated
#
# All rights reserved not granted herein.
#
# Limited License.
#
# Texas Instruments Incorporated grants a world-wide, royalty-free, non-exclusive
# license under copyrights and patents it now or hereafter owns or controls to make,
# have made, use, import, offer to sell and sell ("Utilize") this software subject to the
# terms herein.  With respect to the foregoing patent license, such license is granted
# solely to the extent that any such patent is necessary to Utilize the software alone.
# The patent license shall not apply to any combinations which include this software,
# other than combinations with devices manufactured by or for TI ("TI Devices").
# No hardware patent is licensed hereunder.
#
# Redistributions must preserve existing copyright notices and reproduce this license
# (including the above copyright notice and the disclaimer and (if applicable) source
# code license limitations below) in the documentation and/or other materials provided
# with the distribution
#
# Redistribution and use in binary form, without modification, are permitted provided
# that the following conditions are met:
#
# *       No reverse engineering, decompilation, or disassembly of this software is
# permitted with respect to any software provided in binary form.
#
# *       any redistribution and use are licensed by TI for use only with TI Devices.
#
# *       Nothing shall obligate TI to provide you with source code for the software
# licensed and provided to you in object code.
#
# If software source code is provided to you, modification and redistribution of the
# source code are permitted provided that the following conditions are met:
#
# *       any redistribution and use of the source code, including any resulting derivative
# works, are licensed by TI for use only with TI Devices.
#
# *       any redistribution and use of any object code compiled from the source code
# and any resulting derivative works, are licensed by TI for use only with TI Devices.
#
# Neither the name of Texas Instruments Incorporated nor the names of its suppliers
#
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# DISCLAIMER.
#
# THIS SOFTWARE IS PROVIDED BY TI AND TI'S LICENSORS "AS IS" AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL TI AND TI'S LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.
"""
Module containing MaxPool layer specific functions and optimizations
"""
import logging
import copy
import onnx_graphsurgeon as gs
import onnx
import numpy as np


def tidl_modify_maxpool(graph: gs.Graph, onnx_graph: onnx.GraphProto, args: dict):
    """
    Wrapper function to modify MaxPool layers to satisfy TIDL constraints
    """
    if args['convert_maxpool_to_cascaded_maxpool']:
        logging.debug("Running convert_maxpool_to_cascaded_maxpool")
        tidl_convert_maxpool_to_cascaded_maxpool(graph, onnx_graph)



def tidl_convert_maxpool_to_cascaded_maxpool(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    The MaxPool layer with large kernel (> 3x3) is replaced with
    cascaded MaxPool layers wiht 3x3 kernel. Assume that the kernel size
    is NxN where N is odd
    """
    max_pools = [node for node in graph.nodes if node.op == "MaxPool"]

    for maxpool in max_pools:
        kernelsize = maxpool.attrs["kernel_shape"][0]

        if (kernelsize > 3):
            num_iter = (kernelsize - 1) // 2 - 1
            assert num_iter > 0

            maxpool.attrs["kernel_shape"] = [3,3]
            maxpool.attrs["pads"]         = [1,1,1,1]
            maxpool.attrs["strides"]      = [1,1]

            # copy and save maxpool.outputs
            saved_outputs = copy.copy(maxpool.outputs)

            outputs = [gs.Variable(f"{saved_outputs[0].name}.0",
                                   shape=maxpool.outputs[0].shape, dtype=np.float32)]
            maxpool.outputs = outputs

            # set inputs for the next maxpool to append
            inputs = maxpool.outputs
            for i in range(num_iter):
                if i == num_iter-1:
                    # For the last maxpool node, ouputs is set to saved_outputs
                    new_maxpool = gs.Node(op="MaxPool", name=f"{maxpool.name}."+f"{i+1}",
                                          attrs=maxpool.attrs, inputs=inputs,
                                          outputs=saved_outputs)
                else:
                    outputs = [gs.Variable(f"{saved_outputs[0].name}."+f"{i+1}",
                                           shape=maxpool.outputs[0].shape, dtype=np.float32)]
                    new_maxpool = gs.Node(op="MaxPool", name=f"{maxpool.name}."+f"{i+1}",
                                          attrs=maxpool.attrs, inputs=inputs,  outputs=outputs)

                    # set inputs for the next maxpool to append
                    inputs = new_maxpool.outputs

                # Append to the graph
                graph.nodes.append(new_maxpool)
