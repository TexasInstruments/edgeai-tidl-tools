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


def tidl_convert_maxpool_to_cascaded_maxpool(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    The MaxPool layer with large kernel (> 3x3) is replaced with
    cascaded MaxPool layers with 3x3 kernel, handling arbitrary stride.
    The strides are set so that the receptive field and downsampling
    match the original MaxPool, using the receptive field formula.
    """
    max_pools = [node for node in graph.nodes if node.op == "MaxPool"]

    for maxpool in max_pools:
        kernelsize = maxpool.attrs["kernel_shape"][0]
        orig_stride = maxpool.attrs["strides"][0]

        logging.debug(f"Checking MaxPool node '{maxpool.name}' (kernel={kernelsize}, stride={orig_stride})")

        if (kernelsize > 3) and (orig_stride >= 1):
            logging.info(f"Converting MaxPool node '{maxpool.name}' to cascaded 3x3 MaxPools.")
            num_layers = (kernelsize - 1) // 2
            strides = [1] * (num_layers - 1) + [orig_stride] if num_layers > 1 else [orig_stride]

            maxpool.attrs["kernel_shape"] = [3,3]
            maxpool.attrs["pads"]         = [1,1,1,1]
            maxpool.attrs["strides"]      = [strides[0], strides[0]]

            saved_outputs = copy.copy(maxpool.outputs)
            outputs = [gs.Variable(f"{saved_outputs[0].name}.0", shape=None, dtype=np.float32)]
            maxpool.outputs = outputs

            inputs = maxpool.outputs
            for i in range(1, num_layers):
                attrs = {
                    "kernel_shape": [3,3],
                    "pads": [1,1,1,1],
                    "strides": [strides[i], strides[i]]
                }
                if i == num_layers - 1:
                    new_maxpool = gs.Node(op="MaxPool", name=f"{maxpool.name}.{i+1}",
                                          attrs=attrs, inputs=inputs, outputs=saved_outputs)
                else:
                    outputs = [gs.Variable(f"{saved_outputs[0].name}.{i+1}", shape=None, dtype=np.float32)]
                    new_maxpool = gs.Node(op="MaxPool", name=f"{maxpool.name}.{i+1}",
                                          attrs=attrs, inputs=inputs, outputs=outputs)
                    inputs = new_maxpool.outputs
                graph.nodes.append(new_maxpool)
            logging.info(f"Finished converting MaxPool node '{maxpool.name}' to {num_layers} cascaded 3x3 MaxPools.")
        else:
            logging.debug(f"MaxPool node '{maxpool.name}' does not require conversion.")
