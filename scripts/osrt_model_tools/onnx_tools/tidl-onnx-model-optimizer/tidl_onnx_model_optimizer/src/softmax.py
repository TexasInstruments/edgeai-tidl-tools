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


def tidl_modify_softmax(graph: gs.Graph, onnx_graph: onnx.GraphProto, args: dict):
    """
    Wrapper function to modify SoftMax layers to satisfy TIDL constraints
    """

    if args['convert_softmax']:
        logging.debug("Running convert_softmax")
        tidl_convert_softmax(graph)


def tidl_convert_softmax(graph: gs.Graph):
    """
    The SoftMax layer with operation in the channel or height dimension is replaced with
    Transpose -> SoftMax -> Transpose to satisfy constraint of SoftMax layer only
    occuring in width dimension
    """
    softmaxes = [node for node in graph.nodes if node.op == "Softmax"]

    for idx, softmax in enumerate(softmaxes):
        # Dimension in which Softmax should occur
        softmax_dimension = softmax.attrs["axis"]

        # Assumes 4D tensor with NxCxHxW input
        if((len(softmax.inputs[0].shape) == 4) and (softmax_dimension != len(softmax.inputs[0].shape) - 1)):

            # If softmax op occurs across 1st axis (channel access)
            if(softmax_dimension == 1):
                logging.debug(f"Converting axis for layer {softmax.name} from dimension {softmax_dimension} to dimension -1")

                output_shape1 = [softmax.inputs[0].shape[0], softmax.inputs[0].shape[2],
                                softmax.inputs[0].shape[3], softmax.inputs[0].shape[1]]
                var_outshape   = [gs.Variable(f"sf_transpose_out.1.{idx}",
                                                dtype=np.float32, shape=output_shape1)]

                # Create transpose node, to swap data between 1 and -1
                transpose1 = gs.Node(op="Transpose", name=f"sf_transpose_1.{idx}",
                                        attrs={"perm": [0,2,3,1]}, inputs=softmax.inputs,
                                        outputs=var_outshape)
                graph.nodes.append(transpose1)
                logging.debug(f"Adding transpose layer {transpose1.name} with inputs {softmax.inputs} and outputs {var_outshape}")

                softmax.inputs = transpose1.outputs

                old_softmax_outputs = copy.copy(softmax.outputs)
                softmax.outputs  = [gs.Variable(f"sf_softmax_out.{idx}",
                                                dtype=np.float32, shape=output_shape1)]

                output_shape1 = [softmax.outputs[0].shape[0], softmax.outputs[0].shape[3],
                                softmax.outputs[0].shape[2], softmax.outputs[0].shape[1]]

                # Create transpose node, to swap data between -1 and 1
                transpose2 = gs.Node(op="Transpose", name=f"sf_transpose_2.{idx}",
                                        attrs={"perm": [0,3,1,2]}, inputs=softmax.outputs,
                                        outputs=old_softmax_outputs)
                graph.nodes.append(transpose2)
                logging.debug(f"Adding transpose layer {transpose1.name} with inputs {softmax.inputs} and outputs {var_outshape}")
        
            # If softmax op occurs across 2nd axis (height access)
            elif (softmax_dimension == 2):
                logging.debug(f"Converting axis for layer {softmax.name} from dimension {softmax_dimension} to dimension -1 ")

                output_shape1 = [softmax.inputs[0].shape[0], softmax.inputs[0].shape[1],
                                softmax.inputs[0].shape[3], softmax.inputs[0].shape[2]]
                var_outshape   = [gs.Variable(f"sf_transpose_out.1.{idx}",
                                                dtype=np.float32, shape=output_shape1)]

                # Create transpose node, to swap data between -1 and 2
                transpose1 = gs.Node(op="Transpose", name=f"sf_transpose_1.{idx}",
                                        attrs={"perm": [0,1,3,2]}, inputs=softmax.inputs,
                                        outputs=var_outshape)
                graph.nodes.append(transpose1)
                logging.debug(f"Adding transpose layer {transpose1.name} with inputs {softmax.inputs} and outputs {var_outshape}")

                softmax.inputs = transpose1.outputs

                old_softmax_outputs = copy.copy(softmax.outputs)
                softmax.outputs = [gs.Variable(f"sf_softmax_out.{idx}",
                                                dtype=np.float32, shape=output_shape1)]

                output_shape1 = [softmax.outputs[0].shape[0], softmax.outputs[0].shape[1],
                                softmax.outputs[0].shape[3], softmax.outputs[0].shape[2]]

                # Create transpose node, to swap data between 2 and -1
                transpose2 = gs.Node(op="Transpose", name=f"sf_transpose_2.{idx}",
                                        attrs={"perm": [0,1,3,2]}, inputs=softmax.outputs,
                                        outputs=old_softmax_outputs)
                graph.nodes.append(transpose2)
                logging.debug(f"Adding transpose layer {transpose1.name} with inputs {softmax.inputs} and outputs {var_outshape}")


            

            
