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



def tidl_convert_softmax_axis_channel_to_width(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    The SoftMax layer with operation in the channel dimension is replaced with
    Transpose -> SoftMax -> Transpose to satisfy constraint of SoftMax layer only
    occuring in width dimension
    """
    softmaxes = [node for node in graph.nodes if node.op == "Softmax"]

    for idx, softmax in enumerate(softmaxes):
        # Dimension in which Softmax should occur
        softmax_dimension = softmax.attrs["axis"]

        # Assumes tensor with channel axis, NxCxHxW or CxHxW order
        if(len(softmax.inputs[0].shape) >= 3):

            # If softmax op occurs across channel axis
            if(softmax_dimension == len(softmax.inputs[0].shape) - 3):
                logging.debug(f"Converting axis for layer {softmax.name} from dimension {softmax_dimension} to {len(softmax.inputs[0].shape) - 1}")

                # Permutation array
                perm = list(range(len(softmax.inputs[0].shape)))
                temp = perm[-1]
                perm[-1] = perm[-3]
                perm[-3] = temp

                # New output shape from transpose1
                new_shape = copy.copy(softmax.inputs[0].shape)
                temp = new_shape[-1]
                new_shape[-1] = new_shape[-3]
                new_shape[-3] = temp

                var_outshape   = [gs.Variable(f"sf_transpose_out.1.{idx}",
                                                dtype=np.float32, shape=new_shape)]

                # Create transpose node to swap channel to width
                transpose1 = gs.Node(op="Transpose", name=f"sf_transpose_1.{idx}",
                                        attrs={"perm": perm}, inputs=softmax.inputs,
                                        outputs=var_outshape)
                graph.nodes.append(transpose1)
                logging.debug(f"Adding transpose layer {transpose1.name} with perm {perm}")

                # Modify softmax layer
                softmax.inputs = transpose1.outputs
                old_softmax_outputs = copy.copy(softmax.outputs)
                softmax.outputs  = [gs.Variable(f"sf_softmax_out.{idx}",
                                                dtype=np.float32, shape=new_shape)]

                # Create transpose node to swap width to channel
                transpose2 = gs.Node(op="Transpose", name=f"sf_transpose_2.{idx}",
                                        attrs={"perm": perm}, inputs=softmax.outputs,
                                        outputs=old_softmax_outputs)
                graph.nodes.append(transpose2)
                logging.debug(f"Adding transpose layer {transpose2.name} with perm {perm}")
        else:
            logging.critical(f"{softmax.inputs[0].name} input to {softmax.name} has no channel dim"
                                         "Unable to convert axis to channel")


def tidl_convert_softmax_axis_height_to_width(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    The SoftMax layer with operation in the height dimension is replaced with
    Transpose -> SoftMax -> Transpose to satisfy constraint of SoftMax layer only
    occuring in width dimension
    """
    softmaxes = [node for node in graph.nodes if node.op == "Softmax"]

    for idx, softmax in enumerate(softmaxes):
        # Dimension in which Softmax should occur
        softmax_dimension = softmax.attrs["axis"]

        # Assumes tensor height axis, with NxCxHxW, CxHxW, HxW order
        if(len(softmax.inputs[0].shape) >= 2):

            # If softmax op occurs across height
            if(softmax_dimension == len(softmax.inputs[0].shape) - 2):
                logging.debug(f"Converting axis for layer {softmax.name} from dimension {softmax_dimension} to {len(softmax.inputs[0].shape) - 1}")

                # Permutation array
                perm = list(range(len(softmax.inputs[0].shape)))
                temp = perm[-1]
                perm[-1] = perm[-2]
                perm[-2] = temp

                # New output shape from transpose1
                new_shape = copy.copy(softmax.inputs[0].shape)
                temp = new_shape[-1]
                new_shape[-1] = new_shape[-2]
                new_shape[-2] = temp

                var_outshape   = [gs.Variable(f"sf_transpose_out.1.{idx}",
                                                dtype=np.float32, shape=new_shape)]

                # Create transpose node to swap height to width
                transpose1 = gs.Node(op="Transpose", name=f"sf_transpose_1.{idx}",
                                        attrs={"perm": perm}, inputs=softmax.inputs,
                                        outputs=var_outshape)
                graph.nodes.append(transpose1)
                logging.debug(f"Adding transpose layer {transpose1.name} with perm {perm}")

                # Modify softmax layer
                softmax.inputs = transpose1.outputs
                old_softmax_outputs = copy.copy(softmax.outputs)
                softmax.outputs = [gs.Variable(f"sf_softmax_out.{idx}",
                                                dtype=np.float32, shape=new_shape)]

                # Create transpose node to swap width to height
                transpose2 = gs.Node(op="Transpose", name=f"sf_transpose_2.{idx}",
                                        attrs={"perm": perm}, inputs=softmax.outputs,
                                        outputs=old_softmax_outputs)
                graph.nodes.append(transpose2)
                logging.debug(f"Adding transpose layer {transpose1.name} with perm {perm}")
        else:
            logging.critical(f"{softmax.inputs[0].name} input to {softmax.name} has no height dim"
                                         "Unable to convert axis to height")



