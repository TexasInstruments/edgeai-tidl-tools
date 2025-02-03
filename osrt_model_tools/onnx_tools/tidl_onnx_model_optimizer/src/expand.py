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
Module containing Expand layer specific functions and optimizations
"""
import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np


def tidl_convert_expand_to_reshape_and_concat(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Convert the Expand layer which can do multiple operations such as reshape and repeat a once. For example :
    Input_Tensor (5x5) -> |Expand to 1,1,5,5| -> Output_Tensor(1x1x5x5) or 
    Input_Tensor (1x3x1x5x64) -> |Expand to 1,3,3,5,64| -> Output_Tensor(1x3x3x5x64) or similar
    This can be replaced by reshape or concat or combination of these. 
    Reshape - if the length of expected shape does not match original 
    Concat - if the input tensor needs to be repeated in a particular dimension 
    """
    nodes = graph.nodes
    
    for node in nodes:
        if (node.op == "Expand"):
            target_shape = node.inputs[1].values
            input_shape = node.inputs[0].shape
            final_output_node = node.outputs[0]
            reshp = None
            # add reshape over here
            if len(target_shape) != len(input_shape):
                expand_dim_val = len(target_shape) - len(input_shape)
                new_shape = np.array([], dtype=np.int64)
                for i in range(len(target_shape)):
                    if i < expand_dim_val:
                        new_shape = np.append(new_shape, 1)
                    else:
                        new_shape = np.append(new_shape, input_shape[i-expand_dim_val])
                #
                
                mismatched_shape = np.not_equal(new_shape, target_shape)
                if mismatched_shape.any(): # if mismatched shapes are present, then new layers are added to the network
                    reshape_output_node = gs.Variable(name= node.name + "_interim_reshape", dtype=np.float32)
                else:
                    reshape_output_node = node.outputs[0]

                reshp_shape = gs.Constant(name = f"{node.name}_Reshape_shape", values = new_shape)
                reshp = gs.Node(name = f"{node.name}_Reshape", op= "Reshape",
                                inputs = [node.inputs[0], reshp_shape], outputs = [reshape_output_node])
                logging.debug(f"Adding Node {reshp.name}")
                graph.nodes.append(reshp)

                input_shape = new_shape
                node.outputs.clear()

            # tensor repeat using concat considering that len(target shape) == len(input shape)
            if len(target_shape) == len(input_shape):
                mismatched_shape = np.not_equal(input_shape, target_shape)
                num_mismatched_axes = np.sum(mismatched_shape)
                num_expanded_axis = 0
                input_node = node.inputs[0] if reshp is None else reshape_output_node
                for i in range(len(target_shape)):
                    if mismatched_shape[i]:                            
                        if num_expanded_axis+1 == num_mismatched_axes:
                            output_node = final_output_node
                        else:
                            output_node = gs.Variable(name= node.name + "_interim_{}".format(i), dtype=np.float32)
                        target_len = target_shape[i]
                        node_inputs = [input_node]*target_len
                        concat_node = gs.Node(name = node.name + "_concat_{}".format(i), op = "Concat",
                                              attrs = dict({"axis" : i-len(target_shape)}),
                                              inputs = node_inputs, outputs = [output_node])
                        logging.debug(f"Adding Node {concat_node.name}")
                        graph.nodes.append(concat_node)
                        num_expanded_axis += 1
                        input_node = output_node
                    #
                #
            #
            else:
                logging.info(f"The target_shape : {target_shape} and input_shape : {input_shape} are not matching for {node.name}")

            node.outputs.clear()
            #
        #
    #
        