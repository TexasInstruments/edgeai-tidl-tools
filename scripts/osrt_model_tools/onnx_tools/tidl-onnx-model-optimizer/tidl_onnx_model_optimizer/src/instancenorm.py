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
Module containing Instance Normalization layer specific functions and optimizations
"""
import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np


def tidl_convert_instancenorm_to_layernorm(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Convert the Instance Normalization to LayerNorm. Assumption that tensor is either (N, C, L) or (N, C, H, W) format
    Case 1 -> InstanceNorm1D - Directly change to LayerNorm
    Case 2 -> InstanceNorm2D - Reshape to (N, C, 1, HW) , Layernorm, Reshape back to (N, C, H, W) 
    """
    nodes = graph.nodes
    
    for node in nodes:
        if (node.op == "InstanceNormalization") and isinstance(node.inputs[1], gs.Constant):
            # found an instancenorm layer 
            is_instancenorm2d = (len(node.inputs[0].shape) == 4)
            interim_output1_reshape = gs.Variable(name= node.name + "_reshape_interim_1", dtype=np.float32)
            interim_output2_reshape = gs.Variable(name= node.name + "_reshape_interim_2", dtype=np.float32)
            layernorm_input_node = interim_output1_reshape if is_instancenorm2d else node.inputs[0]
            layernorm_output_node = interim_output2_reshape if is_instancenorm2d else node.outputs[0]
            out_dim = node.inputs[0].shape[-1]*node.inputs[0].shape[-2] if is_instancenorm2d else node.inputs[0].shape[-1]
            mul_input = node.inputs[1].values.reshape(node.inputs[0].shape[1], 1, 1) if is_instancenorm2d \
                else node.inputs[1].values.reshape(node.inputs[0].shape[1], 1)
            add_input = node.inputs[2].values.reshape(node.inputs[0].shape[1], 1, 1) if is_instancenorm2d \
                else node.inputs[2].values.reshape(node.inputs[0].shape[1], 1)
            
            if is_instancenorm2d:
                # found InstanceNorm2D - convert to reshape -> layernorm -> reshape, first reshape here
                N, C, H, W = node.inputs[0].shape
                shape1 = gs.Constant(name= node.name + "_shape_val_1", values=np.array((N, C, 1, H*W), dtype=np.int64))
                reshape_node_1 = gs.Node(name = node.name + "_reshape_1", op = "Reshape", attrs = dict({"allowzero" : 0}),
                                         inputs = [node.inputs[0], shape1], outputs = [interim_output1_reshape])
                graph.nodes.append(reshape_node_1)
                logging.debug(f"Adding Node {reshape_node_1.name}")
                
            # the original scale and bias dimension for instancenorm is different from layernorm and they are applied in another axis
            new_scale = gs.Constant(name= node.name + "_scale", values=np.ones(out_dim, dtype=np.float32))
            new_bias = gs.Constant(name= node.name + "_B", values=np.zeros(out_dim, dtype=np.float32))
            
            scale = node.inputs[1].values
            B = node.inputs[2].values
            scale_all_ones = not np.any(scale-1)
            bias_all_zero = not np.any(B)
            
            if not(scale_all_ones and bias_all_zero):
                logging.info(f"The node {node.name} has scales and biases for them, specific mul and add layers will be added.")
                interim_output1 = gs.Variable(name= node.name + "_output_interim_1", dtype=np.float32)
                interim_output2 = gs.Variable(name= node.name + "_output_interim_2", dtype=np.float32)
                layernorm_node = gs.Node(name = node.name + "_instance", op = "LayerNormalization", 
                                        attrs = dict({"axis": -1, "epsilon": node.attrs['epsilon'], "stash_type": 1}), 
                                        inputs = [layernorm_input_node, new_scale, new_bias], outputs = [interim_output1])
                graph.nodes.append(layernorm_node)
                mul_val_contant = gs.Constant(name= node.name + "_mul_constant", values=mul_input)
                mul_node = gs.Node(name = node.name + "_mul", op = "Mul", inputs = [interim_output1, mul_val_contant],
                                    outputs = [interim_output2])
                graph.nodes.append(mul_node)
                add_val_contant = gs.Constant(name= node.name + "_add_constant", values=add_input)
                add_node = gs.Node(name = node.name + "_add", op = "Add", inputs = [interim_output2, add_val_contant],
                                    outputs = [layernorm_output_node])
                graph.nodes.append(add_node)
            else:
                layernorm_node = gs.Node(name = node.name + "_instance", op = "LayerNormalization", 
                    attrs = dict({"axis": -1, "epsilon": node.attrs['epsilon'], "stash_type": 1}), 
                    inputs = [layernorm_input_node, new_scale, new_bias], outputs = [layernorm_output_node])
                graph.nodes.append(layernorm_node)
                
            logging.debug(f"Adding Node {layernorm_node.name}")
            
            if is_instancenorm2d:
                # found InstanceNorm2D - convert to reshape -> layernorm -> reshape, second reshape here
                shape2 = gs.Constant(name= node.name + "_shape_val_2", values=np.array((N, C, H, W), dtype=np.int64))
                reshape_node_2 = gs.Node(name = node.name + "_reshape_2", op = "Reshape", attrs = dict({"allowzero" : 0}),
                                         inputs = [layernorm_output_node, shape2], outputs = [node.outputs[0]])
                graph.nodes.append(reshape_node_2)
                logging.debug(f"Adding Node {reshape_node_2.name}")
                
            node.outputs.clear()
        