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
Module containing QDQ specific functions and optimizations
"""
import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np
from .common import find_in_layers, find_out_layers

def find_prev_dequantize(node, iter_idx):
    if node.op == "DequantizeLinear":
        return node
    if iter_idx < 0:
        return -1 
    for n_id in find_in_layers(node):
        if n_id.op == "DequantizeLinear":
            return n_id
        else:
            dequant_node = find_prev_dequantize(n_id, iter_idx-1)
            if dequant_node != -1:
                return dequant_node
    return -1
    

def tidl_add_bias_qdq(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Add the bias QDQ nodes in the layers which have bias already. This is done as the scales used for bias 
    are basically Weight_scale*input_act_scale. This occurs in the case of Conv and Gemm layers.
    """
    nodes = graph.nodes
    
    for node in nodes:
        if (node.op in ["Gemm", "Conv"]):
            if len(node.inputs)<3:
                logging.info(f"The node {node.name} does not have a bias, skipping.")
                continue 
            if not(find_in_layers(node)[1].op == "DequantizeLinear" and isinstance(node.inputs[2], gs.Constant)):
                logging.info(f"The node {node.name} does not have a weight quantized or the bias is quantized as well, skipping.")
                return
            
            input_act = find_prev_dequantize(find_in_layers(node)[0], 4)
            if not(isinstance(input_act, gs.Node)):
                logging.info(f"The node {node.name} does not have an input activation scale, the model is not quantized, skipping.")
                return 
            input_act_scale = input_act.inputs[1].values # an array of single value
            
            weight_act = find_in_layers(node)[1]
            weight_act_scale = weight_act.inputs[1].values
            
            bias_scale = input_act_scale*weight_act_scale            
            bias_zero_point = np.zeros_like(bias_scale, dtype=np.int32)
            bias_scale_inp = gs.Constant(name= node.name + "_quantize_scale", values=bias_scale)
            bias_zero_point_inp = gs.Constant(name= node.name + "_quantize_zero_point", values=bias_zero_point)
            interim_output1 = gs.Variable(name= node.name + "_interim", dtype=np.float32)
            
            bias_input_quantized = np.round(np.divide(node.inputs[2].values, bias_scale)).astype(np.int32)
            bias_quantized = gs.Constant(name= node.name + "_quantized_bias", values=bias_input_quantized)
            
            dequantize_node = gs.Node(name = node.name + "_dequantize_bias", op = "DequantizeLinear", 
                                    attrs = dict({"axis": 0}), 
                                    inputs = [bias_quantized, bias_scale_inp, bias_zero_point_inp], outputs = [interim_output1]) 
            node.inputs[2] = interim_output1
            
            graph.nodes.append(dequantize_node)
            logging.debug(f"Adding Node {dequantize_node.name}")


def tidl_remove_quantize_initializer(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Remove the float initializer of weights (sometimes FP32->Q->DQ-> occurs instead of INT8->DQ-> ) 
    """
    nodes = graph.nodes
    
    for node in nodes:
        if (node.op == "QuantizeLinear") and isinstance(node.inputs[0], gs.Constant): #found the needed weight
            input_weight = node.inputs[0].values
            weight_shape = input_weight.shape
            if len(weight_shape)==0:
                target_shape = (-1)
            else:
                target_shape = np.ones_like(weight_shape)
                target_shape[0] = -1
            input_scale =  node.inputs[1].values
            if len(node.inputs)>2:
                input_zero_point = node.inputs[2].values
            else:
                input_zero_point = np.zeros_like(input_scale, dtype=np.uint8)
            dequant_node = find_out_layers(node)[0]
            zero_point_dtype = dequant_node.inputs[2].values.dtype if len(dequant_node.inputs)>2 else np.uint8

            # directly using "astype" has a circular (not clipping implementation)
            quant_weight = (np.round(np.divide(input_weight, input_scale.reshape(target_shape))) + input_zero_point.reshape(target_shape))
            if zero_point_dtype == np.int8:
                quant_weight = np.clip(quant_weight, -128, 127)
            elif zero_point_dtype == np.uint8:
                quant_weight = np.clip(quant_weight, 0, 255)
            else:
                logging.info(f"{zero_point_dtype} is not a currently accepted dtype format for removing initialisers. Only int8 or uint8 are supported")
            quant_weight = quant_weight.astype(zero_point_dtype)

            if quant_weight.shape[0]==1 and len(quant_weight.shape)==1:
                dequant_node.inputs[0] = gs.Constant(name= node.name + "_quantized", values=np.array(quant_weight[0], dtype=zero_point_dtype))
            else:
                dequant_node.inputs[0] = gs.Constant(name= node.name + "_quantized", values=quant_weight)
            node.outputs.clear()
            logging.debug(f"Changing input of node {dequant_node.name}")



def tidl_remove_duplicate_quantize_dequantize(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Remove the duplicate Q-DQ layers that occur in the quantized model, will just keep the first Q-DQ in the bunch of layers
    """
    nodes = graph.nodes
    
    for node in nodes:
        if (node.op == "QuantizeLinear") and isinstance(node.inputs[0], gs.Constant): #found the needed weight
            logging.info(f"remove_duplicate_quantize_dequantize is not yet implemented \n")
            pass
            # input_weight = node.inputs[0].values
            # weight_shape = input_weight.shape
            # if len(weight_shape)==0:
            #     target_shape = (-1)
            #     continue
            # else:
            #     target_shape = np.ones_like(weight_shape)
            #     target_shape[0] = -1
            # input_scale =  node.inputs[1].values
            # if len(node.inputs)>2:
            #     input_zero_point = node.inputs[2].values
            # else:
            #     input_zero_point = np.zeros_like(input_scale, dtype=np.uint8)
            # dequant_node = find_out_layers(node)[0]
            # zero_point_dtype = dequant_node.inputs[2].values.dtype if len(dequant_node.inputs)>2 else np.uint8
            # quant_weight = np.round(np.divide(input_weight, input_scale.reshape(target_shape)) + input_zero_point.reshape(target_shape)).astype(zero_point_dtype)
            # if quant_weight.shape[0]==1 and len(quant_weight.shape)==1:
            #     dequant_node.inputs[0] = gs.Constant(name= node.name + "_quantized", values=np.array(quant_weight[0], dtype=zero_point_dtype))
            # else:
            #     dequant_node.inputs[0] = gs.Constant(name= node.name + "_quantized", values=quant_weight)
            # node.outputs.clear()
            # logging.debug(f"Changing input of node {dequant_node.name}")

