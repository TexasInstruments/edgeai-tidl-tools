# Copyright (c) {2023 - 2024} Texas Instruments Incorporated
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
Module containing elementwise layer (e.g. multiply, subtract, mean) specific functions and optimizations
"""
import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np


def tidl_replace_sub_with_neg_add(graph: gs.Graph,
                            onnx_graph: onnx.GraphProto):
    '''
    Sub node is not supported, but this can be replaced (less efficiently) with negation and add
    '''

    for node in graph.nodes:

        if node.op == "Sub":
            #Sub -> C = A-B. inputs=[A,B]
            A, B = node.inputs
            C = node.outputs[0]
            broadcast_neg = 1
            if A.shape != B.shape:
                logging.warning('This is a broadcasted node; not yet supported for Sub replacment')
                continue

            logging.debug(f'Replacing Sub node {node.name} with Multiply-Add')
            #Create Mul node, and use B as one input and -1 (constant; broadcasted) as the other
            base_name = node.name
            mul_name = base_name + '_Mul'
            #We will broadcast -1 multiplication across the whole input B
            neg_values = np.ndarray((1), dtype=B.dtype)
            neg_values[0] = -1
            negation_tensor = gs.Constant(mul_name + '/neg', neg_values)
            negation_output = gs.Variable(mul_name + '/negative', dtype=B.dtype, shape=B.shape)
            mul_node = gs.Node('Mul', mul_name, {}, [B, negation_tensor], [negation_output])

            #Create add node for A + (-B) 
            add_name = base_name + '_Add'
            add_node = gs.Node('Add', add_name, {} , [A, negation_output], outputs=[C])

            node.outputs.pop(0)

            logging.debug(f'Add Multiply node {mul_node.name}')
            logging.debug(f'Add "Add" node {add_node.name}')
            graph.nodes.append(mul_node)
            graph.nodes.append(add_node)
            #old Sub node will be removed with graph.cleanup. 
            #The new nodes will require graph.toposort()

def tidl_replace_mean_with_eltwise(graph: gs.Graph,
                            onnx_graph: onnx.GraphProto):
    '''
    Elementwise Mean node is not supported, but we can emulate with add -> multiply

    Currently only supports Mean between two input tensors, but should be trivial to extend
    Note that quantization may impact this layer, especially for many inputs
    '''
    
    for node in graph.nodes:
        if node.op == "Mean":
            if len(node.inputs) != 2:
                logging.warning(f'Mean between arbitrary number of inputs is not supported; only 2. Skip node {node.name}')
                continue
            logging.debug(f'Replacing Mean ({node.name}) of two inputs with representative Add->Multiply elementwise layers')
            base_name = node.name
            A, B = node.inputs[0:2]
            output = node.outputs[0]

            if A.shape != B.shape:
                logging.warning('Detected non-elementwise operation / broadcasting -- this is not supported')
                continue

            add_name = base_name + '_Add'
            sum_tensor = gs.Variable(base_name + '/Sum', dtype=A.dtype, shape=A.shape)
            add_node = gs.Node('Add', add_name, {} , [A, B], outputs=[sum_tensor])

            div_name = base_name + '_Mul_by_half'
            divisor_tensor_name = div_name + '/divisor'

            if  'float' not in str(A.dtype):
                logging.warning(f'potential issue with dtype {str(A.dtype)}; this may cause problems with quantization')

            div_by_two_values = np.ndarray((1), dtype=A.dtype)
            div_by_two_values[0] = 1/2
            divisor_tensor = gs.Constant(divisor_tensor_name, div_by_two_values)

            mul_node = gs.Node('Mul', div_name, {}, [sum_tensor, divisor_tensor], [output])

            node.outputs.clear()

            logging.debug(f'Add Multiply node {mul_node.name}')
            logging.debug(f'Add "Add" node {add_node.name}')
            graph.nodes.append(mul_node)
            graph.nodes.append(add_node)

            #cleanup and toposort graph to fully apply changes

def tidl_support_broadcast_ops_constant_input(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    '''
    Replaces the constants in elt-wise arithmetic operators to prevent multidimensional broadcast or cross-broadcast
    '''
    change_multi_broadcast = True
    for node in graph.nodes:
        if node.op in ('Add','Sub','Mul','Div'):
            inp_types = [type(inp) for inp in node.inputs]
            if not(gs.Variable in inp_types and gs.Constant in inp_types):
                continue
            inp = node.inputs[inp_types.index(gs.Variable)]
            const = node.inputs[inp_types.index(gs.Constant)]
            if not (isinstance(inp, gs.Variable) and isinstance(const, gs.Constant)):
                continue
            inp_shape = inp.shape
            old_shape = const_shape = const.shape
            const_value = const.values

            if len(inp_shape) <= 3 or len(const_shape)==0:
                continue
            new_shape = [1 for _ in inp_shape[:-(len(const_shape))]]+ list(const_shape)
            const_value = np.reshape(const_value, new_shape)
            const_shape = new_shape

            broadcast_dims = []
            broadcast_dims = [i for i, x in enumerate(const_shape) if x == 1 and inp_shape[i] != 1]
            
            ## to check any broad cast from variable with constant will happen or not
            cross_broadcast_dims = [i for i, x in enumerate(inp_shape) if x == 1 and const_shape[i] != 1]

            except_last_3_dims = list(range(len(inp_shape)-3))

            if len(broadcast_dims)==1:
                if broadcast_dims[0] in except_last_3_dims or len(cross_broadcast_dims)>0:
                    ## cross broadcast and broadcasting outside C, H, W is not allowed
                    logging.debug(f"Constant {const.name} at dim {broadcast_dims[0]} from {const_shape[broadcast_dims[0]]} to {inp_shape[broadcast_dims[0]]}")
                    const_value = np.concatenate([const_value for _ in range(inp_shape[broadcast_dims[0]])], axis=broadcast_dims[0])
                else:
                    continue
            elif broadcast_dims:
                broadcast_dims_in_last_3 = [dim for dim in broadcast_dims if dim not in except_last_3_dims]
                for dim in broadcast_dims:
                    if dim in except_last_3_dims or len(cross_broadcast_dims)>0:
                        ## cross broadcast and broadcasting outside C, H, W is not allowed
                        logging.debug(f"Constant {const.name} at dim {dim} from {const_shape[dim]} to {inp_shape[dim]}")
                        const_value = np.concatenate([const_value for _ in range(inp_shape[dim])], axis=dim)
                    else:
                        if change_multi_broadcast and len(broadcast_dims_in_last_3) > 1:
                            ## to preserve atleast one dimension to broadcast
                            logging.debug(f"Constant {const.name} at dim {dim} from {const_shape[dim]} to {inp_shape[dim]}")
                            const_value = np.concatenate([const_value for _ in range(inp_shape[dim])], axis=dim)
                            broadcast_dims_in_last_3.remove(dim)
                        else:
                            continue
            else:
                continue
                
            node.inputs[inp_types.index(gs.Constant)].values = const_value
            logging.debug(f"Modified input {const.name} from {old_shape} to {const.shape} Node {node.name}")

