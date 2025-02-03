# Copyright (c) {2025 - 2026} Texas Instruments Incorporated
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

import onnx_graphsurgeon as gs
import numpy as np
import onnx


def find_attentions(graph: gs.Graph):
    r'''
    finds the hf attention blocks in the graph 
                        input(split)
                        / | \
                       /  |  \
                      q   k   v   - branches
                      |   |   |
                      \   /   |
                      matmul  |
                        |     |
                     process  |
                        |     |
                     softmax  |
                          \   /
                          matmul
                            |
    this function assumes that q,k,v and process are straight chain of operations their is no dependencies on nodes outside of the block
    '''
    
    attentions = []
    for node in graph.nodes:
        if node.op != 'Softmax' or len(node.outputs[0].outputs) > 1:
            continue
        softmax = node
        is_attention = True
        while True:
            if len(node.inputs[0].inputs) == 0 :
                break
            prev_node = [inp for inp in node.inputs if isinstance(inp, gs.Variable)][0].inputs[0]
            if prev_node.op == 'MatMul':
                node1 = node
                break
            if len([inp for inp in prev_node.inputs if isinstance(inp, gs.Variable)]) != 1:
                is_attention = False
                break
            node = prev_node
        if not is_attention:
            continue
        
        if len(node1.inputs[0].inputs) == 0  or (node2 := node1.inputs[0].inputs[0]).op != 'MatMul':
            continue
        if not all(isinstance(inp, gs.Variable) for inp in node2.inputs):
            continue
        if (node3 := softmax.outputs[0].outputs[0]).op != 'MatMul':
            continue
        matmul1, matmul2 = node2, node3
        
        qkv = [inp.inputs[0] for inp in matmul1.inputs]+[matmul2.inputs[1].inputs[0]]
        qkv_branches = [[node] for node in qkv ]
        i=0 
        branch_progress = [True for _ in qkv_branches]
        while True:
            curr_branch = qkv_branches[i]
            rest_branches = [(i,branch) for i,branch in enumerate(qkv_branches) if branch is not curr_branch]
            if all(curr_branch[-1] is branch[1][-1] for branch in rest_branches):
                break
            for j,branch in rest_branches:
                if branch[-1] is curr_branch[-1]:
                    branch_progress[i] = False
                    branch_progress[j] = False
            if all(not branch_pgs for branch_pgs in branch_progress):
                break 
            if not branch_progress[i]:
                i = (i+1)%3
                continue
            curr_node = curr_branch[-1]
            var_inputs = [inp for inp in curr_node.inputs if isinstance(inp, gs.Variable)]
            if len(var_inputs) != 1:
                is_attention = False
                break
            curr_branch.append(var_inputs[0].inputs[0])
            i = (i+1)%3
        
        if not is_attention:
            continue
        
        attentions.append((*qkv_branches, matmul1, softmax, matmul2))
        
    return attentions


def tidl_merge_adds_between_matmul_and_softmax(graph:gs.Graph, onnx_graph:onnx.GraphProto, attentions=None):
    '''
    this function takes care of ADD -> Reshape -> ADD -> Reshape pattern observed in attention blocks in swin models from Hugging Face
    it merges the two ADD nodes into one and updates the shape of the constant node accordingly, if possible 
    
    This function assumes those ADD nodes between the matmul and softmax nodes in the attention block do not take more than one variable inputs, 
    i.e. it's a straight chain. 
    '''
    
    attentions = attentions or find_attentions(graph)
    for q_branch, k_branch, v_branch, matmul1, softmax, matmul2 in attentions:
        q_branch.reverse()
        k_branch.reverse()
        v_branch.reverse()
        node = softmax
        intermediate_nodes = []
        while True:
            if len(node.inputs[0].inputs) == 0 :
                break
            prev_node = [inp for inp in node.inputs if isinstance(inp, gs.Variable)][0].inputs[0]
            intermediate_nodes.append(prev_node)
            if prev_node.op == 'MatMul':
                node1 = node
                break
            if len([inp for inp in prev_node.inputs if isinstance(inp, gs.Variable)]) != 1:
                break
            node = prev_node
        intermediate_ops = [node.op  for node in intermediate_nodes]
        pattern = ['Reshape', 'Add', 'Reshape', 'Add']
        start_index = None
        for i in range(len(intermediate_ops) - len(pattern) + 1):
            if intermediate_ops[i:i+len(pattern)] == pattern:
                start_index = i
                break
        if start_index is  None:
            continue
        add2 = intermediate_nodes[start_index+1]
        add1 = intermediate_nodes[start_index+3]
        index, const1 = [(i,inp) for i,inp in enumerate(add1.inputs) if isinstance(inp, gs.Constant)][0]
        const2 = [inp for inp in add2.inputs if isinstance(inp, gs.Constant)][0]
        if const1.shape[-2:] != const2.shape[-2:]:
            continue
        old_shape = list(const1.shape)
        if len(const1.shape) > len(const2.shape):
            const1, const2 = const2, const1
        new__shape_const1 = list(const1.shape).copy()
        while len(new__shape_const1) < len(const2.shape):
            new__shape_const1 = [1] + new__shape_const1
        
        const1.values = np.reshape(const1.values, new__shape_const1)
        for axis in range(len(new__shape_const1)-2):
            if new__shape_const1[axis] == const2.shape[axis]:
                continue
            if new__shape_const1[axis] == 1:
                const1.values = np.concatenate([const1.values for _ in range((const2.shape[axis]))], axis=axis)
            if const2.shape[axis] == 1:
                const2.values = np.concatenate([const2.values for _ in range((new__shape_const1[axis]))], axis=axis)
        const1.values = const1.values + const2.values
        const1.values = np.reshape(const1.values, const1.shape[-len(old_shape):])
        add1.inputs[index] = const1
        output_node = intermediate_nodes[start_index-1] if start_index > 0 else softmax
        input_index = [i for i in range(len(output_node.inputs)) if isinstance(output_node.inputs[i], gs.Variable) ][0]
        output_node.inputs[input_index] =add1.outputs[0]

def tidl_optimize_hf_attention(graph:gs.Graph, onnx_graph:onnx.GraphProto):
    r'''
    finds attention blocks and optimizes them
    '''
    attentions = find_attentions(graph)    
    
    tidl_merge_adds_between_matmul_and_softmax(graph, onnx_graph, attentions)
    
    for q_branch, k_branch, v_branch, matmul1, softmax, matmul2 in attentions:
        if q_branch[0].op == 'Split':
            split = q_branch[0]
            split.outputs.clear()
            split_inp = split.inputs[0]
            q_split, k_split, v_split =  split.inputs[1].values
            branches_upto_transposes = []
            for branch in  (q_branch, k_branch, v_branch):
                new_branch = []
                for node in branch:
                    if node is split:
                        continue
                    new_branch.append(node)
                    if node.op == 'Transpose':
                        break
                branches_upto_transposes.append(new_branch)
            q_branch, k_branch, v_branch = branches_upto_transposes
            assert len(q_branch) == len(k_branch) == len(v_branch)
            new_input = split.inputs[0]
            change_split_axis = False
            
            if all( branch[0].op == 'Add'  for branch in (q_branch, k_branch, v_branch)):
                add_nodes = [branch[0] for branch in (q_branch, k_branch, v_branch)]
                for i in (0,1,2):
                    add_nodes[i].outputs[0].inputs[0] = split
                add_node = add_nodes[0]
                add_node.inputs[1] = split.inputs[0]
                add_node.inputs[0].values = np.concatenate([node.inputs[0].values for node in add_nodes])
                add_out = gs.Variable(f'{add_node.name}_out',dtype=add_node.inputs[1].dtype, shape = add_node.inputs[1].shape)
                add_node.outputs.append(add_out)
                split.inputs[0] = add_out
                for branch in (q_branch, k_branch, v_branch):
                    branch.pop(0)
                new_input = add_out               
                
            if all( branch[0].op == 'Reshape'  for branch in (q_branch, k_branch, v_branch)):
                reshape_nodes = [branch[0] for branch in (q_branch, k_branch, v_branch)]
                split.outputs.clear()
                for i in (0,1,2):
                    reshape_nodes[i].outputs[0].inputs[0] = split
                reshape_node = reshape_nodes[0]
                change_split_axis = (q_split==k_split==v_split)
                output_shape = split.outputs[0].shape
                reshape_node.inputs[0] =  new_input
                if change_split_axis:
                    split_dim = output_shape[-2]
                    output_shape[-2] = 3*output_shape[-2]
                else:
                    split_axis = split.attrs['axis']
                    output_shape[split_axis] = sum([out.shape[split_axis] for out in split.outputs])
                reshape_node.inputs[1].values = np.array(output_shape, reshape_node.inputs[1].values.dtype)
                reshape_out = gs.Variable(f'{reshape_node.name}_out',dtype=reshape_node.inputs[0].dtype, shape=output_shape)
                reshape_node.outputs.append(reshape_out)
                split.inputs[0] = reshape_out
                for branch in (q_branch, k_branch, v_branch):
                    branch.pop(0)
                if change_split_axis:
                    split.attrs['axis'] = -2 
                    split.inputs[1].values = np.array([split_dim,split_dim,split_dim], split.inputs[1].values.dtype)
                new_input = reshape_out
                
            if all( branch[0].op == 'Transpose'  for branch in (q_branch, k_branch, v_branch)):
                transpose_nodes = [branch[0] for branch in (q_branch, k_branch, v_branch)]
                split.outputs.clear()
                for i in (0,1,2):
                    if i == 1:
                        split.outputs.append(transpose_nodes[i].inputs[0])
                        perm = list(range(len(transpose_nodes[i].outputs[0].shape)))
                        perm[-2:] = perm[-2:][::-1]
                        transpose_nodes[i].attrs['perm'] = perm
                    else:
                        transpose_nodes[i].outputs[0].inputs[0] = split
                transpose_node = transpose_nodes[0]
                transpose_node.inputs[0] =  new_input
                transpose_out = gs.Variable(f'{transpose_node.name}_out',dtype=transpose_node.inputs[0].dtype, shape=None)
                transpose_node.outputs.append(transpose_out)
                split.inputs[0] = transpose_out
                for branch in (q_branch, k_branch, v_branch):
                    branch.pop(0)
                if change_split_axis or split.attrs['axis'] in (-2, len(transpose_node.inputs[0].shape)-2):
                    split.attrs['axis'] = -3
                elif split.attrs['axis'] in (-3, len(transpose_node.inputs[0].shape)-3):
                    split.attrs['axis'] = -2
            for out in  split.outputs:
                out.shape = None
        