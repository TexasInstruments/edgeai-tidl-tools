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
Modify models with batch dims to create branched inputs removing batches

################## WARNING ####################
-------------------------------------------------
This utility is at its early stage of development and the
contents are likely to change drastically in future. It is
advised not to make modifications in this file as
this may not give expected result.
"""
import logging
import onnx_graphsurgeon as gs
import onnx
from .common import find_out_layers

START_NODE_NAME = "/Mul"
END_NODE_NAME   = "/model/Transpose"


def tidl_modify_batch_dim (graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Wrapper function to modify batch input dimension to satisfy TIDL constraints
    """
    duplicate_for_multi_batch(graph, START_NODE_NAME, END_NODE_NAME)
    split_batched_inputs(graph)


def add_node(graph:gs.Graph, name:str,op:str, dtype, attrs, inputs, output_shapes=[]):
    node_outs = [gs.Variable(f"{name}_out_{i}", dtype=dtype,shape=out) for i,out in enumerate(output_shapes)]
    node = gs.Node(op=op,name=name,attrs=attrs, inputs=inputs, outputs=node_outs)
    graph.nodes.append(node)
    logging.debug(f"Adding Node {node.name}")
    return node


def _duplicate_nodes(graph:gs.Graph,nodes:list[gs.Node],count:int):
    result:list[gs.Node] = []
    nodes = [node for node in nodes if node.op != 'Constant']
    batch_size  = nodes[0].inputs[0].shape[0]
    for node in nodes:
        node_outs = [gs.Variable(f"{node.name}_{count}_out_{i}", dtype=out.dtype,shape= None) for i,out in enumerate(node.outputs)]

        nnode = gs.Node(op=node.op,name=f'{node.name}_dup_{count}',attrs=node.attrs, inputs=node.inputs, outputs=node_outs)
        if nnode.op == 'Reshape':
            nnode.inputs[1].values[0] = 1
        if nnode.op == 'Resize' and len(nnode.inputs) == 4  :
            nnode.inputs[3].values[0] = 1
        var_inputs  =  [(inp,i) for i,inp in enumerate(nnode.inputs) if isinstance(inp,gs.Variable) or inp not in graph.inputs]
        for var,i in var_inputs:
            if len(var.inputs):
                inp_node:gs.Node = var.inputs[0]
                output_index =inp_node.outputs.index(var)
                try:
                    curr_inp_node_index = nodes.index(inp_node)
                    nnode.inputs[i] = result[curr_inp_node_index].outputs[output_index]
                except:
                    pass

        result.append(nnode)
        logging.debug(f"Adding Node {nnode.name}")

    graph.nodes.extend(result)
    return result

def add_identity_for(graph:gs.Graph,output:gs.Variable):
    identity_node = add_node(graph,'identity','Identity',output.dtype,{},[output],[output.shape])
    output_nodes = []
    for out in output.outputs:
        if out == identity_node :
            continue
        output_nodes.append(out)

    for out in output_nodes:
        output.outputs.pop(out)

    identity_node.outputs[0].outputs = output_nodes

    graph.toposort()

def split_batched_inputs(graph: gs.Graph):
    """
    Convert batched input feeding into a split layer into separate branches
    linked to separate inputs instead of batched input
    """
    nodes = graph.nodes
    split_outputs = []
    #Find the first split node
    for node in nodes:
        if node.op == "Split":
            split_outputs.extend(find_out_layers(node))
            node_out_names = [node_out.name for node_out in node.outputs]
            if node.inputs[0] in graph.inputs:
                split_in = node.inputs[0]
                split_in_shape  = split_in.shape.copy()
                #Set final shape to have one batch:
                split_in_shape[0] = 1
                print(split_in_shape)
                new_inputs = []
                #iterate over split batches:
                for i in range(split_in.shape[0]):
                     new_inputs.append(gs.Variable(name= f"{split_in.name}_split_{i}", dtype= split_in.dtype, shape=split_in_shape))
                     #Update consumers of split output with new inputs:
                     #Identify the idx of input to be replaced:
                     for j in range(len(split_outputs[i].inputs)):
                         if split_outputs[i].inputs[j].name in node_out_names:
                             split_outputs[i].inputs[j] = new_inputs[i]
                #Update graph inputs:
                #Remove the graph's original input:
                for input in graph.inputs:
                    if node.inputs[0] != input:
                        new_inputs.append(input)
                graph.inputs = new_inputs
            break
    graph.cleanup().toposort()
    return graph


def duplicate_for_multi_batch(graph: gs.Graph, start_node_name:str= START_NODE_NAME, end_node_name:str= END_NODE_NAME):

    # finding out the nodes to be duplicated
    nodes = list(graph.nodes)
    node_names = [node.name for node in nodes]
    start_node_index = node_names.index(start_node_name)
    end_node_index = node_names.index(end_node_name)
    nodes = nodes[start_node_index:(end_node_index+1)]

    # splitting the input to the node
    var_inputs = [inp for inp in nodes[0].inputs if  isinstance(inp,gs.Variable) ]
    split_nodes:list[gs.Node]=[]
    for i,var in enumerate(var_inputs):
        input_shape = None
        split_node=add_node(graph,f'split_{i}','Split',var.dtype,{},[var],[input_shape  for _ in range(var.shape[0])])
        split_nodes.append(split_node)
    split_outputs = []

    # duplicating the nodes from start to end
    for i in range(nodes[0].inputs[0].shape[0]):
        result =_duplicate_nodes(graph,nodes,i+1)
        for j,var in enumerate(var_inputs):
            for k,out in enumerate(var.outputs):
                var_index = out.inputs.index(var)
                if out in nodes:
                    node_index = nodes.index(out)
                    result[node_index].inputs[var_index]=split_nodes[j].outputs[i]

        split_outputs.append(result[-1].outputs)

    # concating the output of the duplicated patterns
    output_shapes=[None  for out in nodes[-1].outputs]
    concat_outputs = []
    for i in range(len(split_outputs[0])):
        dtype = split_outputs[0][i].dtype
        concat_node=add_node(graph,f'concat_{i}','Concat',dtype,{'axis':0},inputs=[out[i] for out in split_outputs],output_shapes=[output_shapes[i]])
        concat_outputs.extend(concat_node.outputs)

    for i,out in enumerate(nodes[-1].outputs):
        out_nodes =  out.outputs
        for node in out_nodes:
            input_index = node.inputs.index(out)
            node.inputs[input_index] = concat_outputs[i]
        if out in graph.outputs:
            output_index = graph.outputs.index(out)
            graph.outputs[output_index] =concat_outputs[i]

    graph.cleanup().toposort()
    return graph


