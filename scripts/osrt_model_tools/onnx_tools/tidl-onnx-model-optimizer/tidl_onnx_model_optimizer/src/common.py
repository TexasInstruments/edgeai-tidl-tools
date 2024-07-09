# Copyright (c) {20 -23 2024} Texas Instruments Incorporated
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
Common utlity functions and useful graph algorithms
"""
from typing import List
import onnx_graphsurgeon as gs
import onnx

class UniqueIdGenerator:
    """
    Unique Id for making name uniques
    """
    def __init__(self):
        self.id = 0

    def get_id (self) -> int:
        """return unique id"""
        self.id += 1
        return self.id

id_generator = UniqueIdGenerator()

def find_in_layers (curr_layer: gs.Node) -> List[gs.Node]:
    """
    Return all input nodes to a given node
    """
    in_layers = list()
    for inp in curr_layer.inputs:
        in_layers.extend(inp.inputs)
    return in_layers

def find_in_layer (curr_layer: gs.Node, idx: int) -> gs.Node|None:
    """
    Return idx-th input node
    if not present returns None
    """
    if len(find_in_layers(curr_layer)) > idx:
        return find_in_layers(curr_layer)[idx]
    else:
        return None


def find_out_layers (curr_layer: gs.Node) -> List[gs.Node]:
    """
    Return all input nodes to a given node
    """
    out_layers = list()
    for outp in curr_layer.outputs:
        out_layers.extend(outp.outputs)
    return out_layers

def find_out_layer (curr_layer: gs.Node, idx: int) -> gs.Node|None:
    """
    Return idx-th output node
    if not present returns None
    """
    if len(find_out_layers(curr_layer)) > idx:
        return find_out_layers(curr_layer)[idx]
    else:
        return None


def find_node_idx (node: gs.Node, graph: gs.Graph) -> int:
    """
    Return node idx in graph.nodes
    """
    for idx, n in enumerate(graph.nodes):
        if node == n:
            return idx
    return -1



def is_ancestor_util (p_node: gs.Node, c_node: gs.Node, graph: gs.Node, visited: List[int]) -> bool:
    """
    Called from wrapper function, recursive function to check
    if p_node is a predecessor of c_node
    """
    # base case: no parents to search
    if len(c_node.inputs) == 0:
        return False

    # found in immediate parents
    if p_node in find_in_layers(c_node):
        return True

    # check for all the layers which is input to this layer
    inp_layers = find_in_layers(c_node)
    for inp in inp_layers:
        if visited[find_node_idx(inp, graph)] == 0:
            visited[find_node_idx(inp, graph)] = 1
            if is_ancestor_util(p_node, inp, graph, visited):
                return True

    return False


def is_ancestor (p_node: gs.Node, c_node: gs.Node, graph: gs.Node) -> bool:
    """
    Return true if p_node is a ancestor of c_node
    """
    # considering every node is ancestor of itself
    if p_node == c_node:
        return True
    visited = [0]*len(graph.nodes)
    return is_ancestor_util(p_node, c_node, graph, visited)


def remove_node (node: gs.Node):
    """
    Remove node from graph
    """
    for inp_node in find_in_layers(node):
        inp_node.outputs = node.outputs
        node.outputs.clear()

def is_first_node(node: gs.Node) -> bool:
    """
    Return True if a node takes input from model input
    """
    if len(find_in_layers(node)) == 0:
        return True
    return False

def is_single_const_single_var_input(node: gs.Node):
    """
    Return true if the node has input 1 constant and 1 variable
    """
    return  (len(node.inputs) == 2 and \
            (isinstance(node.inputs[0], gs.Variable) and isinstance(node.inputs[1], gs.Constant)) or \
            (isinstance(node.inputs[1], gs.Variable) and isinstance(node.inputs[0], gs.Constant)))

def bordered(text):
    """
    Print bordered text banner
    """
    lines = text.splitlines()
    width = max(len(s) for s in lines)
    res = ['┌' + '─' * width + '┐']
    for s in lines:
        res.append('│' + (s + ' ' * width)[:width] + '│')
    res.append('└' + '─' * width + '┘')
    return '\n'.join(res)

def reset_shape_inference(onnx_graph:onnx.GraphProto):
    '''
    Clear all value_info entries that hold shape inference information
    '''
    while len(onnx_graph.value_info) > 0: 
        onnx_graph.value_info.pop()
    return onnx_graph

