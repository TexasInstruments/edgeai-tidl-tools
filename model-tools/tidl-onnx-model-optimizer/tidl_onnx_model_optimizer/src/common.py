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
import logging

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
    Return all output nodes to a given node
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
    # for inp_node in find_in_layers(node):
    #     inp_node.outputs = node.outputs
    #     node.outputs.clear()
    for inp_node in find_in_layers(node):
        for i, opt in enumerate(inp_node.outputs):
            if opt.outputs[0] == node:
                inp_node.outputs[i] = node.outputs[0]
        node.outputs.clear()

def is_first_node(node: gs.Node) -> bool:
    """
    Return True if a node takes input from model input
    """
    if len(find_in_layers(node)) == 0:
        return True
    return False

def is_end_node(node: gs.Node) -> bool:
    """
    Return True if a node is an output node of the model
    """
    if len(find_out_layers(node)) == 0:
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

def has_unk_axis(inp : gs.Variable):
    shp = inp.shape
    if shp is None:
        return True
    for iter in shp:
        if not isinstance(iter, int):
            return True
    return False
    
def format_logger (log_level):
    """
    Format logger
    """

    logging.basicConfig(format='[%(levelname)s]:%(message)s')
    # colored logs
    yellow  = "\x1b[33;20m"
    red     = "\x1b[31;1m"
    reset   = "\x1b[0m"
    logging.addLevelName(logging.WARNING, yellow + logging.getLevelName(logging.WARNING) + reset)
    logging.addLevelName(logging.CRITICAL, yellow + logging.getLevelName(logging.WARNING) + reset)
    logging.addLevelName(logging.ERROR, red + logging.getLevelName(logging.ERROR) + reset)
    # set log level
    if log_level == "info":
        logging.getLogger().setLevel(logging.INFO)
    elif log_level == "debug":
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        print(f"Unknown log level {log_level}")


def extract_constant_values(tensor:gs.Tensor, graph:gs.Graph):
    values = None
    if type(tensor) == gs.Constant : 
        values = tensor.values
    elif type(tensor) == gs.Variable : 
        constant_as_variable = tensor
        constant_node = None

        for node in graph.nodes:
            for output_tensor in node.outputs:
                if  output_tensor == constant_as_variable:
                    constant_node = node
                    break
            
            if constant_node is not None: 
                values = constant_node.attrs['value'].values
                break

    return values

    
def reset_shape_inference(onnx_graph:onnx.GraphProto):
    '''
    Clear all value_info entries that hold shape inference information
    '''
    logging.debug('Resetting inferred shapes for graph')
    while len(onnx_graph.value_info) > 0: 
        onnx_graph.value_info.pop()
    return onnx_graph

def tidl_remove_duplicates(graph:gs.Graph, onnx_graph:onnx.GraphProto, do_cleanup=True):
    '''
    Some nodes are simply duplicates of each other. 
    There is no need to process these, and we can reuse the outputs of one for all of them
    '''

    #find set of nodes that we can remove and replace with an existing one
    replacement_node_pairs = []
    for i, node_i in enumerate(graph.nodes):
        
        for j, node_j in enumerate(graph.nodes):
            if node_i.op != node_j.op: continue

            nodes_to_remove = list(map(lambda x: x[1], replacement_node_pairs))

            if node_i == node_j or node_i in nodes_to_remove: 
                continue # skip itself or if node is already to be replaced

            if node_i.inputs == node_j.inputs and \
                node_i.attrs == node_j.attrs and \
                len(node_j.inputs) != 0:
                # check if there are any inputs/attributes that are not identical
                # hang onto the nodes we will remove/replace. We should not remove them while iterating
                # If no inputs, meaning a constant/initializer, skip
                replacement_node_pairs.append((node_i, node_j)) #(node to keep as replacement, node to remove)           
            
    for n in replacement_node_pairs: 
        removal_node = n[1]
        keep_node = n[0]
        logging.debug(f'Remove duplicate node {removal_node.name} and replace with equivalent {keep_node.name}')
        removal_outputs = removal_node.outputs
        keep_outputs = keep_node.outputs

        # for each output in the node to remove, find the consuming nodes and change their input to use the output that we will keep
        out_layers = find_out_layers(removal_node)

        for layer in out_layers:
            for i, in_tensor in enumerate(layer.inputs):
                for j, out_tensor in enumerate(removal_outputs):
                    if in_tensor == out_tensor:
                        
                        replacement_tensor = keep_outputs[j]
                        layer.inputs[i] = replacement_tensor

        #clear the outputs to that graph.cleanup() will remove them
        removal_node.outputs.clear()

    if do_cleanup:
        graph.cleanup().toposort()

def find_consumers(node: gs.Node, graph: gs.Graph) -> list:
    consumers = []
    node_output_names = set([out.name for out in node.outputs])
    for n in graph.nodes:
        for inp in n.inputs:
            if inp.name in node_output_names:
                consumers.append(n)
                break
    return consumers

def is_constant_node(node: gs.Node) -> bool:
    """
    Return True if all outputs of the node are gs.Constant, else False.
    """
    return all(isinstance(out, gs.Constant) for out in node.outputs)

def get_node_names_by_op(graph: gs.Graph, op_type: str) -> list:
    """
    Return a list of node names in the graph that match the given operator type.
    """
    return [node.name for node in graph.nodes if node.op == op_type]

def insert_subgraph_between_tensors(graph: gs.Graph, before_tensor_name: str, after_tensor_name: str, new_subgraph: gs.Graph):
    """
    Replace the subgraph between before_tensor_name and after_tensor_name with new_subgraph.
    Assumes user ensures input/output compatibility.
    """
    before_tensor = next((t for t in graph.tensors().values() if t.name == before_tensor_name), None)
    after_tensor = next((t for t in graph.tensors().values() if t.name == after_tensor_name), None)

    if before_tensor is None:
        logging.error(f"Tensor '{before_tensor_name}' not found in the graph.")
        return False
    if after_tensor is None:
        logging.error(f"Tensor '{after_tensor_name}' not found in the graph.")
        return False
    
    subgraph_input_tensor = new_subgraph.inputs[0]
    subgraph_output_tensor = new_subgraph.outputs[0]
    assert tuple(before_tensor.shape) == tuple(subgraph_input_tensor.shape), \
    f"Shape mismatch: before_tensor shape {before_tensor.shape} != subgraph input shape {subgraph_input_tensor.shape}"
    assert tuple(after_tensor.shape) == tuple(subgraph_output_tensor.shape), \
    f"Shape mismatch: after_tensor shape {after_tensor.shape} != subgraph output shape {subgraph_output_tensor.shape}"

    # dfs to find all the nodes between before_tensor and after_tensor
    nodes_to_remove = []

    def dfs_collect_nodes_bt(tensor, stop_tensor, path, nodes_to_remove):
        for consumer in tensor.outputs:
            path.append(consumer)
            for out_tensor in consumer.outputs:
                if out_tensor is stop_tensor:
                    for node in path:
                        if node not in nodes_to_remove:
                            nodes_to_remove.append(node)
                else:
                    dfs_collect_nodes_bt(out_tensor, stop_tensor, path, nodes_to_remove)
            path.pop()
            
    dfs_collect_nodes_bt(before_tensor, after_tensor, [], nodes_to_remove)
    
    def get_external_produced_tensors(graph, nodes_to_remove, before_tensor_name, after_tensor_name):
        """
        Returns a set of tensor names produced by nodes NOT in nodes_to_remove,
        and graph input tensors that are NOT initializers/constants,
        excluding before_tensor and after_tensor.
        """
        external_tensors = set()
        # Add outputs of nodes not being removed
        for node in graph.nodes:
            if node not in nodes_to_remove:
                # Skip constant nodes
                if hasattr(node, "op") and node.op == "Constant":
                    continue
                for out_tensor in node.outputs:
                    if out_tensor.name not in (before_tensor_name, after_tensor_name):
                        external_tensors.add(out_tensor.name)
        # Add graph input tensors that are not initializers/constants
        for inp in graph.inputs:
            if inp.name not in (before_tensor_name, after_tensor_name):
                is_initializer = hasattr(inp, "values") and inp.values is not None
                if not is_initializer:
                    external_tensors.add(inp.name)
        return external_tensors

    external_tensors = get_external_produced_tensors(graph, nodes_to_remove, before_tensor_name, after_tensor_name)

    def has_external_input(node, external_tensors):
        for inp in node.inputs:
            if inp.name in external_tensors:
                return inp.name
        return None

    def has_external_output(node, nodes_to_remove, after_tensor_name):
        for out_tensor in node.outputs:
            if out_tensor.name == after_tensor_name:
                continue  # skip after_tensor
            for consumer in out_tensor.outputs:
                if consumer not in nodes_to_remove:
                    return out_tensor.name, getattr(consumer, "name", "<unnamed>")
        return None, None

    # Check for external inputs/outputs
    for node in graph.nodes:
        if node not in nodes_to_remove:
            continue
        ext_in = has_external_input(node, external_tensors)
        if ext_in:
            logging.warning(
                f"Node '{node}' has input '{ext_in}' from outside the subgraph region. Skipping transformation."
            )
            return False
        ext_out, ext_consumer = has_external_output(node, nodes_to_remove, after_tensor_name)
        if ext_out:
            logging.warning(
                f"Node '{node}' has output '{ext_out}' consumed by node '{ext_consumer}' outside the subgraph region. Skipping transformation."
            )
            return False
        
    graph.nodes = [node for node in graph.nodes if node not in nodes_to_remove]

    # change graph connections
    # Only connect before_tensor to new subgraph's input for nodes that are being removed
    for node in new_subgraph.nodes:
        for idx, inp in enumerate(node.inputs):
            if inp.name == subgraph_input_tensor.name:
                node.inputs[idx] = before_tensor

    for node in graph.nodes:
        if node in nodes_to_remove:
            continue
        for idx, inp in enumerate(node.inputs):
            if inp.name == after_tensor.name:
                node.inputs[idx] = subgraph_output_tensor
                
    # Handle the case where after_tensor is a graph output
    for idx, out in enumerate(graph.outputs):
        if out.name == after_tensor.name:
            graph.outputs[idx] = subgraph_output_tensor

    # insert all nodes from new_subgraph into the original graph
    graph.nodes.extend(new_subgraph.nodes)

    graph.cleanup().toposort()
    # onnx.save(gs.export_onnx(graph), "") <- for testing

    return True

def insert_subgraph_with_mappings(
    graph: gs.Graph,
    input_mapping: dict,   # {subgraph_input_name: original_graph_tensor_name}
    output_mapping: dict,  # {subgraph_output_name: original_graph_tensor_name}
    new_subgraph: gs.Graph,
    suffix : str = None
):
    """
    Replace the subgraph between before_tensor_name and after_tensor_name with new_subgraph.
    Assumes user ensures input/output compatibility.
    """
    # Rename the nodes and tensor names in the new subgraph to avoid conflicts
    if suffix:
        # 1. Rename nodes
        for node in new_subgraph.nodes:
            if node.name:
                node.name = node.name + suffix
            elif hasattr(node, "op") and node.op:
                node.name = f"{node.op}{suffix}"
            else:
                node.name = f"node_<no_name>{suffix}"

        # 2. Rename tensors and update input/output mapping keys
        tensor_rename_map = {}
        # First, rename all tensors and build a mapping
        for tensor in new_subgraph.tensors().values():
            old_name = tensor.name
            tensor.name = old_name + suffix
            tensor_rename_map[old_name] = tensor.name

        # Update input_mapping keys if needed
        new_input_mapping = {}
        for k, v in input_mapping.items():
            new_k = tensor_rename_map.get(k, k)
            new_input_mapping[new_k] = v
        input_mapping = new_input_mapping

        # Update output_mapping keys if needed
        new_output_mapping = {}
        for k, v in output_mapping.items():
            new_k = tensor_rename_map.get(k, k)
            new_output_mapping[new_k] = v
        output_mapping = new_output_mapping
    
    
    original_inputs = {}
    for sub_in, orig_in in input_mapping.items():
        tensor = next((t for t in graph.tensors().values() if t.name == orig_in), None)
        if tensor is None:
            logging.error(f"Input tensor '{orig_in}' not found in the original graph.")
            return False
        original_inputs[sub_in] = tensor

    original_outputs = {}
    for sub_out, orig_out in output_mapping.items():
        tensor = next((t for t in graph.tensors().values() if t.name == orig_out), None)
        if tensor is None:
            logging.error(f"Output tensor '{orig_out}' not found in the original graph.")
            return False
        original_outputs[sub_out] = tensor
        
    
    # check for missing mappings
    missing_inputs = [inp.name for inp in new_subgraph.inputs if inp.name not in input_mapping]
    if missing_inputs:
        logging.error(f"No mapping provided for subgraph input(s): {missing_inputs}")
        return False

    missing_outputs = [out.name for out in new_subgraph.outputs if out.name not in output_mapping]
    if missing_outputs:
        logging.error(f"No mapping provided for subgraph output(s): {missing_outputs}")
        return False

    # assert shape compatibility for each mapping
    for sub_in in new_subgraph.inputs:
        orig_tensor = original_inputs[sub_in.name]
        if tuple(sub_in.shape) != tuple(orig_tensor.shape):
            logging.error(f"Shape mismatch for input mapping: subgraph input '{sub_in.name}' shape {sub_in.shape} != original tensor '{orig_tensor.name}' shape {orig_tensor.shape}")
            return False

    for sub_out in new_subgraph.outputs:
        orig_tensor = original_outputs[sub_out.name]
        if tuple(sub_out.shape) != tuple(orig_tensor.shape):
            logging.error(f"Shape mismatch for output mapping: subgraph output '{sub_out.name}' shape {sub_out.shape} != original tensor '{orig_tensor.name}' shape {orig_tensor.shape}")
            return False



    nodes_to_remove = []

    # def dfs_collect_nodes_multi(tensor, output_tensors, path, nodes_to_remove):
    #     for consumer in tensor.outputs:
    #         path.append(consumer)
    #         for out_tensor in consumer.outputs:
    #             if out_tensor in output_tensors:
    #                 # Found a mapped output tensor: add all nodes in the path
    #                 for node in path:
    #                     if node not in nodes_to_remove:
    #                         nodes_to_remove.append(node)
    #                 # Continue traversal to catch chained outputs
    #                 dfs_collect_nodes_multi(out_tensor, output_tensors, path, nodes_to_remove)
    #             else:
    #                 dfs_collect_nodes_multi(out_tensor, output_tensors, path, nodes_to_remove)
    #         path.pop()
    def dfs_collect_nodes_multi(tensor, output_tensors, path, nodes_to_remove, visited):
        if tensor in visited:
            return
        visited.append(tensor)
        for consumer in tensor.outputs:
            path.append(consumer)
            for out_tensor in consumer.outputs:
                if out_tensor in output_tensors:
                    for node in path:
                        if node not in nodes_to_remove:
                            nodes_to_remove.append(node)
                    dfs_collect_nodes_multi(out_tensor, output_tensors, path, nodes_to_remove, visited)
                else:
                    dfs_collect_nodes_multi(out_tensor, output_tensors, path, nodes_to_remove, visited)
            path.pop()

    input_tensors = list(original_inputs.values())   # from input_mapping
    output_tensors = list(original_outputs.values()) # from output_mapping

    for tensor in input_tensors:
        dfs_collect_nodes_multi(tensor, output_tensors, [], nodes_to_remove, [])



    def get_external_produced_tensors_multi(graph, nodes_to_remove, input_tensor_names, output_tensor_names):
        """
        Returns a set of tensor names produced by nodes NOT in nodes_to_remove,
        and graph input tensors that are NOT initializers/constants,
        excluding input_tensor_names and output_tensor_names.
        """
        external_tensors = set()
        # Add outputs of nodes not being removed
        for node in graph.nodes:
            if node not in nodes_to_remove:
                if hasattr(node, "op") and node.op == "Constant":
                    continue
                for out_tensor in node.outputs:
                    if out_tensor.name not in input_tensor_names and out_tensor.name not in output_tensor_names:
                        external_tensors.add(out_tensor.name)
        # Add graph input tensors that are not initializers/constants
        for inp in graph.inputs:
            if inp.name not in input_tensor_names and inp.name not in output_tensor_names:
                is_initializer = hasattr(inp, "values") and inp.values is not None
                if not is_initializer:
                    external_tensors.add(inp.name)
        return external_tensors

    external_tensors = get_external_produced_tensors_multi(
        graph, nodes_to_remove, set(input_mapping.values()), set(output_mapping.values())
    )

    def has_external_input_multi(node, external_tensors):
        for inp in node.inputs:
            if inp.name in external_tensors:
                return inp.name
        return None

    def has_external_output_multi(node, nodes_to_remove, output_tensor_names):
        for out_tensor in node.outputs:
            if out_tensor.name in output_tensor_names:
                continue  # skip mapped outputs
            for consumer in out_tensor.outputs:
                if consumer not in nodes_to_remove:
                    return out_tensor.name, getattr(consumer, "name", "<unnamed>")
        return None, None

    # Check for external inputs/outputs for all nodes to be removed
    for node in nodes_to_remove:
        ext_in = has_external_input_multi(node, external_tensors)
        if ext_in:
            logging.warning(
                f"Node '{node}' has input '{ext_in}' from outside the subgraph region. Skipping transformation."
            )
            return False
        ext_out, ext_consumer = has_external_output_multi(node, nodes_to_remove, set(output_mapping.values()))
        if ext_out:
            logging.warning(
                f"Node '{node}' has output '{ext_out}' consumed by node '{ext_consumer}' outside the subgraph region. Skipping transformation."
            )
            return False


    # remove the nodes
    graph.nodes = [node for node in graph.nodes if node not in nodes_to_remove]
    
    # wiring the new subgraph
    # 1. Connect subgraph inputs
    for node in new_subgraph.nodes:
        for idx, inp in enumerate(node.inputs):
            if inp.name in input_mapping:
                node.inputs[idx] = original_inputs[inp.name]

    # 2. Connect subgraph outputs
    for node in graph.nodes:
        for idx, inp in enumerate(node.inputs):
            for sub_out_name, orig_out_tensor in original_outputs.items():
                if inp is orig_out_tensor:
                    # Find the corresponding subgraph output tensor
                    subgraph_out_tensor = next((t for t in new_subgraph.outputs if t.name == sub_out_name), None)
                    if subgraph_out_tensor:
                        node.inputs[idx] = subgraph_out_tensor

    # 3. Update graph outputs if needed
    for idx, out in enumerate(graph.outputs):
        for sub_out_name, orig_out_tensor in original_outputs.items():
            if out is orig_out_tensor:
                subgraph_out_tensor = next((t for t in new_subgraph.outputs if t.name == sub_out_name), None)
                if subgraph_out_tensor:
                    graph.outputs[idx] = subgraph_out_tensor

    # 4. Insert new subgraph nodes
    graph.nodes.extend(new_subgraph.nodes)
    graph.cleanup().toposort()
    
    # onnx.save(gs.export_onnx(graph), "")   <- For testing
    return True

def get_all_deformal_convolution_nodes(graph: gs.Graph):
    
    def search_for_straigt_pattern_in_inputs(variable:gs.Variable, pattern):
        nodes = []
        for p in pattern:
            if len(variable.inputs) == 0: return None
            inp_node = variable.inputs[0]
            if inp_node.op != p:
                return None
            nodes.append(inp_node)
            variable = inp_node.inputs[0]
        return nodes
    
    deform_convs = []
    
    for node in graph.nodes:
        if node.op != 'GridSample':
            continue
        grid_sample_node = node
        out_node = grid_sample_node.outputs[0]
        if len(out_node.outputs) != 1: continue
        out_node = out_node.outputs[0]
        
        if out_node.op != 'Mul':
            continue
        mul1_node = out_node
        out_node = mul1_node.outputs[0]
        if len(out_node.outputs) != 1: continue
        out_node = out_node.outputs[0]
        
        if out_node.op != 'Reshape':
            continue
        reshape2_node = out_node
        out_node = reshape2_node.outputs[0]
        if len(out_node.outputs) != 1: continue
        out_node = out_node.outputs[0]
        
        if out_node.op != 'Conv':
            continue
        final_conv_node = out_node
        
        inp_node = node.inputs[0]
        if len(inp_node.inputs) == 0: continue
        inp_node = inp_node.inputs[0]
        
        if inp_node.op != 'Pad':
            continue
        pad_node = inp_node
        
        main_inp = pad_node.inputs[0]
        offset_branch = grid_sample_node.inputs[1]
        mask_branch = mul1_node.inputs[1]
        
        result = search_for_straigt_pattern_in_inputs(mask_branch, ['Reshape', 'Sigmoid'])
        if result is None:
            continue
        reshape1_node, sigmoid_node = result
        
        inp_node = sigmoid_node.inputs[0]
        if len(inp_node.inputs) == 0: continue
        inp_node = inp_node.inputs[0]
        if inp_node.op not in  ('Conv', 'Slice'):
            continue
        if inp_node.op == 'Conv':
            if inp_node.inputs[0] is  main_inp:
                mask_node = mask_conv_node = inp_node
            else: continue
        elif inp_node.op == 'Slice':
            mask_node = inp_node
            inp_node = mask_node.inputs[0]
            if len(inp_node.inputs) == 0: continue
            inp_node = inp_node.inputs[0]
            if inp_node.op != 'Conv':
                continue
            mask_conv_node = inp_node
            if mask_conv_node.inputs[0] is not main_inp:
                continue
        
        result = search_for_straigt_pattern_in_inputs(offset_branch,['Reshape', 'Sub', 'Mul', 'Concat'])
        if result is None:
            continue
        reshape_node, sub_node, mul_node, concat_node = result
        if len(concat_node.inputs) != 2: continue
        x_side, y_side = concat_node.inputs

        result = search_for_straigt_pattern_in_inputs(x_side, ['Unsqueeze', 'Div','Add','Slice'])
        if result is None:
            continue
        x_unsqueeze_node, x_div_node, x_add_node, x_slice_node = result
        result = search_for_straigt_pattern_in_inputs(y_side, ['Unsqueeze', 'Div','Add','Slice'])
        if result is None:
            continue
        y_unsqueeze_node, y_div_node, y_add_node, y_slice_node = result
        if x_slice_node.inputs[0] is not y_slice_node.inputs[0]:
            continue
        inp_node = x_slice_node.inputs[0]
        if len(inp_node.inputs)==0: continue
        inp_node = inp_node.inputs[0]
        if inp_node.op == 'Conv':
            offset_conv_node = inp_node
        if offset_conv_node.inputs[0] is not main_inp:
            continue
        deform_convs.append([offset_conv_node, mask_node, x_slice_node, y_slice_node,
                            x_add_node, y_add_node, x_div_node, y_div_node, x_unsqueeze_node, y_unsqueeze_node,
                            concat_node, mul_node, sub_node, reshape_node, pad_node, grid_sample_node, 
                            sigmoid_node, reshape1_node, mul1_node, reshape2_node, final_conv_node
                            ])
    return deform_convs