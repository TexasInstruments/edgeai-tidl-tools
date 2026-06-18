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
Module containing GatherElements layer specific functions and optimizations
"""
import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np



def tidl_replace_tile_gatherelements_with_reshape_gather (graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Reads an ONNX model, finds Tile+GatherElements patterns, and replaces them with Reshape+Gather operations
    for better performance optimization.
    
    This function identifies a specific pattern in ONNX models where a Tile operation is followed by a 
    GatherElements operation. This pattern is then transformed into more efficient Reshape+Gather operations.
    The optimization focuses particularly on patterns where:
    - The Tile node has a variable input and constant repeat
    - The input shape is 3D and the last dimension is 1
    - The GatherElements operation has axis=1
    
    If the Tile node has consumers other than GatherElements nodes or its output is a graph output,
    the original Tile node is preserved, and a new Reshape node is created specifically for the
    GatherElements consumers.
    
    Parameters
    ----------
    graph : gs.Graph
        The ONNX graph surgeon Graph object representing the model to be optimized.
        This graph will be modified in-place.
        
    onnx_graph : onnx.GraphProto
        The original ONNX GraphProto object. This is used as reference but not modified.
        
    Returns
    -------
    None
        The function modifies the input graph in-place and doesn't return any value.
        
    Notes
    -----
    This optimization:
    1. Identifies Tile nodes feeding into GatherElements nodes
    2. Validates that the pattern meets the required shape constraints
    3. Replaces Tile operations with Reshape operations (or creates a new Reshape for GatherElements)
    4. Transforms GatherElements operations into Gather operations
    5. Adds necessary intermediate operations (Add, Reshape) to maintain functional equivalence

    """
    logging.debug("Starting Tile+GatherElements pattern replacement optimization")
    
    # Find all Tile nodes
    tile_nodes = [node for node in graph.nodes if node.op == "Tile"]
    logging.debug(f"Found {len(tile_nodes)} Tile nodes in the graph")
    
    for tile_node in tile_nodes:
        logging.debug(f"Processing Tile node: {tile_node.name}")
        # Check if this Tile node feeds into a GatherElements node
        inp, repeat = tile_node.inputs
        if not isinstance(inp, gs.Variable) or not isinstance(repeat, gs.Constant):
            logging.debug(f"Tile node {tile_node.name} should have a variable input and constant repeat, skipping")
            continue
        
        if not tile_node.outputs or not tile_node.outputs[0].outputs:
            logging.debug(f"Tile node {tile_node.name} has no outputs, skipping")
            continue
        tile_out = tile_node.outputs[0]
        inp_shape = inp.shape
        if not inp_shape:
            logging.debug(f"Tile node {tile_node.name} input has no shape, skipping")
            continue
        if len(inp_shape) != 3 or inp_shape[-1] != 1:
            logging.debug(f"Tile node {tile_node.name} input has to be 3D input and last dimension should be 1, skipping")
            continue
        changed = False
        gather_elements_nodes = [n for n in tile_out.outputs if n.op == "GatherElements"]
        
        if not gather_elements_nodes:
            logging.debug(f"Tile node {tile_node.name} does not feed into GatherElements, skipping")
            continue
        
        # Check if Tile has any consumers other than GatherElements
        other_consumers = [n for n in tile_out.outputs if n.op != "GatherElements"]
        # Also check if the output is a graph output, which should be preserved
        is_graph_output = tile_out in graph.outputs
        has_other_consumers = len(other_consumers) > 0 or is_graph_output
        
        if has_other_consumers:
            if len(other_consumers) > 0:
                logging.debug(f"Tile node {tile_node.name} has consumers other than GatherElements: {[n.name for n in other_consumers]}")
            if is_graph_output:
                logging.debug(f"Tile node {tile_node.name} output '{tile_out.name}' is a graph output and will be preserved")
        
        # We found a Tile->GatherElements pattern
        tile_out_shape = np.array([np.prod(inp_shape[:2])]).astype(np.int64).tolist()
        logging.debug(f"Found Tile->GatherElements pattern: {tile_node.name} -> {[n.name for n in gather_elements_nodes]}")
        for gather_elements_node in gather_elements_nodes:
            logging.debug(f"Processing GatherElements node: {gather_elements_node.name}")
            try:
                # Get the data tensor for the GatherElements (first input)
                data_tensor = gather_elements_node.inputs[0]
                data_shape = data_tensor.shape
                if not data_shape or len(data_shape) != 3:
                    logging.debug(f"GatherElements node {gather_elements_node.name} data input has no shape or is not 3D, skipping")
                    continue
                gather_axis = gather_elements_node.attrs.get("axis", 0)
                if gather_axis != 1:
                    continue
                logging.debug(f"GatherElements axis: {gather_axis}")
                # Determine the input to use for further processing
                if has_other_consumers and not changed:
                    # If this is the first GatherElements node and we need to preserve Tile,
                    # create the Reshape node now
                    reshape_out = gs.Variable(f"{tile_node.name}_reshape_out", tile_out.dtype, tile_out_shape)
                    shape_const = gs.Constant(f"{tile_node.name}_shape", np.array(tile_out_shape))
                    reshape_node = gs.Node("Reshape", f"{tile_node.name}_reshape", {}, 
                                          [tile_out, shape_const], [reshape_out])
                    graph.nodes.append(reshape_node)
                    changed = True
                    
                    # Use the reshape output for further processing
                    current_input = reshape_out
                else:
                    # Use the tile/reshape output for further processing
                    current_input = tile_out
                
                # Add offset if needed
                if inp_shape[0] != 1:
                    offset = np.arange(data_shape[0]).reshape(-1,1)*np.ones(inp_shape[:2])*data_shape[1]
                    offset = offset.flatten().astype(np.int64)
                    offset = gs.Constant(f'{gather_elements_node.name}_offset', offset)
                    add_out = gs.Variable(f'{gather_elements_node.name}_offset_add_out', current_input.dtype, tile_out_shape)
                    add_node = gs.Node('Add', f'{gather_elements_node.name}_offset_add', {}, [current_input, offset], [add_out])
                    graph.nodes.append(add_node)
                    gather_elements_node.inputs[1] = add_out
                    index_out_shape = add_out.shape
                else:
                    # No offset needed
                    gather_elements_node.inputs[1] = current_input
                    index_out_shape = tile_out_shape
                # Get the GatherElements axis
                # Get the output of the GatherElements
                if isinstance(data_tensor, gs.Constant):
                    data_tensor.values = data_tensor.values.reshape(-1,data_tensor.shape[-1])
                else:
                    shape = gs.Constant(data_tensor.name+'_shape', np.array([-1,data_shape[-1]]))
                    reshape_out = gs.Variable(f'{data_tensor.name}_reshape_out', data_tensor.dtype, [int(np.prod(data_shape[:-1])), data_shape[-1]])
                    reshape_node = gs.Node('Reshape', f'{data_tensor.name}_reshape',{}, [data_tensor, shape], [reshape_out])
                    graph.nodes.append(reshape_node)
                    data_tensor = reshape_out
                    gather_elements_node.inputs[0] = data_tensor
                gather_output = gather_elements_node.outputs[0]
                gather_elements_node.op = 'Gather'
                gather_elements_node.attrs['axis'] = gather_axis = 0
                out_shape = list(data_tensor.shape).copy()
                out_shape[gather_axis] = index_out_shape[0]
                gather_out = gs.Variable(f'{gather_elements_node.name}_gather_out', data_tensor.dtype, out_shape)
                gather_elements_node.outputs[0] = gather_out
                out_shape = gs.Constant(f'{gather_elements_node.name}_shape',np.array(gather_output.shape))
                reshape = gs.Node('Reshape', f'{gather_elements_node.name}_reshape',{},[gather_out, out_shape],[gather_output])
                graph.nodes.append(reshape)
                # Determine the shape needed for Reshape
                # For a typical case where Tile expands a dimension that needs to be flattened
                if not changed:
                    if has_other_consumers:
                        # Create a new Reshape node for GatherElements consumers, preserve original Tile
                        reshape_out = gs.Variable(f"{tile_node.name}_reshape_out", tile_out.dtype, tile_out_shape)
                        shape_const = gs.Constant(f"{tile_node.name}_shape", np.array(tile_out_shape))
                        reshape_node = gs.Node("Reshape", f"{tile_node.name}_reshape", {}, 
                                              [tile_out, shape_const], [reshape_out])
                        graph.nodes.append(reshape_node)
                        
                        # Redirect all subsequent operations to use the new reshape output
                        # Note: gather_elements_node.inputs has already been modified above
                        changed = True
                    else:
                        # Transform the Tile node into a Reshape node
                        tile_node.name += '_reshape'
                        tile_node.op = 'Reshape'
                        repeat.name += '_shape'
                        repeat.values = np.array(tile_out_shape)
                        tile_out.shape = tile_out_shape
                        changed = True
                
            except Exception as e:
                logging.debug(f"Exception occurred while replacing pattern for {tile_node.name}: {str(e)}")
                print(f"Error replacing pattern for {tile_node.name}: {str(e)}")
    
    logging.debug("Completed Tile+GatherElements pattern replacement optimization")


def tidl_replace_expand_gatherelements_with_reshape_gather(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Reads an ONNX model, finds Expand+GatherElements patterns, converts Expand to Tile,
    and then calls the tidl_replace_tile_gatherelements_with_reshape_gather function
    for better performance optimization.
    
    This function identifies a specific pattern in ONNX models where an Expand operation is followed by a 
    GatherElements operation. The Expand operation is first converted to a functionally equivalent Tile 
    operation, and then the existing optimization for Tile+GatherElements is applied.
    The optimization focuses particularly on patterns where:
    - The Expand node has a variable input and constant shape
    - The input shape is 3D and the last dimension is 1
    - The GatherElements operation has axis=1
    
    If the Expand node has consumers other than GatherElements nodes, the original Expand node
    is preserved, and a new Tile node is created specifically for the GatherElements consumers.
    
    Parameters
    ----------
    graph : gs.Graph
        The ONNX graph surgeon Graph object representing the model to be optimized.
        This graph will be modified in-place.
        
    onnx_graph : onnx.GraphProto
        The original ONNX GraphProto object. This is used as reference but not modified.
        
    Returns
    -------
    None
        The function modifies the input graph in-place and doesn't return any value.
        
    Notes
    -----
    This optimization:
    1. Identifies Expand nodes feeding into GatherElements nodes
    2. Converts the Expand operations to equivalent Tile operations for GatherElements consumers
    3. Preserves the original Expand node for any other consumers
    4. Calls tidl_replace_tile_gatherelements_with_reshape_gather to optimize the resulting pattern
    """
    logging.debug("Starting Expand+GatherElements pattern replacement optimization")
    
    # Find all Expand nodes
    expand_nodes = [node for node in graph.nodes if node.op == "Expand"]
    logging.debug(f"Found {len(expand_nodes)} Expand nodes in the graph")
    
    # Track if any nodes were converted for later processing
    converted_nodes = False
    
    for expand_node in expand_nodes:
        logging.debug(f"Processing Expand node: {expand_node.name}")
        # Check if this Expand node feeds into a GatherElements node
        inp, shape = expand_node.inputs
        if not isinstance(inp, gs.Variable) or not isinstance(shape, gs.Constant):
            logging.debug(f"Expand node {expand_node.name} should have a variable input and constant shape, skipping")
            continue
        
        if not expand_node.outputs or not expand_node.outputs[0].outputs:
            logging.debug(f"Expand node {expand_node.name} has no outputs, skipping")
            continue
            
        expand_out = expand_node.outputs[0]
        inp_shape = inp.shape
        
        if not inp_shape:
            logging.debug(f"Expand node {expand_node.name} input has no shape, skipping")
            continue
            
        # Check if any outputs are GatherElements nodes
        gather_elements_nodes = [n for n in expand_out.outputs if n.op == "GatherElements"]
        
        if not gather_elements_nodes:
            logging.debug(f"Expand node {expand_node.name} does not feed into GatherElements, skipping")
            continue
        
        # Check if Expand has any consumers other than GatherElements
        other_consumers = [n for n in expand_out.outputs if n.op != "GatherElements"]
        # Also check if the output is a graph output, which should be preserved
        is_graph_output = expand_out in graph.outputs
        has_other_consumers = len(other_consumers) > 0 or is_graph_output
        
        if has_other_consumers:
            if len(other_consumers) > 0:
                logging.debug(f"Expand node {expand_node.name} has consumers other than GatherElements: {[n.name for n in other_consumers]}")
            if is_graph_output:
                logging.debug(f"Expand node {expand_node.name} output '{expand_out.name}' is a graph output and will be preserved")
        
        # We found an Expand->GatherElements pattern, create a Tile node
        logging.debug(f"Found Expand->GatherElements pattern: {expand_node.name} -> {[n.name for n in gather_elements_nodes]}")
        
        try:
            # Calculate repeat values for Tile
            target_shape = shape.values
            input_shape = np.array(inp_shape)
            
            # Handle broadcasting
            if len(target_shape) > len(input_shape):
                # Prepend 1s to input_shape to match target_shape length
                padding = len(target_shape) - len(input_shape)
                input_shape = np.concatenate([np.ones(padding, dtype=np.int64), input_shape])
            elif len(target_shape) < len(input_shape):
                logging.debug(f"Expand node {expand_node.name} target shape has fewer dimensions than input, skipping")
                continue
                
            # Calculate repeats (how many times to repeat each dimension)
            repeats = np.divide(target_shape, input_shape).astype(np.int64)
            
            # Create a new Tile node
            repeat_const = gs.Constant(f"{expand_node.name}_repeats", repeats)
            tile_out = gs.Variable(f"{expand_node.name}_tile_out", expand_out.dtype, expand_out.shape)
            
            # Create and insert the Tile node
            tile_node = gs.Node("Tile", f"{expand_node.name}_tile", {}, [inp, repeat_const], [tile_out])
            graph.nodes.append(tile_node)
            
            # If there are other consumers, only redirect the GatherElements nodes to the Tile node's output
            # Otherwise, replace all consumers with the Tile node's output
            if has_other_consumers:
                # Only redirect GatherElements nodes to use the Tile node's output
                for ge_node in gather_elements_nodes:
                    for i, node_inp in enumerate(ge_node.inputs):
                        if node_inp == expand_out:
                            ge_node.inputs[i] = tile_out
                            logging.debug(f"Redirected GatherElements node {ge_node.name} to use Tile node output")
            else:
                # Replace all consumers with the Tile node's output
                for consumer in expand_out.outputs:
                    for i, node_inp in enumerate(consumer.inputs):
                        if node_inp == expand_out:
                            consumer.inputs[i] = tile_out
                
                # Remove the Expand node from the graph as it's no longer needed
                graph.cleanup()
            
            converted_nodes = True
            logging.debug(f"Successfully created Tile node {tile_node.name} for GatherElements consumers of Expand node {expand_node.name}")
            
        except Exception as e:
            logging.debug(f"Exception occurred while processing Expand node {expand_node.name}: {str(e)}")
            print(f"Error processing Expand node {expand_node.name}: {str(e)}")
    
    # If any Tile nodes were created for GatherElements consumers, run the Tile+GatherElements optimization
    if converted_nodes:
        logging.debug("Calling tidl_replace_tile_gatherelements_with_reshape_gather after creating Tile nodes")
        tidl_replace_tile_gatherelements_with_reshape_gather(graph, onnx_graph)
    
    logging.debug("Completed Expand+GatherElements pattern replacement optimization")
