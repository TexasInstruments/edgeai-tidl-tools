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
    3. Replaces Tile operations with Reshape operations
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
                if inp_shape[0] != 1:
                    offset=  np.arange(data_shape[0]).reshape(-1,1)*np.ones(inp_shape[:2])*data_shape[1]
                    offset = offset.flatten().astype(np.int64)
                    offset = gs.Constant(f'{gather_elements_node.name}_offset', offset)
                    add_out = gs.Variable(f'{gather_elements_node.name}_offset_add_out',tile_out.dtype, tile_out_shape)
                    add_node = gs.Node('Add', f'{gather_elements_node.name}_offset_add', {},[tile_out, offset], [add_out])
                    graph.nodes.append(add_node)
                    gather_elements_node.inputs[1] = add_out
                    index_out_shape = add_out.shape
                else:
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
