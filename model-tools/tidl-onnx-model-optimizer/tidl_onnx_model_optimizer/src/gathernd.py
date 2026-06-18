import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np

def tidl_convert_single_axis_gethernd_to_gather(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    '''
    Converts GatherND nodes that operate on a single axis to Gather nodes.
    This optimization is applicable when the GatherND operation is effectively 
    gathering from a single axis, which can be more efficiently represented as a Gather operation.
    '''
    # Get all GatherND nodes in the graph
    gathernd_nodes = [node for node in graph.nodes if node.op == "GatherND"]
    logging.debug(f"Found {len(gathernd_nodes)} GatherND nodes to convert")
    # Process each GatherND node
    for node in graph.nodes:
        if node.op != "GatherND":
            continue
            
        logging.debug(f"Processing GatherND node: {node.name if node.name else 'unnamed'}")
        # Get input tensors: data and indices
        data, indices = node.inputs
        logging.debug(f"Data shape: {data.shape}, Indices shape: {indices.shape}")
        
        # Skip if shapes are unknown
        if data.shape is None or indices.shape is None:
            logging.debug(f"GatherND node has unknown shapes. Skipping!")
            continue
        # Check if indices is a constant (required for this optimization)
        if isinstance(indices, gs.Variable):
            logging.debug(f'GatherND node {node.name} has variable indices. Skipping!')
            continue
        
        logging.debug(f'GatherND node indices values shape: {indices.values.shape}')
        # assert isinstance(indices, gs.Constant)
        # Extract shapes and attributes
        inp_shape = data.shape
        indices_shape = indices.shape
        batch_dims = node.attrs.get('batch_dims', 0)
        logging.debug(f'Input shape: {inp_shape}, Indices shape: {indices_shape}, Batch dims: {batch_dims}')

        # Determine the axis for the Gather operation
        if all(s==1 for s in indices_shape):
            # All dimensions are 1, use the length of indices shape
            axis = batch_dims+len(indices_shape)
            logging.debug(f'All indices dimensions are 1, using axis: {axis}')
        else:
            # Find the non-unit dimension (if there's exactly one)
            non_ones = [i for i,s in enumerate(indices_shape) if s!=1]
            if len(non_ones) != 1:
                logging.debug(f'GatherND node {node.name} has multi axes gathering. Skipping!')
                continue
            axis = batch_dims+non_ones[0]
            logging.debug(f'Found single non-unit dimension at index {non_ones[0]}, using axis: {axis}')
        # Convert GatherND to Gather
        node.attrs['axis'] = axis
        node.op = 'Gather'
        
        # Remove batch_dims attribute as it's not used in Gather
        if 'batch_dims' in node.attrs:
            node.attrs.pop('batch_dims')
            
        # Reshape indices to 1D array for Gather operation
        indices.values = indices.values.reshape([-1])
        
        logging.debug(f'Successfully converted GatherND node {node.name} to Gather with axis={axis}')
