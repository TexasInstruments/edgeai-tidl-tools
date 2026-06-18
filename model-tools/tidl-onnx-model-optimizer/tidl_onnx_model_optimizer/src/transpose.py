import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np

def tidl_break_transpose_of_width_to_dim1_dim2_of_input_more_than_4d(graph:gs.Graph, onnx_graph:onnx.GraphProto):
    '''
    Transforms ONNX Transpose operations that have incompatible permutation patterns with TIDL.
    
    For inputs with more than 4 dimensions (>4D), permutations where the width dimension is mapped to either 
    the first or second position (Dim 0 or Dim 1) in the output shape are not supported by TIDL.
    
    Unsupported permutation patterns:
    [W, X, X, X, X, X] - Width dimension at position 0
    [X, W, X, X, X, X] - Width dimension at position 1
    
    This function breaks down such unsupported Transpose operations into a sequence of:
    Reshape -> Transpose -> Reshape operations that can be handled by TIDL.
    '''
    # Find all Transpose nodes in the graph
    transpose_nodes = [node for node in graph.nodes if node.op == 'Transpose']
    logging.debug(f"Found {len(transpose_nodes)} Transpose nodes to analyze")
    
    for node in transpose_nodes:
        # Get the permutation attribute
        perm = node.attrs['perm']
        D = len(perm)
        
        # Skip if dimensionality is <= 4 as these are directly supported
        if D <= 4:
            logging.debug(f"Transpose node {node.name} has {D} dimensions, skipping (<=4D)")
            continue
        
        # Check if width dimension (D-1) is NOT in the first two positions of permutation
        # If width is not in first two positions, no need to transform
        if all(p!=(D-1) for p in perm[:2]):
            logging.debug(f"Transpose node {node.name} doesn't have width in first two positions, skipping")
            continue
        
        # Initialize identity permutation and its copy
        perm1 = list(range(D))  # Identity permutation [0,1,2,...,D-1]
        perm2 = perm1.copy()
        
        # Find which dimension maps to the width dimension (D-1)
        dim = [i for i, p in enumerate(perm) if p==(D-1)]
        if len(dim) != 1:
            logging.debug(f"Transpose node {node.name} has invalid permutation, skipping")
            continue
        
        # Get the dimension that maps to width
        dim = dim[0]
        dim1 = D-1  # Width dimension
        
        # Swap positions in identity permutation to get initial transpose
        perm1[dim], perm1[dim1] = perm1[dim1], perm1[dim]
        
        # Find where the dimensions of interest end up after permutation
        d1 = perm.index(dim)
        d2 = perm.index(dim1)
        
        # Get input tensor to the transpose node
        inp = node.inputs[0]
        
        # Create a new permutation that swaps the positions
        new_perm = perm.copy()
        new_perm[d1], new_perm[d2] = new_perm[d2], new_perm[d1]
        
        # Check if permutations match our patterns and decide whether to reuse or create new node
        if perm==perm1 or new_perm == perm2:
            if perm==perm1 and new_perm==perm2:
                # If both conditions match, reuse existing node
                transpose = node
                logging.debug(f"Reusing transpose node {node.name} for transformation")
            else:
                # Skip if only one condition matches but not both
                logging.debug(f"Transpose node {node.name} doesn't match transformation pattern, skipping")
                continue
        else:
            # Create a new transpose node with perm1 permutation
            logging.debug(f"Creating new transpose node for {node.name}")
            transpose_in = gs.Variable(f'{node.name}_in', inp.dtype)
            transpose = gs.Node('Transpose', f'{node.name}_1', dict(perm=perm1), [inp], [transpose_in])
            
            # Update original node to use new permutation and connect to the new transpose
            node.attrs['perm'] = new_perm
            node.inputs[0] = transpose_in
            graph.nodes.append(transpose)
        
        # Get updated input and output of transpose node
        inp = transpose.inputs[0]
        out = transpose.outputs[0]
        
        # Skip if shape information is not available
        if inp.shape is None:
            logging.debug(f"Input shape for node {transpose.name} is None, skipping")
            continue
        
        # Determine permutation needed based on which dimension has width
        if dim == 0:
            # When width is in first dimension, use 3D permutation
            new_perm = [2, 1, 0]
            logging.debug(f"Using 3D permutation {new_perm} for node {transpose.name}")
        elif dim == 1:
            # When width is in second dimension, use 4D permutation
            new_perm = [0, 3, 2, 1]
            logging.debug(f"Using 4D permutation {new_perm} for node {transpose.name}")
        else:
            logging.debug(f"Width dimension position {dim} not supported for transformation, skipping")
            continue
        
        # Get current permutation and calculate output shape
        perm = transpose.attrs['perm']
        out.shape = [inp.shape[i] for i in perm]
        
        # Find indices where dimensions of interest end up
        d1 = perm.index(dim)
        d2 = perm.index(dim1)
        
        # Calculate new shape for the first reshape operation
        # This is a critical part that reshapes the tensor to group dimensions appropriately
        # for the following transpose operation
        shape = inp.shape[:d2] + inp.shape[d2:d2+1] + [np.prod(inp.shape[d2+1:d1]).tolist()] + inp.shape[d1:]
        logging.debug(f"First reshape for {transpose.name}: {inp.shape} -> {shape}")
        
        # Create first reshape node
        reshape1_out = gs.Variable(f'{transpose.name}_reshape1_out', inp.dtype, shape)
        shape_const = gs.Constant(f'{transpose.name}_shape1', np.array(shape).astype(np.int64))
        reshape1 = gs.Node('Reshape', f'{transpose.name}_reshape1', {}, [inp, shape_const], [reshape1_out])
        transpose.inputs[0] = reshape1_out  # Connect transpose to reshape output
        graph.nodes.append(reshape1)
        
        # Calculate shape after transpose operation
        shape = reshape1_out.shape
        shape = [shape[i] for i in new_perm]  # Apply permutation to shape
        
        # Update transpose node with new permutation
        transpose.attrs['perm'] = new_perm
        reshape2_in = gs.Variable(f'{transpose.name}_reshape2_in', inp.dtype, shape)
        transpose.outputs[0] = reshape2_in  # Connect transpose output to second reshape input
        
        # Create second reshape node to restore original dimensions
        shape = out.shape  # Original output shape
        shape_const = gs.Constant(f'{transpose.name}_shape2', np.array(shape).astype(np.int64))
        reshape2 = gs.Node('Reshape', f'{transpose.name}_reshape2', {}, [reshape2_in, shape_const], [out])
        graph.nodes.append(reshape2)
        
        logging.debug(f"Completed transformation for transpose node {transpose.name}")