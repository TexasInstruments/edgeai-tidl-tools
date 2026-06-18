import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np

def tidl_convert_pad_above_height_axis_to_height_axis(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    '''
    convert pad at any axis above height to transpose -> pad at height -> transpose
    
    This function processes Pad operations in an ONNX graph and converts them to a
    form more compatible with the TIDL hardware. Specifically:
    
    1. Identifies Pad operations in the graph
    2. Skips variable padding (where pad values are dynamic inputs)
    3. For pads with multiple axes, breaks them down into separate pad operations for each axis
    4. Handles different opset versions (< 11 and >= 11)
    
    For opset < 11:
    - Pad attributes: mode, pads, value
    
    For opset >= 11:
    - Pad inputs: X, pads, constant_value (optional)
    
    The function converts pads that operate on axes to the left of height axis (axis < len(shape) - 2)
    to transpose -> pad at height axis -> transpose to ensure compatibility with the 
    TIDL hardware accelerator. Input can be of any shape up to 6D.
    '''
    # Get the opset version from the graph
    opset_version = graph.opset
    logging.debug(f"ONNX opset version: {opset_version}")
    
    # Find all Pad nodes in the graph
    pad_nodes = [node for node in graph.nodes if node.op == 'Pad']
    logging.info(f"Found {len(pad_nodes)} Pad nodes to process")
    
    for node in pad_nodes:
        # Skip nodes with no inputs
        if len(node.inputs) == 0 or node.inputs[0] is None:
            logging.warning(f"Pad node {node.name} has no inputs, skipping")
            continue
        
        # Get the input tensor
        input_tensor = node.inputs[0]
        
        # Check if the input tensor has a shape
        if input_tensor.shape is None:
            logging.warning(f"Input shape for Pad node {node.name} is unknown, skipping")
            continue
        
        input_shape = input_tensor.shape
        input_rank = len(input_shape)
        
        # Calculate the height axis index (second to last dimension)
        height_axis = input_rank - 2
        
        # Handle different opset versions
        if opset_version >= 11:
            # In opset >= 11, pads are provided as an input tensor
            
            # Skip if pads are variable (not a constant)
            if len(node.inputs) < 2 or not isinstance(node.inputs[1], gs.Constant):
                logging.warning(f"Pad node {node.name} has variable padding, skipping")
                continue
            
            # Get the pads from the second input
            pads_tensor = node.inputs[1]
            if not hasattr(pads_tensor, 'values'):
                logging.warning(f"Pad node {node.name} has invalid pads tensor, skipping")
                continue
            
            pads_values = pads_tensor.values
            
            # Get the constant value (default is 0.0)
            constant_value = 0.0
            if len(node.inputs) > 2 and isinstance(node.inputs[2], gs.Constant):
                constant_tensor = node.inputs[2]
                if hasattr(constant_tensor, 'values'):
                    constant_value = float(constant_tensor.values)
        else:
            # In opset < 11, pads and constant value are attributes
            
            # Check if pads attribute exists
            if 'pads' not in node.attrs:
                logging.warning(f"Pad node {node.name} doesn't have pads attribute, skipping")
                continue
            
            pads_values = node.attrs['pads']
            
            # Get constant value (default is 0.0)
            constant_value = 0.0
            if 'value' in node.attrs:
                constant_value = float(node.attrs['value'])
        
        # Ensure pads_values has the correct length
        if len(pads_values) != 2 * input_rank:
            logging.warning(f"Pad node {node.name} has incorrect pads length, skipping")
            continue
        
        # Get the mode (default is "constant")
        mode = "constant"
        if 'mode' in node.attrs:
            mode = node.attrs['mode']
        
        # Only proceed with constant mode padding
        if mode != "constant":
            logging.warning(f"Pad node {node.name} uses '{mode}' mode, only 'constant' mode is supported, skipping")
            continue
        
        # Determine which axes are being padded
        # For each dimension, we have a padding at beginning and end
        padded_axes = []
        for i in range(input_rank):
            if pads_values[i] != 0 or pads_values[i + input_rank] != 0:
                padded_axes.append(i)
        
        # If no axes are padded, skip
        if not padded_axes:
            logging.info(f"Pad node {node.name} has no effective padding, skipping")
            continue
        
        # If we have multiple padded axes, we need to break it down into separate pads
        if len(padded_axes) > 1:
            logging.info(f"Breaking Pad node {node.name} into multiple pads for axes {padded_axes}")
            
            # Create a chain of pad operations
            current_tensor = input_tensor
            
            for i, axis in enumerate(padded_axes):
                # Check if this axis is to the left of the height axis (axis < len(shape) - 2)
                if axis < height_axis:
                    # For axes to the left of height, we need to transpose, pad, transpose back
                    
                    # Create pads specifically for this axis
                    single_axis_pads = np.zeros(2 * input_rank, dtype=np.int64)
                    single_axis_pads[axis] = pads_values[axis]
                    single_axis_pads[axis + input_rank] = pads_values[axis + input_rank]
                    
                    # Create permutation that brings this axis to height position (height_axis)
                    # and height to this axis's position
                    perm = list(range(input_rank))
                    perm[axis], perm[height_axis] = perm[height_axis], perm[axis]
                    
                    # Create transpose to make axis the height
                    transpose1_output = gs.Variable(f"{node.name}_transpose1_axis{axis}_output", dtype=input_tensor.dtype)
                    transpose1_node = gs.Node(op="Transpose",
                                            name=f"{node.name}_transpose1_axis{axis}",
                                            attrs={"perm": perm},
                                            inputs=[current_tensor],
                                            outputs=[transpose1_output])
                    graph.nodes.append(transpose1_node)
                    
                    # Calculate new pad values after transpose
                    transposed_pads = np.zeros(2 * input_rank, dtype=np.int64)
                    transposed_pads[height_axis] = single_axis_pads[axis]
                    transposed_pads[height_axis + input_rank] = single_axis_pads[axis + input_rank]
                    
                    # Create pad constants for opset >= 11
                    if opset_version >= 11:
                        pads_constant = gs.Constant(f"{node.name}_pads_axis{axis}", np.array(transposed_pads, dtype=np.int64))
                        value_constant = gs.Constant(f"{node.name}_value_axis{axis}", np.array([constant_value], dtype=np.float32))
                        pad_inputs = [transpose1_output, pads_constant, value_constant]
                        pad_attrs = {}
                    else:
                        pad_inputs = [transpose1_output]
                        pad_attrs = {
                            "pads": transposed_pads.tolist(),
                            "value": constant_value,
                            "mode": "constant"
                        }
                    
                    # Create pad at height axis
                    pad_output = gs.Variable(f"{node.name}_pad_axis{axis}_output", dtype=input_tensor.dtype)
                    pad_node = gs.Node(op="Pad",
                                      name=f"{node.name}_pad_axis{axis}",
                                      attrs=pad_attrs,
                                      inputs=pad_inputs,
                                      outputs=[pad_output])
                    graph.nodes.append(pad_node)
                    
                    # Create transpose back
                    # If this is the last axis to process, use the original node's outputs
                    if i == len(padded_axes) - 1:
                        transpose2_node = gs.Node(op="Transpose",
                                                name=f"{node.name}_transpose2_axis{axis}",
                                                attrs={"perm": perm},
                                                inputs=[pad_output],
                                                outputs=node.outputs)
                    else:
                        transpose2_output = gs.Variable(f"{node.name}_transpose2_axis{axis}_output", dtype=input_tensor.dtype)
                        transpose2_node = gs.Node(op="Transpose",
                                                name=f"{node.name}_transpose2_axis{axis}",
                                                attrs={"perm": perm},
                                                inputs=[pad_output],
                                                outputs=[transpose2_output])
                    
                    graph.nodes.append(transpose2_node)
                    
                    # Update current tensor for next iteration
                    if i < len(padded_axes) - 1:
                        current_tensor = transpose2_output
                else:
                    # For axes >= height_axis, we can pad directly
                    
                    # Create pads specifically for this axis
                    single_axis_pads = np.zeros(2 * input_rank, dtype=np.int64)
                    single_axis_pads[axis] = pads_values[axis]
                    single_axis_pads[axis + input_rank] = pads_values[axis + input_rank]
                    
                    # Create pad constants for opset >= 11
                    if opset_version >= 11:
                        pads_constant = gs.Constant(f"{node.name}_pads_axis{axis}", np.array(single_axis_pads, dtype=np.int64))
                        value_constant = gs.Constant(f"{node.name}_value_axis{axis}", np.array([constant_value], dtype=np.float32))
                        pad_inputs = [current_tensor, pads_constant, value_constant]
                        pad_attrs = {}
                    else:
                        pad_inputs = [current_tensor]
                        pad_attrs = {
                            "pads": single_axis_pads.tolist(),
                            "value": constant_value,
                            "mode": "constant"
                        }
                    
                    # Create pad node
                    # If this is the last axis to process, use the original node's outputs
                    if i == len(padded_axes) - 1:
                        pad_node = gs.Node(op="Pad",
                                          name=f"{node.name}_pad_axis{axis}",
                                          attrs=pad_attrs,
                                          inputs=pad_inputs,
                                          outputs=node.outputs)
                    else:
                        pad_output = gs.Variable(f"{node.name}_pad_axis{axis}_output", dtype=input_tensor.dtype)
                        pad_node = gs.Node(op="Pad",
                                          name=f"{node.name}_pad_axis{axis}",
                                          attrs=pad_attrs,
                                          inputs=pad_inputs,
                                          outputs=[pad_output])
                    
                    graph.nodes.append(pad_node)
                    
                    # Update current tensor for next iteration
                    if i < len(padded_axes) - 1:
                        current_tensor = pad_output
            
            # Disconnect the original node
            node.inputs.clear()
            node.outputs.clear()
        else:
            # Single axis pad
            axis = padded_axes[0]
            
            # Check if this axis is to the left of the height axis (axis < len(shape) - 2)
            if axis < height_axis:
                logging.info(f"Converting Pad node {node.name} at axis {axis} to height axis ({height_axis})")
                
                # For axes to the left of height, we need to transpose, pad, transpose back
                
                # Create permutation that brings this axis to height position (height_axis)
                # and height to this axis's position
                perm = list(range(input_rank))
                perm[axis], perm[height_axis] = perm[height_axis], perm[axis]
                
                # Create transpose to make axis the height
                transpose1_output = gs.Variable(f"{node.name}_transpose1_output", dtype=input_tensor.dtype)
                transpose1_node = gs.Node(op="Transpose",
                                        name=f"{node.name}_transpose1",
                                        attrs={"perm": perm},
                                        inputs=[input_tensor],
                                        outputs=[transpose1_output])
                graph.nodes.append(transpose1_node)
                
                # Calculate new pad values after transpose
                transposed_pads = np.zeros(2 * input_rank, dtype=np.int64)
                transposed_pads[height_axis] = pads_values[axis]
                transposed_pads[height_axis + input_rank] = pads_values[axis + input_rank]
                
                # Create pad constants for opset >= 11
                if opset_version >= 11:
                    pads_constant = gs.Constant(f"{node.name}_pads", np.array(transposed_pads, dtype=np.int64))
                    value_constant = gs.Constant(f"{node.name}_value", np.array([constant_value], dtype=np.float32))
                    pad_inputs = [transpose1_output, pads_constant, value_constant]
                    pad_attrs = {}
                else:
                    pad_inputs = [transpose1_output]
                    pad_attrs = {
                        "pads": transposed_pads.tolist(),
                        "value": constant_value,
                        "mode": "constant"
                    }
                
                # Create pad at height axis
                pad_output = gs.Variable(f"{node.name}_pad_output", dtype=input_tensor.dtype)
                pad_node = gs.Node(op="Pad",
                                  name=f"{node.name}_pad",
                                  attrs=pad_attrs,
                                  inputs=pad_inputs,
                                  outputs=[pad_output])
                graph.nodes.append(pad_node)
                
                # Create transpose back
                transpose2_node = gs.Node(op="Transpose",
                                        name=f"{node.name}_transpose2",
                                        attrs={"perm": perm},
                                        inputs=[pad_output],
                                        outputs=node.outputs)
                graph.nodes.append(transpose2_node)
                
                # Disconnect the original node
                node.inputs.clear()
                node.outputs.clear()
            else:
                # For axes >= height_axis, we can use the original node
                logging.info(f"Pad node {node.name} operates on axis {axis} which is >= height axis ({height_axis}), keeping as is")
                # Keep the original node as is


def tidl_convert_nonzero_constant_pad_to_zero_pad_add(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    pads = [node for node in graph.nodes if node.op == 'Pad']
    for pad in pads:
        if pad.attrs.get('mode','constant') != 'constant':
            continue
        if pad.outputs[0].shape is None:
            continue
        if graph.opset>=11:
            if len(pad.inputs)<3 :
                continue
            if not isinstance(pad.inputs[1], gs.Constant):
                continue
            pads = pad.inputs[1].values.tolist()
            const = pad.inputs[2]
            if isinstance(const, gs.Constant):
                if const.values == 0:
                    continue
                const_val = np.copy(const.values)
                pad.inputs[2] = gs.Constant(f'{pad.name}_const_val', np.copy(const_val))
                const = pad.inputs[2]
                const.values *= 0
            else:
                if len(const.inputs) == 0:
                    continue
                const_node = const.inputs[0] 
                if const_node.op != 'Constant':
                    continue
                if 'value' in const_node.attrs:
                    const = const_node.attrs['value']
                    if not isinstance(const , gs.Constant):
                        continue
                    const_node.attrs['value'] = gs.Constant(f'{pad.name}_const_val', np.copy(const.values))
                    const_val = np.copy(const.values)
                    const_node.attrs['value'].values *= 0
                if 'value_float' in const_node.attrs:
                    const_val = np.copy(const_node.attrs['value_float'])
                    const_node.attrs['value_float'] *= 0
                if 'value_int' in const_node.attrs:
                    const_val = np.copy(const_node.attrs['value_int'])
                    const_node.attrs['value_int'] *= 0
        else:
            pads = pad.attrs['pads']
            const_val = pad.attrs.get('value', 0)
            if  const_val == 0:
                continue
            pads.attrs['value'] = 0
        
        pad_out = pad.outputs[0]
        const_val = np.array(const_val) if not isinstance(const_val, np.ndarray) else const_val
        add_in = gs.Variable(f'{pad.name}_add_in', pad_out.dtype, pad_out.shape)
        pad_const = np.zeros(pad_out.shape, const_val.dtype)
        d = len(pad_out.shape)
        for i in range (d):
            start_pad = pads[i]
            end_pad = pads[i+d]
            if start_pad == 0 and end_pad == 0:
                continue
            # Build slice objects dynamically
            slices = [slice(None)] * d  # Start with [:, :, :, ...]
            
            # Fill start padding
            if start_pad > 0:
                slices[i] = slice(0, start_pad)
                pad_const[tuple(slices)] = const_val
            
            # Fill end padding
            if end_pad > 0:
                slices[i] = slice(-end_pad, None)
                pad_const[tuple(slices)] = const_val
        pad_const = gs.Constant(f'{pad.name}_pad_const', pad_const)
        pad.outputs[0] = add_in
        add = gs.Node('Add', f'{pad.name}_add', {}, [add_in, pad_const], [pad_out])
        graph.nodes.append(add)
        