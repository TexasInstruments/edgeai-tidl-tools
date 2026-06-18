import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np

def tidl_adjust_clip_minval_maxval(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    '''
    to adjust clip with minval >0 or maxval < 0 using adds
    for example clip(x,minval=a) where a >0 changes to clip(x-a, minval=0)+a
    for example clip(x,maxval=a) where a <0 changes to clip(x+a, maxval=0)-a
    '''
    # Get the opset version directly from the graph
    opset_version = graph.opset
    logging.debug(f"ONNX opset version: {opset_version}")
    
    # Get all Clip nodes in the graph
    clip_nodes = [node for node in graph.nodes if node.op == 'Clip']
    logging.debug(f"Found {len(clip_nodes)} Clip nodes to adjust")
    
    for node in clip_nodes:
        # Skip nodes with no inputs
        if len(node.inputs) == 0 or node.inputs[0] is None:
            logging.debug(f"Clip node {node.name} has no inputs, skipping")
            continue
        
        # Get the input tensor
        input_tensor = node.inputs[0]
        
        # Initialize min and max values to default
        min_value = None
        max_value = None
        
        # For opset >= 11, min/max are provided as inputs
        if opset_version >= 11:
            # Check if min value is provided (second input)
            if len(node.inputs) > 1 and node.inputs[1] is not None:
                min_tensor = node.inputs[1]
                if isinstance(min_tensor, gs.Constant):
                    min_value = min_tensor.values
                    logging.debug(f"Clip node {node.name} has min value: {min_value} (from input)")
            
            # Check if max value is provided (third input)
            if len(node.inputs) > 2 and node.inputs[2] is not None:
                max_tensor = node.inputs[2]
                if isinstance(max_tensor, gs.Constant):
                    max_value = max_tensor.values
                    logging.debug(f"Clip node {node.name} has max value: {max_value} (from input)")
        # For opset < 11, min/max are provided as attributes
        else:
            # Check for min attribute
            if 'min' in node.attrs:
                min_value = node.attrs['min']
                logging.debug(f"Clip node {node.name} has min value: {min_value} (from attribute)")
            
            # Check for max attribute
            if 'max' in node.attrs:
                max_value = node.attrs['max']
                logging.debug(f"Clip node {node.name} has max value: {max_value} (from attribute)")
        
        # Create a unique name for the node
        node_name = node.name if node.name else f"clip_{id(node)}"
        
        # Case 1: min value > 0
        if min_value is not None and min_value > 0:
            # 1. Create Subtract node: x - min_value
            sub_output = gs.Variable(f"{node_name}_sub_output", dtype=input_tensor.dtype)
            min_constant = gs.Constant(f"{node_name}_min_const", np.array(min_value, dtype=np.float32))
            sub_node = gs.Node(op="Sub", 
                              name=f"{node_name}_sub",
                              inputs=[input_tensor, min_constant],
                              outputs=[sub_output])
            graph.nodes.append(sub_node)
            
            # 2. Create Clip node with min=0
            clip_output = gs.Variable(f"{node_name}_clip_output", dtype=input_tensor.dtype)
            zero_constant = gs.Constant(f"{node_name}_zero", np.array(0, dtype=np.float32))
            
            # Create a new Clip node based on opset version
            if opset_version >= 11:
                # For opset >= 11, provide min/max as inputs
                clip_inputs = [sub_output, zero_constant]
                if len(node.inputs) > 2 and node.inputs[2] is not None:
                    # Add original max value
                    clip_inputs.append(node.inputs[2])
                
                clip_node = gs.Node(op="Clip",
                                  name=f"{node_name}_clip",
                                  inputs=clip_inputs,
                                  outputs=[clip_output])
            else:
                # For opset < 11, provide min/max as attributes
                clip_attrs = {'min': 0.0}
                if max_value is not None:
                    clip_attrs['max'] = max_value
                
                clip_node = gs.Node(op="Clip",
                                  name=f"{node_name}_clip",
                                  attrs=clip_attrs,
                                  inputs=[sub_output],
                                  outputs=[clip_output])
            graph.nodes.append(clip_node)
            
            # 3. Create Add node to add min_value back
            add_node = gs.Node(op="Add",
                              name=f"{node_name}_add",
                              inputs=[clip_output, min_constant],
                              outputs=node.outputs)
            graph.nodes.append(add_node)
            
            # 4. Disconnect the original Clip node
            node.inputs.clear()
            node.outputs.clear()
            
            logging.debug(f"Adjusted Clip node {node_name} with min value {min_value} > 0")
            
        # Case 2: max value < 0
        elif max_value is not None and max_value < 0:
            # 1. Create Add node: x + (-max_value)
            add_output = gs.Variable(f"{node_name}_add_output", dtype=input_tensor.dtype)
            neg_max_constant = gs.Constant(f"{node_name}_neg_max_const", np.array(-max_value, dtype=np.float32))
            add_node = gs.Node(op="Add", 
                              name=f"{node_name}_add",
                              inputs=[input_tensor, neg_max_constant],
                              outputs=[add_output])
            graph.nodes.append(add_node)
            
            # 2. Create Clip node with max=0
            clip_output = gs.Variable(f"{node_name}_clip_output", dtype=input_tensor.dtype)
            zero_constant = gs.Constant(f"{node_name}_zero", np.array(0, dtype=np.float32))
            
            # Create a new Clip node based on opset version
            if opset_version >= 11:
                # For opset >= 11, provide min/max as inputs
                clip_inputs = [add_output]
                if len(node.inputs) > 1 and node.inputs[1] is not None:
                    # Add original min value
                    clip_inputs.append(node.inputs[1])
                clip_inputs.append(zero_constant)
                
                clip_node = gs.Node(op="Clip",
                                  name=f"{node_name}_clip",
                                  inputs=clip_inputs,
                                  outputs=[clip_output])
            else:
                # For opset < 11, provide min/max as attributes
                clip_attrs = {'max': 0.0}
                if min_value is not None:
                    clip_attrs['min'] = min_value
                
                clip_node = gs.Node(op="Clip",
                                  name=f"{node_name}_clip",
                                  attrs=clip_attrs,
                                  inputs=[add_output],
                                  outputs=[clip_output])
            graph.nodes.append(clip_node)
            
            # 3. Create Sub node to subtract -max_value back
            sub_node = gs.Node(op="Sub",
                              name=f"{node_name}_sub",
                              inputs=[clip_output, neg_max_constant],
                              outputs=node.outputs)
            graph.nodes.append(sub_node)
            
            # 4. Disconnect the original Clip node
            node.inputs.clear()
            node.outputs.clear()
            
            logging.debug(f"Adjusted Clip node {node_name} with max value {max_value} < 0")