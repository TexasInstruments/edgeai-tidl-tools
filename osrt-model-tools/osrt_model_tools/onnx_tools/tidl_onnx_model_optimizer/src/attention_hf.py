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
import logging
from typing import List, Tuple, Optional, Dict

def find_qk_matmul(softmax: gs.Node) -> Optional[gs.Node]:
    """Find Q·K^T MatMul before Softmax"""
    current = softmax
    for _ in range(10):
        var_inputs = [inp for inp in current.inputs if isinstance(inp, gs.Variable)]
        if not var_inputs or not var_inputs[0].inputs:
            return None
        prev = var_inputs[0].inputs[0]
        if prev.op == 'MatMul':
            return prev
        if prev.op not in ('Div', 'Mul', 'Add', 'Reshape', 'Transpose'):
            return None
        current = prev
    return None

def find_attention_matmul(softmax: gs.Node) -> Optional[gs.Node]:
    """Find Attention·V MatMul after Softmax"""
    if not softmax.outputs or not softmax.outputs[0].outputs:
        return None
    if len(softmax.outputs[0].outputs) != 1:
        return None
    next_node = softmax.outputs[0].outputs[0]
    return next_node if next_node.op == 'MatMul' else None

def get_const(var) -> Optional[np.ndarray]:
    """Get constant value from variable or constant."""
    if isinstance(var, gs.Constant):
        return var.values
    if isinstance(var, gs.Variable) and var.inputs and isinstance(var.inputs[0], gs.Constant):
        return var.inputs[0].values
    return None

def extract_gather_debug(gathers: List[gs.Node]) -> Optional[List[Dict]]:
    """
    Extract indices and axis debug from Gather nodes.
    """
    debug = []
    
    for gather in gathers:
        # Find indices constant
        indices = next((inp.values for inp in gather.inputs if isinstance(inp, gs.Constant)), None)
        if indices is None:
            logging.debug(f"    Gather {gather.name} has no constant indices")
            return None
        
        axis = gather.attrs.get('axis', 0)
        
        # Determine if scalar or array
        if indices.size == 1:  # Scalar index: [0], [1], or [2]
            debug.append({
                'gather': gather,
                'type': 'scalar',
                'index': int(indices.flat[0]),
                'axis': axis
            })
        else:  # Array of indices: [0, 1, 2, ...]
            debug.append({
                'gather': gather,
                'type': 'array',
                'indices': indices,
                'axis': axis,
                'start': int(indices[0]),
                'end': int(indices[-1]) + 1,
                'size': len(indices)
            })
    
    # Verify all use same axis
    axes = [g['axis'] for g in debug]
    if len(set(axes)) != 1:
        logging.debug(f"    Gathers use different axes: {axes}")
        return None
    
    return debug

def handle_scalar_gather_pattern(graph: gs.Graph, common_node: gs.Node, debug: List[Dict], path: List[gs.Node]) -> Optional[gs.Node]:
    """
    Handle pattern: Gather with scalar indices [0], [1], [2] if Input shape is  [B, S, 3, D] and creates equal Split along the axis and needed then add the reshape to remove extra dimensions.
    """
    axis = debug[0]['axis']
    num_outputs = len(debug)
    
    # Get input shape
    input_tensor = common_node.outputs[0]
    if not hasattr(input_tensor, 'shape') or input_tensor.shape is None:
        logging.warning("Cannot get input shape for Gather pattern")
        return None
    
    input_shape = list(input_tensor.shape)
    rank = len(input_shape)
    
    # Convert negative axis
    if axis < 0:
        axis = rank + axis
    
    if axis >= rank:
        logging.warning(f"Axis {axis} out of bounds for rank {rank}")
        return None
    
    dimension = input_shape[axis]
    
    # Validate dimension
    if not isinstance(dimension, (int, np.integer)) or dimension <= 0:
        logging.warning(f"Dimension at axis {axis} is dynamic: {dimension}")
        return None
    
    # Determine if Reshape is needed
    needs_reshape = (dimension == num_outputs)
    
    split_outs = [g['gather'].outputs[0] for g in debug]
    split_input = input_tensor
    
    # Add Reshape to merge dimensions
    if needs_reshape:
        if axis == rank - 1:
            # Last axis: merge with previous
            if axis == 0:
                logging.warning("Cannot merge single dimension")
                return None
            
            dim_prev = input_shape[axis - 1]
            if not isinstance(dim_prev, (int, np.integer)) or dim_prev <= 0:
                logging.warning(f"Previous dimension is dynamic: {dim_prev}")
                return None
            
            merged_dim = dim_prev * dimension
            new_shape = input_shape[:-2] + [merged_dim]
            split_axis = axis - 1
            split_size = dim_prev
            
        else:
            # Not last: merge with next
            if axis + 1 >= rank:
                logging.warning("Cannot merge with next dimension")
                return None
            
            dim_next = input_shape[axis + 1]
            if not isinstance(dim_next, (int, np.integer)) or dim_next <= 0:
                logging.warning(f"Next dimension is dynamic: {dim_next}")
                return None
            
            merged_dim = dimension * dim_next
            new_shape = input_shape[:axis] + [merged_dim] + input_shape[axis+2:]
            split_axis = axis
            split_size = dim_next
        
        # Create Reshape node
        reshaped_var = gs.Variable(name=f"{common_node.name}_gather_reshape_out")
        reshape_node = gs.Node(
            op='Reshape',
            name=f'{common_node.name}_gather_reshape',
            inputs=[
                input_tensor,
                gs.Constant(
                    name=f'{common_node.name}_reshape_shape',
                    values=np.array(new_shape, dtype=np.int64)
                )
            ],
            outputs=[reshaped_var]
        )
        graph.nodes.append(reshape_node)
        
        # Update split input to use reshaped tensor
        split_input = reshaped_var
        split_sizes = [int(split_size)] * num_outputs
        
        logging.debug(f"Added Reshape: {input_shape} → {new_shape}")
    
    # No Reshape needed
    else:
        split_axis = axis
        split_size = dimension // num_outputs
        split_sizes = [int(split_size)] * num_outputs
        
        logging.debug(f"No Reshape needed (dimension {dimension} > {num_outputs})")
    
    #---------------------------------------------------------------------------
    # Create Split Node (always created)
    #---------------------------------------------------------------------------
    if graph.opset >= 13:
        split_node = gs.Node(
            op='Split',
            name=f'{common_node.name}_gather_to_split',
            inputs=[
                split_input,
                gs.Constant(
                    name=f'{common_node.name}_split_sizes',
                    values=np.array(split_sizes, dtype=np.int64)
                )
            ],
            outputs=split_outs,
            attrs={'axis': split_axis}
        )
    else:
        split_node = gs.Node(
            op='Split',
            name=f'{common_node.name}_gather_to_split',
            inputs=[
                split_input,
            ],
            outputs=split_outs,
            attrs={'axis': split_axis,
                   'split': split_sizes }
        )

    graph.nodes.append(split_node)
    
    logging.debug(f"Created Split: axis={split_axis}, sizes={split_sizes}")
    
    # Clear old Gathers
    for g_debug in debug:
        g_debug['gather'].inputs.clear()
        g_debug['gather'].outputs.clear()
    
    path[-1] = split_node
    return split_node

def handle_array_gather_pattern(graph: gs.Graph, common_node: gs.Node,debug: List[Dict], path: List[gs.Node]) -> Optional[gs.Node]:
    """
    Handle pattern: Gather with contiguous array indices
    """
    # Sort by start index
    debug.sort(key=lambda x: x['start'])
    
    # Verify contiguous (no gaps)
    for i in range(len(debug) - 1):
        if debug[i]['end'] != debug[i+1]['start']:
            return None
    #----------------------------------------------------------------------------
    # Create the Split Node 
    #----------------------------------------------------------------------------
    axis = debug[0]['axis']
    sizes = [g['size'] for g in debug]
    
    split_outs = [g['gather'].outputs[0] for g in debug]
    
    if graph.opset >=13:
        split_node = gs.Node(
            op='Split',
            name=f'{common_node.name}_gather_to_split',
            inputs=[
                common_node.outputs[0],
                gs.Constant(
                    name=f'{common_node.name}_split_sizes',
                    values=np.array(sizes, dtype=np.int64)
                )
            ],
            outputs=split_outs,
            attrs={'axis': axis}
        )
    else :
        split_node = gs.Node(
            op='Split',
            name=f'{common_node.name}_gather_to_split',
            inputs=[common_node.outputs[0]],
            outputs=split_outs,
            attrs={'axis': axis, 'split': sizes}
        )
    
    graph.nodes.append(split_node)
    
    for g_debug in debug:
        g_debug['gather'].inputs.clear()
        g_debug['gather'].outputs.clear()
    
    # Update path
    path[-1] = split_node
    return split_node

def replace_gathers_with_split(graph: gs.Graph, common_node: gs.Node, gathers: List[gs.Node], path: List[gs.Node]) -> Optional[gs.Node]:
    """
    Replace 3 Gather nodes with a Split node.
    """
    # Extract indices debug
    debug = extract_gather_debug(gathers)
    if not debug:
        return None
    
    # Detect pattern type
    if all(g['type'] == 'scalar' for g in debug):
        return handle_scalar_gather_pattern(graph, common_node, debug, path)
    
    elif all(g['type'] == 'array' for g in debug):
        return handle_array_gather_pattern(graph, common_node, debug, path)
    
    else:
        logging.debug(f"Gather types - cannot fuse")
        return None

def replace_slices_with_split(graph: gs.Graph, common_node: gs.Node, slices: List[gs.Node], path: List[gs.Node]) -> Optional[gs.Node]:
    """
    Replace 3 Slice nodes with a Split node.
    """
    # Extract slice debug
    debug = []
    
    for s in slices:
        if len(s.inputs) >= 3: # For opset 10+
            starts = get_const(s.inputs[1])
            ends = get_const(s.inputs[2])
            axes = get_const(s.inputs[3]) if len(s.inputs) > 3 else np.array([0])
            steps = get_const(s.inputs[4]) if len(s.inputs) > 4 else np.array([1])

        elif 'starts' in s.attrs and 'ends' in s.attrs: # for opset 1-9
            starts = np.array(s.attrs['starts'])
            ends = np.array(s.attrs['ends'])
            axes = np.array(s.attrs.get('axes', [0]))
            steps = np.array([1])
        
        else:
            return None
        
        # Validate
        if starts is None or ends is None or len(starts) != 1 or steps[0] != 1:
            return None
        
        debug.append({
            'slice': s,
            'start': int(starts[0]),
            'end': int(ends[0]),
            'axis': int(axes[0])
        })
    
    # Validate all same axis
    if len(set(i['axis'] for i in debug)) != 1:
        return None
    
    # Sort by start
    debug.sort(key=lambda x: x['start'])
    for i in range(len(debug) - 1):
        if debug[i]['end'] != debug[i+1]['start']:
            logging.debug(f"Slices not contiguous: {debug[i]['end']} != {debug[i+1]['start']}")
            return None
    
    #----------------------------------------------------------------------------
    # Create the Split Node 
    #----------------------------------------------------------------------------
    axis = debug[0]['axis']
    sizes = [i['end'] - i['start'] for i in debug]
    split_outs = [i['slice'].outputs[0] for i in debug]
    
    if graph.opset>=13:
        split = gs.Node(
            op='Split',
            name=f'{common_node.name}_slice_to_split',
            inputs=[
                common_node.outputs[0],
                gs.Constant(
                    name=f'{common_node.name}_split_sizes',
                    values=np.array(sizes, dtype=np.int64)
                )
            ],
            outputs=split_outs,
            attrs={'axis': axis}
        )
    else:
        split = gs.Node(
            op='Split',
            name=f'{common_node.name}_slice_to_split',
            inputs=[common_node.outputs[0]],
            outputs=split_outs,
            attrs={'axis': axis, 'split': sizes}
        )
    graph.nodes.append(split)
    
    # Clear Slice nodes
    for i in debug:
        i['slice'].inputs.clear()
        i['slice'].outputs.clear()
    
    path[-1]=split
    
    return split


def trace_to_split_or_common_source(graph: gs.Graph, start_var: gs.Variable) -> Tuple[List[gs.Node], Optional[gs.Node], int, str]:
    """
    Trace backward to Split OR common source node 
    Returns: (path, source_node, output_index, source_type)
    """
    if not isinstance(start_var, gs.Variable) or not start_var.inputs:
        return [], None, None
    
    path = []
    current = start_var
    
    for _ in range(20):
        if not current.inputs :
            return path, None, 'common'
        
        node = current.inputs[0]
        path.append(node)
        
        # Check for existing Split
        if node.op == 'Split':
            return path, node, 'split'
        
        # Check for common source (node with 3 outputs)
        if len(node.outputs) > 0 and len(node.outputs[0].outputs) == 3:
            consumers = list(node.outputs[0].outputs)
            path = path [:-1]
            # Pattern 1: Common -> 3 Gathers (replace with Split)
            if all(c.op == 'Gather' for c in consumers):
                if not all(c.outputs and c.outputs[0].shape is not None and list(c.outputs[0].shape) == list(consumers[0].outputs[0].shape) for c in consumers):
                    return path, None, None
                else:
                    split_node = replace_gathers_with_split(graph, node, consumers, path)
                    if split_node:
                        return path, split_node, 'split'
                    else:
                        return path, None, None
            
            # Pattern 2: Common -> 3 Slice (replace with Split)
            if all(c.op == 'Slice' for c in consumers):
                if not all(c.outputs and c.outputs[0].shape is not None and list(c.outputs[0].shape) == list(consumers[0].outputs[0].shape) for c in consumers):
                    return path, None, None
                else:
                    split_node = replace_slices_with_split(graph, node, consumers, path)
                    if split_node:
                        return path, split_node, 'split'
                    else:
                        return path, None, None

            # Pattern 3: Common → 3 common operation
            elif all(c.op == consumers[0].op for c in consumers):
                    return path, node, 'common'
            
        # Continue tracing
        if node.op not in ('Add', 'Reshape', 'Transpose', 'Mul', 'Div', 'MatMul', 'Gather', 'Slice'):
            return path, None , None
        
        var_inputs = [inp for inp in node.inputs if isinstance(inp, gs.Variable)]
        if len(var_inputs) != 1:
            return path, None, None
        
        current = var_inputs[0]

        if not current.inputs and current.outputs and path:
            return path, current, 'common'
    
    return path, None, None

def find_attention_patterns(graph: gs.Graph) -> List[Dict]:
    """Find attention patterns - handles both Split and non-Split patterns"""
    patterns = []
    
    for softmax in graph.nodes:
        if softmax.op != 'Softmax':
            continue
        
        qk_matmul = find_qk_matmul(softmax)
        if not qk_matmul:
            continue
        
        attn_matmul = find_attention_matmul(softmax)
        if not attn_matmul:
            continue
        
        # Trace COMPLETE branches
        q_path, q_source, q_type = trace_to_split_or_common_source(graph, qk_matmul.inputs[0])
        k_path, k_source, k_type = trace_to_split_or_common_source(graph, qk_matmul.inputs[1])
        
        v_input = attn_matmul.inputs[1] if (attn_matmul.inputs[0].inputs and 
                  attn_matmul.inputs[0].inputs[0] is softmax) else attn_matmul.inputs[0]
        v_path, v_source, v_type = trace_to_split_or_common_source(graph, v_input)
        
        # Check if all branches come from same source
        if not (q_source and k_source and v_source and q_source is k_source is v_source):
            continue
        
        source_node = q_source
        source_type = q_type
        
        # Remove source node from paths and reverse to get forward direction
        q_ops = [n for n in reversed(q_path) if n is not source_node]
        k_ops = [n for n in reversed(k_path) if n is not source_node]
        v_ops = [n for n in reversed(v_path) if n is not source_node]
        
        if source_type == 'common':
            consumers = list(source_node.outputs[0].outputs)
            
            patterns.append({
                'source': source_node,
                'source_type': 'common',
                'consumers': consumers,
                'q_ops': q_ops,
                'k_ops': k_ops,
                'v_ops': v_ops,
                'qk_matmul': qk_matmul,
                'softmax': softmax
            })
        
        elif source_type == 'split':
            # Find common operations to merge
            min_len = min(len(q_ops), len(k_ops), len(v_ops))
            num_common = 0
            for i in range(min_len):
                q_op = q_ops[i].op
                k_op = k_ops[i].op
                v_op = v_ops[i].op
                
                if q_op == k_op == v_op and q_op in ('Add', 'Reshape', 'Transpose'):
                    num_common += 1
                else:
                    break
            
            if num_common == 0:
                continue
            
            patterns.append({
                'source': source_node,
                'source_type': 'split',
                'q_ops': q_ops,
                'k_ops': k_ops,
                'v_ops': v_ops,
                'qk_matmul': qk_matmul,
                'softmax': softmax
            })
    
    return patterns

def fuse_matmul(graph: gs.Graph, pattern: Dict, idx: int) -> bool:
    """
    Fuse MatMul at position idx.
    """
    q_matmul = pattern['q_ops'][idx]
    k_matmul = pattern['k_ops'][idx]
    v_matmul = pattern['v_ops'][idx]

    # Check if all MatMul input shapes are the same
    matmul_inputs = [m.inputs[0] for m in [q_matmul, k_matmul, v_matmul]]
    if not all(inp.shape is not None and list(inp.shape) == list(matmul_inputs[0].shape) for inp in matmul_inputs):
        logging.debug(f"    MatMul input shapes are different or undefined")
        return False
    
    # Extract weights
    weights = []
    matmul_outs=[]
    for m in [q_matmul, k_matmul, v_matmul]:
        weight = next((inp.values for inp in m.inputs if isinstance(inp, gs.Constant)), None)
        weights.append(weight)
        matmul_outs.append(m.outputs[0])
    
    # Validate
    if not all(w.shape[0] == weights[0].shape[0] for w in weights):
        logging.debug(f"    MatMul weights have different input dims")
        return False
    
    # Concatenate weights
    fused_weight = np.concatenate(weights, axis=-1)
    head_dims = [w.shape[-1] for w in weights]
    #--------------------------------------------------------------------------
    # Create new Fused Matmul 
    #--------------------------------------------------------------------------
    if pattern['source_type'] == 'common':
        if(type(pattern['source'])==gs.Variable):
            current_input = pattern['source']
        else:
            current_input = pattern['source'].outputs[0]
    else:
        current_input = pattern['source'].inputs[0]
    fused_out = gs.Variable(f'{q_matmul.name}_fused_out')
    
    fused_matmul = gs.Node(
                        op = 'MatMul', 
                        name = f'{q_matmul.name}_fused',
                        inputs = [current_input, gs.Constant(f'{q_matmul.name}_w', fused_weight)],
                        outputs =  [fused_out]
                    )
    graph.nodes.append(fused_matmul)
    if(pattern['source_type']=='common'):
        #--------------------------------------------------------------------------
        # Create new Split Node
        #--------------------------------------------------------------------------
        if graph.opset>=13:
            split = gs.Node(
                        op = 'Split', 
                        name = f'new_{fused_matmul.name}_split',
                        inputs= [fused_out, gs.Constant(f'new_{fused_matmul.name}_sz', np.array(head_dims, np.int64))],
                        outputs = matmul_outs, 
                        attrs={'axis': -1}
                    )
        else:
            split = gs.Node(
                        op = 'Split', 
                        name = f'new_{fused_matmul.name}_split',
                        inputs= [fused_out],
                        outputs = matmul_outs, 
                        attrs={'axis': -1, 'split': head_dims}
                    )
        
        pattern['source_type']='split'
        pattern['source']= split
        graph.nodes.append(split)

    else:
        pattern['source'].inputs[0]=fused_out
        pattern['source'].outputs = matmul_outs

    # Clear old MatMuls
    for m in [q_matmul, k_matmul, v_matmul]:
        m.inputs.clear()
        m.outputs.clear()

    return True

def fuse_add(graph: gs.Graph, pattern: Dict, idx: int) -> bool:
    """
    Fuse Add Operation
    """
    q_add = pattern['q_ops'][idx]
    k_add = pattern['k_ops'][idx]
    v_add = pattern['v_ops'][idx]
    
    # Check if all Add input shapes are the same
    add_inputs = [m.inputs[0] for m in [q_add, k_add, v_add]]
    if not all(inp.shape is not None and list(inp.shape) == list(add_inputs[0].shape) for inp in add_inputs):
        logging.debug(f"    Add input shapes are different or undefined")
        return False

    # Extract biases and outputs
    biases = []
    add_outs = []
    for a in [q_add, k_add, v_add]:
        bias = next((inp.values for inp in a.inputs if isinstance(inp, gs.Constant)), None)
        biases.append(bias)
        add_outs.append(a.outputs[0])
    
    # Concatenate biases
    fused_bias = np.concatenate(biases, axis=-1)
    bias_dims = [b.shape[-1] for b in biases]

    # Determine input for fused Add
    if pattern['source_type'] == 'common':
        if(type(pattern['source'])==gs.Variable):
            current_input = pattern['source']
        else:
            current_input = pattern['source'].outputs[0]
    else:
        current_input = pattern['source'].inputs[0]
    
    #--------------------------------------------------------------------------
    # Create New fused ADD Node
    #--------------------------------------------------------------------------
    fused_out = gs.Variable(f'{q_add.name}_fused_out')
    fused_add = gs.Node(
        op='Add',
        name=f'{q_add.name}_fused',
        inputs=[current_input, gs.Constant(f'{q_add.name}_bias', fused_bias)],
        outputs=[fused_out]
    )
    
    graph.nodes.append(fused_add)
    if pattern['source_type'] == 'common':
        #----------------------------------------------------------------------
        # Create NEW Split node
        #----------------------------------------------------------------------
        if graph.opset >=13 :
            split = gs.Node(
                op='Split',
                name=f'{fused_add.name}_split',
                inputs=[fused_out, gs.Constant(f'{fused_add.name}_sz', np.array(bias_dims, np.int64))],
                outputs=add_outs,
                attrs={'axis': -1}
            )
        else:
            split = gs.Node(
                op='Split',
                name=f'{fused_add.name}_split',
                inputs=[fused_out],
                outputs=add_outs,
                attrs={'axis': -1, 'split' : bias_dims}
            )
        
        graph.nodes.append(split)
        
        # Update pattern to 'split' type
        pattern['source'] = split
        pattern['source_type'] = 'split'
    
    else:
        # Update the split input and output connection
        pattern['source'].inputs[0] = fused_out
        pattern['source'].outputs = add_outs
        
    # Clear old Adds
    for a in [q_add, k_add, v_add]:
        a.inputs.clear()
        a.outputs.clear()
    
    return True

def fuse_reshape(graph: gs.Graph, pattern: Dict, idx: int) -> bool:
    """
    Fuse Reshape at position idx.
    """
    q_reshape = pattern['q_ops'][idx]
    k_reshape = pattern['k_ops'][idx]
    v_reshape = pattern['v_ops'][idx]

    # Check if all Add input shapes are the same
    reshape_inputs = [m.inputs[0] for m in [q_reshape, k_reshape, v_reshape]]
    if not all(inp.shape is not None and list(inp.shape) == list(reshape_inputs[0].shape) for inp in reshape_inputs):
        logging.debug(f"Reshape input shapes are different or undefined")
        return False
    
    # Extract target shapes
    q_shape = get_const(q_reshape.inputs[1]) if len(q_reshape.inputs) > 1 else None
    k_shape = get_const(k_reshape.inputs[1]) if len(k_reshape.inputs) > 1 else None
    v_shape = get_const(v_reshape.inputs[1]) if len(v_reshape.inputs) > 1 else None
    
    if q_shape is None or k_shape is None or v_shape is None:
        logging.warning("Cannot get reshape target shapes")
        return False
    
    # Validate all shapes are identical
    if not (np.array_equal(q_shape, k_shape) and np.array_equal(k_shape, v_shape)):
        logging.warning(f"Reshape shapes are not identical")
        return False
    
    reshape_outs = [q_reshape.outputs[0], k_reshape.outputs[0], v_reshape.outputs[0]]
    target_shape = list(q_shape.copy())
    
    # Determine input for fused Reshape and Shape
    if pattern['source_type'] == 'common':
        if isinstance(pattern['source'], gs.Variable):
            current_input = pattern['source']
        else:
            current_input = pattern['source'].outputs[0]
        fused_shape = target_shape
        
    else:  # source_type == 'split'
        current_input = pattern['source'].inputs[0]

        # Get old split axis
        old_split_axis = pattern['source'].attrs.get('axis', -1)

        input_shape = list(q_reshape.inputs[0].shape)

        # Convert negative axis to positive
        if old_split_axis < 0:
            old_split_axis = len(input_shape) + old_split_axis

        split_dim_value = input_shape[old_split_axis]

        # Find where the split-axis dimension lands in the target shape.
        # The reshape may add/remove dims (e.g. [12,197,64] → [1,12,197,64]).
        # We locate the split dim by matching prefix/suffix element products.
        prefix_prod = int(np.prod(input_shape[:old_split_axis])) if old_split_axis > 0 else 1
        suffix_prod = int(np.prod(input_shape[old_split_axis + 1:])) if old_split_axis + 1 < len(input_shape) else 1

        new_split_axis = None
        for pos in range(len(target_shape)):
            t_pre = int(np.prod(target_shape[:pos])) if pos > 0 else 1
            t_suf = int(np.prod(target_shape[pos + 1:])) if pos + 1 < len(target_shape) else 1
            if (target_shape[pos] == split_dim_value and
                    t_pre == prefix_prod and t_suf == suffix_prod):
                new_split_axis = pos
                break

        if new_split_axis is None:
            logging.debug(f"Cannot map split-axis dim {split_dim_value} from {input_shape} to {target_shape}, skipping fuse")
            return False

        # Build fused shape: multiply the split-axis dim by 3
        fused_shape = target_shape.copy()
        fused_shape[new_split_axis] = target_shape[new_split_axis] * 3

        # Split sizes and update axis to new position
        split_size = target_shape[new_split_axis]
        split_sizes = [split_size] * 3
        old_split_axis = new_split_axis
    
    #--------------------------------------------------------------------------
    # Create fused Reshape
    #--------------------------------------------------------------------------
    fused_out = gs.Variable(f'{q_reshape.name}_fused_out')
    fused_reshape = gs.Node(
        op='Reshape',
        name=f'{q_reshape.name}_fused',
        inputs=[current_input, gs.Constant(f'{q_reshape.name}_shape', np.array(fused_shape, dtype=np.int64))],
        outputs=[fused_out]
    )
    graph.nodes.append(fused_reshape)
    
    # Update pattern based on source type
    if pattern['source_type'] == 'common':
        # Fused Reshape outputs directly to each branch
        fused_reshape.outputs = reshape_outs
        pattern['source'] = fused_reshape
        logging.debug(f"Fused Reshapes (common): shape={fused_shape}")
        
    else:  # source_type == 'split'
        # Update existing Split
        pattern['source'].inputs[0] = fused_out
        pattern['source'].outputs = reshape_outs
        pattern['source'].attrs['axis'] = old_split_axis  # Same axis in new shape
        
        # Update split sizes
        if len(pattern['source'].inputs) > 1:
            if graph.opset >=13:
                pattern['source'].inputs[1].values = np.array(split_sizes, dtype=np.int64)
            else:
                pattern['source'].attrs['split'] = split_sizes
        else:
            if graph.opset >= 13:
                pattern['source'].inputs.append(gs.Constant(f'{pattern["source"].name}_sizes', np.array(split_sizes, dtype=np.int64)))
            else:
                pattern['source'].attrs['split'] = split_sizes
        
        logging.debug(f"Fused Reshapes (split): fused_shape={fused_shape}, split_axis={old_split_axis}, split_sizes={split_sizes}")
    
    #--------------------------------------------------------------------------
    # Clear old Reshapes
    #--------------------------------------------------------------------------
    for r in [q_reshape, k_reshape, v_reshape]:
        r.inputs.clear()
        r.outputs.clear()
    
    return True

def fuse_transpose(graph: gs.Graph, pattern: Dict, idx: int) -> bool:
    """
    Fuse Transpose at position idx.
    """
    if pattern['source_type']=='common':
        logging.warning("Not fused the Transposed because source is not Split")
        return False

    q_transpose = pattern['q_ops'][idx]
    k_transpose = pattern['k_ops'][idx]
    v_transpose = pattern['v_ops'][idx]

    # Check if all Add input shapes are the same
    transpose_inputs = [m.inputs[0] for m in [q_transpose, k_transpose, v_transpose]]
    if not all(inp.shape is not None and list(inp.shape) == list(transpose_inputs[0].shape) for inp in transpose_inputs):
        logging.debug(f"    Tranpose input shapes are different or undefined")
        return False
    
    # Extract permutations
    q_perm = list(q_transpose.attrs.get('perm', []))
    k_perm = list(k_transpose.attrs.get('perm', []))
    v_perm = list(v_transpose.attrs.get('perm', []))

    # Validate: Q and V must have same perm
    if not (q_perm and v_perm and q_perm == v_perm):
        logging.debug(f"Transpose Q/V perms don't match: Q={q_perm}, V={v_perm}")
        return False

    # Check if all 3 perms are identical or K is different
    all_same = (q_perm == k_perm)

    # Determine input for fused Transpose
    current_input = pattern['source'].inputs[0]

    #--------------------------------------------------------------------------
    # Create fused Transpose (using Q/V perm)
    #--------------------------------------------------------------------------
    fused_out = gs.Variable(f'{q_transpose.name}_fused_out')
    fused_transpose = gs.Node(
        op='Transpose',
        name=f'{q_transpose.name}_fused',
        inputs=[current_input],
        outputs=[fused_out],
        attrs={'perm': q_perm}
    )
    graph.nodes.append(fused_transpose)

    # Update Split outputs — bypass Q and V transposes always
    for i, output in enumerate(pattern['source'].outputs):
        if output == q_transpose.inputs[0]:
            pattern['source'].outputs[i] = q_transpose.outputs[0]
        elif output == v_transpose.inputs[0]:
            pattern['source'].outputs[i] = v_transpose.outputs[0]
        elif output == k_transpose.inputs[0]:
            if all_same:
                # K perm == Q perm: bypass K transpose too
                pattern['source'].outputs[i] = k_transpose.outputs[0]
            else:
                # K perm != Q perm: keep K transpose, change perm to swap last 2 dims
                pattern['source'].outputs[i] = k_transpose.inputs[0]
                k_transpose.inputs[0].shape = None

    pattern['source'].inputs[0] = fused_out

    # Update Split axis after transpose
    old_axis = pattern['source'].attrs.get('axis', -1)
    if old_axis < 0:
        old_axis = len(q_perm) + old_axis
    new_axis = q_perm.index(old_axis) if old_axis in q_perm else old_axis
    pattern['source'].attrs['axis'] = new_axis

    # Update the split sizes
    q_shape = q_transpose.inputs[0].shape
    new_split = [q_shape[old_axis]] * 3
    if graph.opset >= 13:
        pattern['source'].inputs[1].values = np.array(new_split).astype(np.int64)
    else:
        pattern['source'].attrs['split'] = new_split

    if all_same:
        # All 3 perms identical — remove all transposes
        for t in [q_transpose, k_transpose, v_transpose]:
            t.inputs.clear()
            t.outputs.clear()
        logging.debug(f"Fused Transpose: all perms identical {q_perm}, removed all 3")
    else:
        # Q/V removed, K kept with last-2-dims swap
        for t in [q_transpose, v_transpose]:
            t.inputs.clear()
            t.outputs.clear()
        new_k_perm = list(range(len(k_perm)))
        new_k_perm[-1], new_k_perm[-2] = new_k_perm[-2], new_k_perm[-1]
        k_transpose.attrs['perm'] = new_k_perm
        logging.debug(f"Fused Transpose: Q/V perm={q_perm} removed, K perm changed {k_perm} → {new_k_perm}")

    return True

def fuse_mul_or_div(graph: gs.Graph, pattern: Dict, idx: int) -> bool:
    """
    Fuse Mul/Div at position idx - ONLY if all scalars are identical.
    """
    OP= 'Mul' if pattern['k_ops'][idx].op == 'Mul' else 'Div' 
    q_op = pattern['q_ops'][idx]
    k_op = pattern['k_ops'][idx]
    v_op = pattern['v_ops'][idx]
    all_nodes = [q_op, k_op, v_op]

    # Check if all MUL/DIV input shapes are the same
    op_inputs = [m.inputs[0] for m in [q_op, k_op, v_op]]
    if not all(inp.shape is not None and list(inp.shape) == list(op_inputs[0].shape) for inp in op_inputs):
        logging.debug(f"    MUL/DIV input shapes are different or undefined")
        return False

    # Extract scalar, output, and input variable
    scalars = []
    old_outputs = []
    data_inputs = []
    for node in all_nodes:
        scalar = next((inp.values for inp in node.inputs if isinstance(inp, gs.Constant)),None)
        if scalar is None:
            return False

        scalars.append(scalar)
        old_outputs.append(node.outputs[0])
        data_in = next((inp for inp in node.inputs if isinstance(inp, gs.Variable)),None)
        if data_in is None:
            return False
        data_inputs.append(data_in)

    # All scalars must be identical
    if not (np.array_equal(scalars[0], scalars[1]) and np.array_equal(scalars[1], scalars[2])):
        return False

    fused_scalar = scalars[0]

    # Decide fused Mul input
    if pattern['source_type'] == 'common':
        if(type(pattern['source'])==gs.Variable):
            current_input = pattern['source']
        else:
            current_input = pattern['source'].outputs[0]
    else:
        current_input = pattern['source'].inputs[0]

    # ---------------------------------------------------------------------
    # Fused Mul/Div Node
    # ---------------------------------------------------------------------
    fused_output = gs.Variable(f'{q_op.name}_fused_out')
    fused_node = gs.Node(
        op=OP,
        name=f'{q_op.name}_fused_{OP}',
        inputs=[current_input,gs.Constant(f'{q_op.name}_scalar', fused_scalar)],
        outputs=[fused_output]
    )
    graph.nodes.append(fused_node)

    # CASE 1: COMMON
    if pattern['source_type'] == 'common':
        fused_node.outputs = old_outputs
        pattern['source'] = fused_node

        # Clear the input and out put of the Mul/Div
        for node in all_nodes:
            node.outputs.clear()
            node.inputs.clear()

        return True

    # CASE 2: SPLIT
    else:
        pattern['source'].inputs[0] = fused_output

        for node in all_nodes:
            out_var=node.outputs[0]
            data_in = next ((inp for inp in node.inputs if isinstance(inp, gs.Variable)), None)
            
            if data_in is None:
                return False
            
            producer = data_in.inputs[0]

            if len(data_in.outputs)==1 and data_in.outputs[0] is node:
                replaced = False
                for i, ov in enumerate(producer.outputs):
                    if ov is data_in:
                        producer.outputs[i] = out_var
                        replaced = True
                        break
                
                if not replaced:
                    return False
            else:
                for consumer in list(out_var.outputs):
                    for i ,inp in enumerate(consumer.inputs):
                        if inp is out_var:
                            consumer.inputs[i]= data_in
                
                out_var.outputs.clear()
            
            node.inputs.clear()
            node.outputs.clear()
        
        return True

def optimize_attention_pattern(graph: gs.Graph, pattern: Dict) -> bool:
    """Optimize attention pattern by iterating through operations."""
    q_ops = pattern['q_ops']
    k_ops = pattern['k_ops']
    v_ops = pattern['v_ops']
    
    # Track current position and input
    current_idx = 0
    
    # Iterate through operations
    max_ops = min(len(q_ops), len(k_ops), len(v_ops))
    
    for i in range(max_ops):
        # Check if all three branches have same operation at position i
        if not (q_ops[i].op == k_ops[i].op == v_ops[i].op):
            logging.debug(f"  Position {i}: ops don't match, stopping")
            continue
        
        op_type = q_ops[i].op
        
        # Call appropriate fusion function
        if op_type == 'MatMul':
            if not fuse_matmul(graph, pattern, i):
                break
        elif op_type == 'Add':
            if not fuse_add(graph, pattern, i):
                break
        elif op_type == 'Reshape':
            if not fuse_reshape(graph, pattern, i):
                break
        elif op_type == 'Transpose':
            if not fuse_transpose(graph, pattern, i):
                break
        elif op_type == 'Mul':
            if not fuse_mul_or_div(graph, pattern, i):
                break
        elif op_type == 'Div':
            if not fuse_mul_or_div(graph, pattern, i):
                break
        else:
            # Unsupported operation, stop merging
            logging.debug(f"  Position {i}: unsupported op {op_type}, stopping")
            break
        
        current_idx = i + 1
    return current_idx > 0


def tidl_optimize_hf_attention(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """Main entry point - handles both Split and non-Split patterns"""
    
    patterns = find_attention_patterns(graph)
    
    if not patterns:
        logging.debug("No attention patterns found")
        return
    
    optimized = 0
    
    for i, pattern in enumerate(patterns):
        try:
            if optimize_attention_pattern(graph, pattern):
                optimized +=1 
        
        except Exception as e:
            logging.warning(f"Optimization failed for pattern {i+1}: {e}")
            import traceback
            traceback.print_exc()
    
    graph.cleanup().toposort()
    
    logging.debug(f"Successfully optimized {optimized}/{len(patterns)} patterns")