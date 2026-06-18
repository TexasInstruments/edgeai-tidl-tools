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

import logging
import onnx_graphsurgeon as gs
import numpy as np
import onnx

import logging
import onnx_graphsurgeon as gs
import numpy as np
import onnx

def tidl_convert_unsupported_argmax_to_supported(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    '''
    Converts unsupported ArgMax configurations to TIDL-supported format.
    
    TIDL Support Requirements:
    - keepdims = 1 (default) only
    - axis = -3 only (for 3D/4D tensors)
    - select_last_index = 0 (default) only
    
    Transformations:
    - Always converts keepdims=0 to keepdims=1 + Reshape (all dimensions)
    - Adds Transpose nodes to move axis to -3 position (3D/4D only)
    - Adds Gather+Sub nodes to handle select_last_index=1 (3D/4D only)
    '''
    
    # Get the opset version from the model
    opset_version=graph.opset
    
    # Determine which attributes are supported in this opset version
    supports_select_last_index = (opset_version >= 12)
    
    nodes = [node for node in graph.nodes if node.op == 'ArgMax']
    
    for node in nodes:
        keep_dims = node.attrs.get('keepdims', 1)
        axis = node.attrs.get('axis', 0)
        select_last_index = node.attrs.get('select_last_index', 0)
        
        inp = node.inputs[0]
        inp_shape = list(inp.shape)
        inp_dim = len(inp_shape)
        
        # Normalize axis to positive
        axis_pos = axis if axis >= 0 else inp_dim + axis
        
        # Determine target axis based on dimension support
        # For 3D/4D: use -3 (TIDL requirement)
        # For others: keep original axis
        if 3 <= inp_dim <= 4:
            target_axis_pos = inp_dim - 3
            can_transform_axis = True
        else:
            target_axis_pos = axis_pos
            can_transform_axis = False
            if inp_dim < 3:
                logging.warning(f"ArgMax '{node.name}': {inp_dim}D tensor, only fixing keepdims")
            else:
                logging.warning(f"ArgMax '{node.name}': {inp_dim}D tensor not fully supported, only fixing keepdims and axis")
        
        # Check if already in correct format
        need_axis_change = (axis_pos != target_axis_pos)
        need_keepdims_change = (keep_dims != 1)
        need_select_last_change = (select_last_index != 0)
        
        if not need_axis_change and not need_keepdims_change and not need_select_last_change:
            if not supports_select_last_index and 'select_last_index' in node.attrs:
                del node.attrs['select_last_index']
            logging.debug(f"ArgMax '{node.name}' already in correct format")
            continue
        
        logging.debug(f"Converting ArgMax '{node.name}': dim={inp_dim}, axis={axis}, keepdims={keep_dims}, select_last_index={select_last_index}")
        
        original_output = node.outputs[0]
        current_tensor = inp
        axis_size = inp_shape[axis_pos]
        
        #---------------------------------------------------------------------------
        # Gather: Reverse data for select_last_index=1 (3D/4D only)
        #---------------------------------------------------------------------------
        if need_select_last_change and can_transform_axis:
            reversed_indices = np.arange(axis_size - 1, -1, -1, dtype=np.int64)
            indices_const = gs.Constant(name=f"{node.name}_reverse_indices", values=reversed_indices)
            reversed_out = gs.Variable(f'{node.name}_reversed_out', dtype=inp.dtype, shape=inp_shape)
            gather_reverse = gs.Node(op="Gather", inputs=[current_tensor, indices_const], outputs=[reversed_out], attrs={'axis': axis_pos})
            graph.nodes.append(gather_reverse)
            logging.warning(f"Added Gather (reverse) along axis {axis_pos}")
            current_tensor = reversed_out
        
        #---------------------------------------------------------------------------
        # Transpose: Move axis to target position (works for any dimension)
        #---------------------------------------------------------------------------
        if need_axis_change:
            perm = list(range(inp_dim))
            perm[axis_pos], perm[target_axis_pos] = perm[target_axis_pos], perm[axis_pos]
            transposed_shape = [inp_shape[p] for p in perm]
            transpose1_out = gs.Variable(f'{node.name}_transpose1_out', dtype=inp.dtype, shape=transposed_shape)
            transpose1 = gs.Node(op="Transpose", inputs=[current_tensor], outputs=[transpose1_out], attrs={'perm': perm})
            graph.nodes.append(transpose1)
            logging.debug(f"Added Transpose before ArgMax: perm={perm}")
            current_tensor = transpose1_out
        
        #---------------------------------------------------------------------------
        # ArgMax: Update to supported configuration
        #---------------------------------------------------------------------------
        node.inputs[0] = current_tensor
        node.attrs['axis'] = target_axis_pos - inp_dim  # Convert to negative index
        node.attrs['keepdims'] = 1
        
        # Handle select_last_index based on opset and dimension
        if supports_select_last_index and can_transform_axis:
            node.attrs['select_last_index'] = 0
        elif 'select_last_index' in node.attrs:
            del node.attrs['select_last_index']
        
        # Calculate ArgMax output shape
        argmax_shape = list(current_tensor.shape)
        argmax_shape[target_axis_pos] = 1
        
        # Determine if we need post-processing
        need_post_processing = need_axis_change or need_keepdims_change or need_select_last_change
        
        if need_post_processing:
            argmax_out = gs.Variable(f'{node.name}_argmax_out', dtype=original_output.dtype, shape=argmax_shape)
            node.outputs[0] = argmax_out
            current_tensor = argmax_out
        else:
            continue
        
        #---------------------------------------------------------------------------
        # Transpose: Restore original axis order (works for any dimension)
        #---------------------------------------------------------------------------
        if need_axis_change:
            inv_perm = [0] * inp_dim
            for i, p in enumerate(perm):
                inv_perm[p] = i
            inv_transposed_shape = [argmax_shape[inv_perm[i]] for i in range(inp_dim)]
            
            transpose2_out = gs.Variable(f'{node.name}_transpose2_out', dtype=original_output.dtype, shape=inv_transposed_shape) if need_keepdims_change or need_select_last_change else original_output
            transpose2 = gs.Node(op="Transpose", inputs=[current_tensor], outputs=[transpose2_out], attrs={'perm': inv_perm})
            graph.nodes.append(transpose2)
            logging.debug(f"Added Transpose after ArgMax: perm={inv_perm}")
            current_tensor = transpose2_out
        
        #---------------------------------------------------------------------------
        # Sub: Adjust indices after reverse (3D/4D only)
        #---------------------------------------------------------------------------
        if need_select_last_change and can_transform_axis:
            size_minus_1 = gs.Constant(name=f"{node.name}_size_minus_1", values=np.array([axis_size - 1], dtype=np.int64))
            sub_out = gs.Variable(f'{node.name}_adjusted_indices', dtype=np.int64, shape=current_tensor.shape) if need_keepdims_change else original_output
            sub_node = gs.Node(op="Sub", inputs=[size_minus_1, current_tensor], outputs=[sub_out])
            graph.nodes.append(sub_node)
            logging.warning(f"Added Sub to adjust indices")
            current_tensor = sub_out
        
        #---------------------------------------------------------------------------
        # Reshape: Remove keepdim dimension (works for any dimension)
        #---------------------------------------------------------------------------
        if need_keepdims_change:
            final_shape = [dim for i, dim in enumerate(current_tensor.shape) if i != axis_pos]
            final_shape_const = gs.Constant(name=f"{node.name}_final_shape", values=np.array(final_shape, dtype=np.int64))
            reshape_node = gs.Node(op="Reshape", inputs=[current_tensor, final_shape_const], outputs=[original_output])
            graph.nodes.append(reshape_node)
            logging.debug(f"Added Reshape to remove keepdim")