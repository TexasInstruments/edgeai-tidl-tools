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
Module containing ReduceMean layer specific functions and optimizations
"""
import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np

def tidl_expand_multiaxes_reducemean_to_single_axis_reducemeans(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Converts multi-axis ReduceMean to single-axis ReduceMean operations.
    
    Strategy:
    1. Find contiguous groups of axes
    2. Process groups in reverse order (to avoid index shifting)
    3. Use keepdims=0 for intermediate operations
    4. Add final Reshape only if original keepdims=1
    
    Note: LayerNorm typically uses single-axis ReduceMean on last dimension,
    so it won't be affected by this multi-axis conversion.
    """
    
    reduce_means = [node for node in graph.nodes if node.op == "ReduceMean"]
    
    for idx, node in enumerate(reduce_means):
        try:
            input_shape = node.inputs[0].shape
            ndims = len(input_shape)
            
            axes_in_attrs = 'axes' in node.attrs
            
            # Get axes
            if 'axes' in node.attrs:
                axes = node.attrs['axes']
            elif len(node.inputs) > 1:
                axes = node.inputs[1].values
            else:
                axes = list(range(ndims))
                axes_in_attrs = True
            
            # Normalize axes
            axes = [ax if ax >= 0 else ndims + ax for ax in axes]
            axes = sorted(axes)
            
            # Skip single-axis (includes LayerNorm cases)
            if len(axes) <= 1:
                logging.debug(f"Skipping {node.name}: single axis (may be LayerNorm)")
                continue
            
            keepdims = node.attrs.get('keepdims', 1)
            dtype = getattr(node.inputs[0], 'dtype', np.float32)
            output_var = node.outputs[0]
            original_output_name = output_var.name
            node.outputs.clear()
            
            # Validate
            if ndims < 2 or not axes:
                logging.debug(f"Skipping {node.name}: invalid dimensions")
                continue
            
            if None in input_shape:
                logging.debug(f"Skipping {node.name}: dynamic shape")
                continue
            
            # Find contiguous groups
            groups = []
            current_group = [axes[0]]
            for i in range(1, len(axes)):
                if axes[i] == axes[i-1] + 1:
                    current_group.append(axes[i])
                else:
                    groups.append(current_group)
                    current_group = [axes[i]]
            groups.append(current_group)
            
            logging.debug(f"Converting {node.name}: axes={axes}, groups={groups}")
            
            # ===== OPTIMIZATION: Single Reshape to merge all groups =====
            
            # Build new shape with all groups merged
            merged_shape = [] # New Shape
            new_reduce_axes = [] # New ReduceMean axes
            new_pos = 0
            
            i = 0
            while i < ndims:
                # Check if this position starts a group
                in_group = False
                for group in groups:
                    if i == min(group):
                        # Merge this entire group
                        merge_size = 1
                        for ax in group:
                            merge_size *= input_shape[ax]
                        merged_shape.append(merge_size)
                        new_reduce_axes.append(new_pos)
                        new_pos += 1
                        i = max(group) + 1
                        in_group = True
                        break
                
                if not in_group:
                    # Check if in middle of a group (skip)
                    skip = False
                    for group in groups:
                        if i in group and i != min(group):
                            skip = True
                            break
                    
                    if not skip:
                        # Keep this dimension as-is
                        merged_shape.append(input_shape[i])
                        new_pos += 1
                    i += 1
            
            logging.debug(f"  Merged shape: {merged_shape}")
            logging.debug(f"  New reduce axes: {new_reduce_axes}")
            
            # Step 1: Single Reshape (merge all contiguous groups)
            shape1_const = gs.Constant(f"{node.name}_shape_merged", 
                                      np.array(merged_shape, dtype=np.int64))
            reshape1_out = gs.Variable(f"{node.name}_reshape_merged", dtype=dtype)
            graph.nodes.append(gs.Node("Reshape", f"{node.name}_reshape1",
                                       inputs=[node.inputs[0], shape1_const],
                                       outputs=[reshape1_out]))
            
            # Step 2: ReduceMean on each merged dimension (reverse order)
            current_input = reshape1_out
            
            for reduce_idx, axis in enumerate(reversed(new_reduce_axes)):
                temp_out = gs.Variable(f"{node.name}_reduce_{reduce_idx}", dtype=dtype)
                
                if axes_in_attrs:
                    graph.nodes.append(gs.Node("ReduceMean", f"{node.name}_rm_{reduce_idx}",
                                               attrs={"keepdims": 0, "axes": [axis]},
                                               inputs=[current_input],
                                               outputs=[temp_out]))
                else:
                    axes_const = gs.Constant(f"{node.name}_axes_{reduce_idx}", 
                                            np.array([axis], dtype=np.int64))
                    graph.nodes.append(gs.Node("ReduceMean", f"{node.name}_rm_{reduce_idx}",
                                               attrs={"keepdims": 0},
                                               inputs=[current_input, axes_const],
                                               outputs=[temp_out]))
                
                current_input = temp_out
            
            # Step 3: Final Reshape (only if keepdims=1)
            if keepdims == 1:
                final_shape = list(input_shape)
                for ax in axes:
                    final_shape[ax] = 1
                
                shape_final_const = gs.Constant(f"{node.name}_shape_final", 
                                               np.array(final_shape, dtype=np.int64))
                final_out = gs.Variable(original_output_name, dtype=dtype)
                graph.nodes.append(gs.Node("Reshape", f"{node.name}_reshape_final",
                                           inputs=[current_input, shape_final_const],
                                           outputs=[final_out]))
            else:
                current_input.name = original_output_name
                final_out = current_input
            
            # Reconnect consumers
            for consumer in output_var.outputs:
                for i, inp in enumerate(consumer.inputs):
                    if inp is output_var:
                        consumer.inputs[i] = final_out
            
            # Update graph outputs
            for i, out in enumerate(graph.outputs):
                if out is output_var:
                    graph.outputs[i] = final_out
            
            logging.debug(f"Converted {node.name}: {len(groups)} group(s), {len(new_reduce_axes)} ReduceMean ops")
            
        except Exception as e:
            logging.info(f"Failed to convert {node.name}: {e}")
            import traceback
            traceback.print_exc()
