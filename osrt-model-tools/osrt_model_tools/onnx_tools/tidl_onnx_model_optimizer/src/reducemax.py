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
Module containing ReduceMax layer specific functions and optimizations
"""
import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np

def tidl_convert_reducemax_for_height_axis(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Convert general ReduceMax operations to TIDL-compatible format:
    - Only supports reduction along height axis (rank-2 position)
    - Only keepdims=1 is supported
    
    Optimization strategy:
    1. Groups continuous axes together
    2. Processes groups in reverse order (high to low indices)
    3. Skips reshape if axis already at height position and shape ≤4D
    4. If keepdim=0 then add squeeze to remove dimension
    """
    
    # Find all ReduceMax nodes in graph
    reducemax_nodes = [node for node in graph.nodes if node.op == "ReduceMax"]
    if len(reducemax_nodes) == 0:
        return
    
    logging.debug(f"Found {len(reducemax_nodes)} ReduceMax node(s) to process")
    processed_count = 0
    
    for node in reducemax_nodes:
        try:
            logging.debug(f"Processing ReduceMax node: {node.name}")
            
            # Get input shape and dtype information
            input_tensor = node.inputs[0]
            dtype = input_tensor.dtype
            shape = input_tensor.shape
            
            if shape is None:
                logging.warning(f"Shape is None for node {node.name}, skipping")
                continue
            
            rank = len(shape)
            logging.debug(f"Input shape: {shape}, dtype: {dtype}, rank: {rank}")
            
            # Extract axes parameter
            axes = None
            if len(node.inputs) > 1 and isinstance(node.inputs[1], gs.Constant):
                axes = node.inputs[1].values
                if isinstance(axes, np.ndarray):
                    axes = axes.tolist()
                if not isinstance(axes, list):
                    axes = [axes]
                logging.debug(f"Found axes in input: {axes}")
            elif "axes" in node.attrs:
                axes = node.attrs["axes"]
                if not isinstance(axes, list):
                    axes = [axes]
                logging.debug(f"Found axes in attributes: {axes}")
            else:
                axes = list(range(rank))
                logging.debug(f"Using default axes (all dimensions): {axes}")
            
            # Extract keepdims parameter (default = 1)
            keepdims = node.attrs.get("keepdims", 1)
            logging.debug(f"keepdims: {keepdims}")
            
            # Normalize negative axes to positive indices
            axes = [ax if ax >= 0 else rank + ax for ax in axes]
            logging.debug(f"Normalized axes: {axes}")
            
            # Group continuous axes together
            axes_sorted = sorted(axes)
            axis_groups = []
            current_group = [axes_sorted[0]]
            
            for i in range(1, len(axes_sorted)):
                if axes_sorted[i] == axes_sorted[i-1] + 1:
                    current_group.append(axes_sorted[i])
                else:
                    axis_groups.append(current_group)
                    current_group = [axes_sorted[i]]
            axis_groups.append(current_group)
            
            logging.debug(f"Grouped continuous axes: {axis_groups}")
            
            # Save original input and output names
            original_input_name = input_tensor.name
            original_output_names = [out.name for out in node.outputs]
            
            # Process each axis group in REVERSE order
            current_tensor = input_tensor
            current_shape = list(shape)
            final_output_tensor = None  # Will be set to the last operation's output
            
            for group_idx, axis_group in enumerate(reversed(axis_groups)):
                is_last_group = (group_idx == len(axis_groups) - 1)
                
                logging.debug(f"Processing axis group {axis_group}")
                
                current_rank = len(current_shape)
                height_axis = current_rank - 2
                
                # Determine if reshape is needed
                needs_reshape = (
                    len(axis_group) > 1 or
                    axis_group[0] != height_axis or
                    current_rank > 4
                )
                
                logging.debug(f"Current shape: {current_shape}, height axis: {height_axis}, needs_reshape: {needs_reshape}")
                
                if needs_reshape:
                    #-------------------------------------------------------
                    # Reshape Node 1
                    #-------------------------------------------------------

                    # Calculate reshape dimensions
                    min_axis = min(axis_group)
                    max_axis = max(axis_group)
                    
                    before_dims = list(range(min_axis))
                    middle_dims = list(range(min_axis, max_axis + 1))
                    after_dims = list(range(max_axis + 1, current_rank))
                    
                    before_size = int(np.prod([current_shape[i] for i in before_dims])) if before_dims else 1
                    middle_size = int(np.prod([current_shape[i] for i in middle_dims]))
                    after_size = int(np.prod([current_shape[i] for i in after_dims])) if after_dims else 1
                    
                    # Build reshaped tensor
                    # Skip singleton dimensions (size=1) to minimize rank
                    reshaped = []
                    has_before = before_size > 1
                    has_after = after_size > 1

                    if has_before:
                        reshaped.append(before_size)
                    reshaped.append(middle_size)
                    if has_after:
                        reshaped.append(after_size)

                    # Calculate where middle dimension is positioned
                    middle_position = 1 if has_before else 0

                    # Ensure middle is at height position (rank-2)
                    target_rank = middle_position + 2

                    # Pad to reach target rank (padding at end)
                    while len(reshaped) < target_rank:
                        reshaped.append(1)

                    reshaped_rank = len(reshaped)
                    height_axis_reshaped = reshaped_rank - 2

                    logging.debug(f"Reshaped dimensions: {reshaped}, middle at position: {middle_position}, height at position: {height_axis_reshaped}")
                           
                    reshape_pre_output = gs.Variable(
                        name=f"{node.name}_reshape_pre_out.{processed_count}.{group_idx}",
                        dtype=dtype
                    )
                    
                    reshape_pre_shape_const = gs.Constant(
                        f"{node.name}_reshape_pre_shape.{processed_count}.{group_idx}",
                        values=np.array(reshaped, dtype=np.int64)
                    )
                    
                    reshape_pre_node = gs.Node(
                        op="Reshape",
                        name=f"{node.name}_reshape_pre.{processed_count}.{group_idx}",
                        inputs=[current_tensor, reshape_pre_shape_const],
                        outputs=[reshape_pre_output]
                    )
                    
                    graph.nodes.append(reshape_pre_node)
                    logging.debug(f"Added Reshape node (pre): {reshape_pre_node.name}")
                    
                    current_tensor = reshape_pre_output
                    reduce_axis = height_axis_reshaped
                else:
                    logging.debug(f"Skipping reshape - axis already at height and shape ≤4D")
                    reduce_axis = height_axis
                
                #--------------------------------------------------------------------
                # ReduceMax Node
                #-----------------------------------------------------------------
                reducemax_output = gs.Variable(
                    name=f"{node.name}_reduce_out.{processed_count}.{group_idx}",
                    dtype=dtype
                )
                
                if len(node.inputs) > 1:
                    axes_const = gs.Constant(
                        f"{node.name}_axes.{processed_count}.{group_idx}",
                        values=np.array([reduce_axis], dtype=np.int64)
                    )
                    reducemax_node = gs.Node(
                        op="ReduceMax",
                        name=f"{node.name}_reduce.{processed_count}.{group_idx}",
                        inputs=[current_tensor, axes_const],
                        outputs=[reducemax_output],
                        attrs={"keepdims": 1}
                    )
                else:
                    reducemax_node = gs.Node(
                        op="ReduceMax",
                        name=f"{node.name}_reduce.{processed_count}.{group_idx}",
                        inputs=[current_tensor],
                        outputs=[reducemax_output],
                        attrs={"axes": [reduce_axis], "keepdims": 1}
                    )
                
                graph.nodes.append(reducemax_node)
                logging.debug(f"Added ReduceMax node: {reducemax_node.name}")
                
                current_tensor = reducemax_output
                
                if needs_reshape:
                    #-----------------------------------------------------
                    # Reshape Node 2
                    #-----------------------------------------------------

                    # Calculate output shape (restore original structure)
                    output_shape = []
                    axis_group_set = set(axis_group)
                    for i in range(len(current_shape)):
                        if i in axis_group_set:
                            output_shape.append(1)
                        else:
                            output_shape.append(current_shape[i])
                    
                    logging.debug(f"Restoring shape to: {output_shape}")
                    
                    reshape_post_output = gs.Variable(
                        name=f"{node.name}_reshape_post_out.{processed_count}.{group_idx}",
                        dtype=dtype
                    )
                    
                    reshape_post_shape_const = gs.Constant(
                        f"{node.name}_reshape_post_shape.{processed_count}.{group_idx}",
                        values=np.array(output_shape, dtype=np.int64)
                    )
                    
                    reshape_post_node = gs.Node(
                        op="Reshape",
                        name=f"{node.name}_reshape_post.{processed_count}.{group_idx}",
                        inputs=[current_tensor, reshape_post_shape_const],
                        outputs=[reshape_post_output]
                    )
                    
                    graph.nodes.append(reshape_post_node)
                    logging.debug(f"Added Reshape node (post): {reshape_post_node.name}")
                    
                    current_tensor = reshape_post_output
                    current_shape = output_shape
                else:
                    # Update shape directly (no reshape needed)
                    current_shape[axis_group[0]] = 1
                    logging.debug(f"Updated shape to: {current_shape}")
                
                # Track the final output tensor
                if is_last_group and keepdims == 1:
                    final_output_tensor = current_tensor
            
            # Add squeeze if keepdims=0
            if keepdims == 0:
                logging.debug(f"keepdims=0 detected, adding Squeeze")
                
                #------------------------------------------------------------------
                # Squeeze Node
                #----------------------------------------------------------------
                squeeze_output = gs.Variable(
                    name=f"{node.name}_squeeze_out.{processed_count}",
                    dtype=dtype
                )
                
                if graph.opset < 13:
                    squeeze_node = gs.Node(
                        op="Squeeze",
                        name=f"{node.name}_squeeze.{processed_count}",
                        inputs=[current_tensor],
                        outputs=[squeeze_output],
                        attrs={"axes": axes}
                    )
                else:
                    squeeze_axes_const = gs.Constant(
                        f"{node.name}_squeeze_axes.{processed_count}",
                        values=np.array(axes, dtype=np.int64)
                    )
                    squeeze_node = gs.Node(
                        op="Squeeze",
                        name=f"{node.name}_squeeze.{processed_count}",
                        inputs=[current_tensor, squeeze_axes_const],
                        outputs=[squeeze_output]
                    )
                
                graph.nodes.append(squeeze_node)
                logging.debug(f"Added Squeeze node: {squeeze_node.name}")
                final_output_tensor = squeeze_output
            else:
                final_output_tensor = current_tensor
            
            # Preserve original output names by renaming final output tensor
            if len(node.outputs) > 0:
                original_output = node.outputs[0]
                final_output_tensor.name = original_output.name
                logging.debug(f"Preserved output name: {final_output_tensor.name}")
            
            # Replace original node's outputs
            original_outputs = node.outputs.copy()
            
            for original_output in original_outputs:
                # Update all consumers
                for consumer_node in original_output.outputs:
                    for i, inp in enumerate(consumer_node.inputs):
                        if inp == original_output:
                            consumer_node.inputs[i] = final_output_tensor
                
                # Update graph outputs
                if original_output in graph.outputs:
                    output_idx = graph.outputs.index(original_output)
                    graph.outputs[output_idx] = final_output_tensor
                    logging.debug(f"Updated graph output to use tensor: {final_output_tensor.name}")
            
            # Clear original node outputs
            node.outputs.clear()
            logging.debug(f"Successfully removed original ReduceMax node: {node.name}")
            
            processed_count += 1
            
        except Exception as e:
            logging.debug(f"Error processing node {node.name}: {str(e)}")
            import traceback
            logging.debug(traceback.format_exc())
            continue
    
    # Cleanup graph
    graph.cleanup().toposort()
    logging.debug(f"Successfully processed {processed_count} ReduceMax node(s)")