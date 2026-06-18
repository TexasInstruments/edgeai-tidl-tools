# Copyright (c) {2023 - 2024} Texas Instruments Incorporated
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
Module containing Concat layer specific functions and optimizations.

This module provides optimizations for Concat operations in ONNX graphs to make them
more efficient for TIDL hardware. It includes functions to:
1. Convert unsupported-axis concatenation to channel-axis concatenation
2. Break down large Concat operations into smaller consecutive operations
"""


import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np



def tidl_convert_concat_unsupported_axis_to_channel(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Convert axis of a Concat layer from unsupported axis to supported axis,
    using Reshape operations to flatten and restore dimensions.

    TIDL hardware supports Concat only on specific axes (channel, height, width).
    This function transforms concatenations on unsupported axes (like batch dimension)
    by using Reshape operations to temporarily flatten the tensor, concatenate on a
    supported axis, and then reshape back to the original structure.

    The transformation uses Reshape (not Transpose) for efficiency:
    1. Reshape each input to flatten dimensions after the unsupported axis
    2. Concat on the unsupported axis (now valid for flattened 2D tensor)
    3. Reshape back to restore original dimensionality

    Example for axis 0 (batch) concat:
      Input: [1, 1, 64, 320, 320] x 6 tensors
      Reshape: [1, 6553600] x 6
      Concat on axis 0: [6, 6553600]
      Reshape back: [6, 1, 64, 320, 320]

    Supported axes for TIDL (for 4D tensors NCHW):
      - Axis 1: Channel (C)
      - Axis 2: Height (H)
      - Axis 3: Width (W)

    Unsupported axes that will be converted:
      - Axis 0: Batch (N)
      - Any other axis not in [1, 2, 3]

    Args:
        graph (gs.Graph): The ONNX GraphSurgeon graph to modify
        onnx_graph (onnx.GraphProto): The original ONNX graph (for reference)
    """
    logging.debug("Starting conversion of unsupported-axis Concat operations using Reshape approach")
    tensors = graph.tensors()

    # Find all Concat nodes in the graph
    concat_nodes = [node for node in graph.nodes if node.op == "Concat"]
    logging.debug(f"Found {len(concat_nodes)} Concat nodes to examine")

    for node in concat_nodes:
        # Get the concat axis (handle negative indices)
        concat_axis = node.attrs['axis']
        if concat_axis < 0:
            concat_axis = len(node.inputs[0].shape) + concat_axis

        # TIDL supports Concat on axes 1, 2, 3 (Channel, Height, Width) for 4D tensors
        # Any other axis (like 0 for batch) needs to be converted
        supported_axes = [1, 2, 3]  # C, H, W in NCHW format

        if concat_axis not in supported_axes:
            logging.debug(f"Found unsupported-axis Concat node: {node.name} with axis {concat_axis}")

            # Validate all inputs have consistent shapes
            valid_input = True
            first_shape = node.inputs[0].shape

            for inp in node.inputs:
                if inp.shape is None or len(inp.shape) < 2:
                    logging.warning(f"{inp.name} input to {node.name} has invalid shape {inp.shape}. Skipping this Concat.")
                    valid_input = False
                    break

            # Skip this node if any input doesn't have valid dimensions
            if not valid_input:
                logging.debug(f"Skipping node {node.name} due to invalid input dimensions")
                continue

            original_axis = concat_axis

            # Calculate flattened shape: keep dimensions up to concat_axis, flatten rest
            # For example: [1, 1, 64, 320, 320] with axis=0 → [1, 1*64*320*320] = [1, 6553600]
            original_shape = list(first_shape)
            num_dims = len(original_shape)

            # Calculate the product of all dimensions after the concat axis
            flat_size = 1
            for i in range(original_axis + 1, num_dims):
                flat_size *= original_shape[i]

            # New shape is [dim0, dim1, ..., dim_concat_axis, flat_size]
            # Then we keep up to and including concat_axis, and flatten everything after
            reshaped_dims = original_shape[:original_axis + 1] + [flat_size]

            logging.debug(f"Converting Concat axis for {node.name}: axis {original_axis}")
            logging.debug(f"Original shape: {original_shape}, Reshaped to: {reshaped_dims}")

            ## Modify inputs to the concat node - reshape all inputs to flatten dimensions
            logging.debug(f"Modifying {len(node.inputs)} inputs to Concat node {node.name}")
            for idx, inp in enumerate(node.inputs):
                ## Handle constant inputs differently - reshape the actual data
                if isinstance(inp, gs.Constant):
                    concat_const_tensor = np.array(tensors[inp.name].values, dtype=np.float32)

                    # Reshape const input
                    logging.debug(f"Reshaping constant input {inp.name}: {concat_const_tensor.shape} → {reshaped_dims}")
                    reshaped_const_tensor = concat_const_tensor.reshape(reshaped_dims)
                    node.inputs[idx] = gs.Constant(name=f'{inp.name}_reshaped_ax{original_axis}_flattened',
                                                    values=reshaped_const_tensor)

                ## Handle variable inputs by creating reshape nodes
                else:
                    # Create shape constant for Reshape operation
                    reshape_shape_name = f'{inp.name}_reshape_shape_flatten'
                    reshape_shape_constant = gs.Constant(name=reshape_shape_name,
                                                        values=np.array(reshaped_dims, dtype=np.int64))

                    # Create reshape layer
                    reshape_out = gs.Variable(name=f'{inp.name}_reshaped_ax{original_axis}_flattened',
                                             dtype=np.float32,
                                             shape=reshaped_dims)
                    reshape_node = gs.Node(name=f'reshape_{inp.name}_ax{original_axis}_flatten',
                                          op='Reshape',
                                          inputs=[inp, reshape_shape_constant],
                                          outputs=[reshape_out])
                    logging.debug(f"Adding reshape node {reshape_node.name}: {inp.shape} → {reshaped_dims}")
                    graph.nodes.append(reshape_node)

                    # feed new input to concat
                    node.inputs[idx] = reshape_out

            # Concat axis remains the same (it's valid for the flattened shape)
            # No need to change node.attrs['axis']

            ## Modify outputs from concat - need to reshape back to restore original dimensionality
            logging.debug(f"Modifying {len(node.outputs)} outputs from Concat node {node.name}")
            for idx, outp in enumerate(node.outputs):
                # Calculate the output shape after concat
                # The concat happens on original_axis, so sum all input sizes on that axis
                concat_flat_shape = reshaped_dims.copy()
                total_size_on_concat_axis = sum(inp.shape[original_axis] for inp in node.inputs)
                concat_flat_shape[original_axis] = total_size_on_concat_axis

                # Calculate the final output shape after reshaping back
                # Restore original dimensions, but with concatenated size on the concat_axis
                final_shape = original_shape.copy()
                final_shape[original_axis] = total_size_on_concat_axis

                logging.debug(f"Concat output shape (flattened): {concat_flat_shape}, target shape: {final_shape}")

                # Create shape constant for Reshape operation
                reshape_back_shape_name = f'{outp.name}_reshape_shape_restore'
                reshape_back_shape_constant = gs.Constant(name=reshape_back_shape_name,
                                                         values=np.array(final_shape, dtype=np.int64))

                # Create reshape layer to restore original dimensionality
                reshape_in = gs.Variable(name=f'{outp.name}_reshaped_ax{original_axis}_restored',
                                        dtype=np.float32,
                                        shape=concat_flat_shape)
                reshape_node = gs.Node(name=f'reshape_{outp.name}_ax{original_axis}_restore',
                                      op='Reshape',
                                      inputs=[reshape_in, reshape_back_shape_constant],
                                      outputs=[outp])
                logging.debug(f"Adding reshape node {reshape_node.name}: {concat_flat_shape} → {final_shape}")
                graph.nodes.append(reshape_node)

                # Update output variable shape
                outp.shape = final_shape

                # Replace original output with the input to our new reshape node
                # This completes the transformation chain:
                # 1. Original input tensors → Reshape nodes (flatten dimensions after concat_axis)
                # 2. Flattened inputs → Concat operation on original axis
                # 3. Concat output → Final Reshape (restore original dimensionality)
                # 4. Final output has same data arrangement as if concat on original axis was performed
                node.outputs[idx] = reshape_in

            logging.debug(f"Successfully converted Concat node {node.name} on axis {original_axis} using Reshape approach")


def tidl_convert_single_concat_to_consecutive_concats (graph: gs.Graph, onnx_graph: onnx.GraphProto, base:int=None):
    """
    Convert a concat operation with too many inputs into a sequence of smaller concat operations.
    
    TIDL has a limitation on the maximum number of inputs that can be processed by a single
    Concat operation. This function breaks down large Concat operations into multiple
    consecutive smaller Concat operations, each taking a limited number of inputs.
    
    The function handles two special cases:
    1. General case: A concat with many different inputs is broken down recursively
    2. Expansion case: A concat with many copies of the same input (used for dimension expansion)
       is converted to a more efficient implementation using powers of the base
    
    Args:
        graph (gs.Graph): The ONNX GraphSurgeon graph to modify
        onnx_graph (onnx.GraphProto): The original ONNX graph (for reference)
        base (int, optional): Maximum number of inputs allowed per Concat node. 
                             Defaults to 24 (TIDL_DEFAULT_MAX_NUM_INPUTS_FOR_CONCAT)
    
    Note:
        It is important to specify "skipped_optimizers=['fuse_consecutive_concats']" 
        when simplifying the graph, otherwise these optimizations might be undone.
    """
    count = 0
    base = int(base or 24) # TIDL_DEFAULT_MAX_NUM_INPUTS_FOR_CONCAT
    logging.debug(f"Starting conversion of large Concat operations using base {base}")
    
    # Helper function to recursively break down concat operations with too many inputs
    def break_concat_gt_base_to_2_concats(graph, concat_node, name=None):
        """
        Recursively break down a concat node with too many inputs into multiple
        concat nodes, each taking at most 'base' inputs.
        
        This function implements a recursive divide-and-conquer approach:
        1. Split inputs into two halves
        2. Create a new concat node for the first half
        3. Modify the original node to take the new concat's output plus second half
        4. Recursively apply this process if either half still has too many inputs
        
        Args:
            graph: The graph being modified
            concat_node: The concat node to process
            name: Base name for created nodes (defaults to original node's name)
        """
        nonlocal count
        inputs = list(concat_node.inputs)
        name = name or concat_node.name
        concat_axis = concat_node.attrs['axis']
        
        # Base case: if number of inputs is within limit, no need to split
        if len(inputs) <= base:
            # Recursion termination point - this concat node is already compliant with TIDL limits
            return
            
        logging.debug(f"Breaking down Concat node {name} with {len(inputs)} inputs (exceeds limit of {base})")
        # Split inputs in half - first half goes to first concat
        concat1_inputs = inputs[:len(inputs)//2]  # First half of inputs
        concat1_output = gs.Variable(f'{name}_{count}_out', inputs[0].dtype)
        
        # Create first concat node with first half of inputs
        concat1 = gs.Node('Concat', f'{name}_{count}', dict(axis=concat_axis), concat1_inputs, [concat1_output])
        count += 1
        graph.nodes.append(concat1)
        logging.debug(f"Created first-level Concat node {concat1.name} with {len(concat1_inputs)} inputs")
        # Calculate and set output shape if all input shapes are known
        if all(inp.shape is not None for inp in concat1_inputs):
            shape = list(concat1_inputs[0].shape)
            # Sum the sizes along the concatenation axis
            shape[concat_axis] = sum([inp.shape[concat_axis] for inp in concat1_inputs])
            concat1_output.shape = shape
            logging.debug(f"Set output shape for {concat1.name}: {shape}")
        
        # Second concat takes output of first concat plus second half of original inputs
        # This creates a hierarchical tree structure of concat operations:
        #
        # Before:  concat(A, B, C, D, E, F, G, H) -> result
        #
        # After:   concat1(A, B, C, D) -> temp
        #          concat2(temp, E, F, G, H) -> result
        #
        # If there are still too many inputs, the process repeats recursively
        concat2_inputs = [concat1_output] + inputs[len(inputs)//2:]
        
        # Modify the original node to be the second concat in the chain
        concat_node.inputs.clear()
        concat_node.inputs.extend(concat2_inputs)
        concat_node.name += f'_{count}'
        logging.debug(f"Modified original Concat node to {concat_node.name} with {len(concat2_inputs)} inputs")
        count += 1
        for concat in [concat1, concat_node]:
            break_concat_gt_base_to_2_concats(graph, concat, name)
    
    # Find all Concat nodes that need to be processed
    concat_nodes = [node for node in graph.nodes if node.op == "Concat"]
    logging.debug(f"Found {len(concat_nodes)} Concat nodes to examine for input count limits")
    
    for node in concat_nodes:
        num_inputs = len(node.inputs)
        concat_axis = node.attrs['axis']
        
        # Only process nodes with too many inputs
        if num_inputs > base:
            logging.debug(f"Processing Concat node {node.name} with {num_inputs} inputs (exceeds limit of {base})")
            count = 0
            # Special case: all inputs are the same tensor (dimension expansion pattern)
            if all(x is node.inputs[0] for x in node.inputs):
                logging.debug(f"Detected dimension expansion pattern in {node.name} - all {num_inputs} inputs are identical")
                input_node = node.inputs[0]
                
                # Calculate coefficients for base representation of num_inputs
                # This is analogous to converting a decimal number to a base-N representation
                # For example, if num_inputs=100 and base=24:
                # 100 = 4*24^0 + 4*24^1 (4 + 96) 
                # coefficients would be [4, 4]
                coefficients = []
                num = num_inputs
                while num > base:
                    coefficients.append(num % base)  # Remainder becomes a coefficient
                    num = int((num-coefficients[-1])/base)  # Integer division for next digit
                logging.debug(f"Decomposed {num_inputs} into coefficients: {coefficients} for base {base}")
                # Clear original inputs and build a more efficient representation
                node.inputs.clear()
                
                # Generate powers of base by repeated concatenation
                # This creates a logarithmic solution instead of a linear one:
                # Instead of creating num_inputs copies (O(n)), we create log_base(n) nodes
                # where each node represents base^i copies of the input
                #
                # exponents[0] = input            (base^0 = 1 copy)
                # exponents[1] = concat(in,...,in) with 'base' copies (base^1)
                # exponents[2] = concat(exp[1],...,exp[1]) with 'base' copies (base^2)
                # And so on...
                exponents = [input_node]  # Base^0 = original input
                
                while len(exponents) < len(coefficients):
                    inp = exponents[-1]
                    out = gs.Variable(f'{node.name}_{count}_out', inp.dtype)
                    
                    # Calculate output shape if input shape is known
                    if inp.shape is not None:
                        shape = list(inp.shape)
                        # Multiplying by base along concat axis (e.g., if base=24, this tensor is 24x larger)
                        shape[concat_axis] = inp.shape[concat_axis] * base
                        out.shape = shape
                    # Create a concat that repeats the input 'base' times
                    # This is an efficient way to exponentially increase tensor copies
                    # For example, if base=24:
                    # - First iteration creates a tensor equivalent to 24 copies
                    # - Second iteration creates a tensor equivalent to 24*24=576 copies
                    # - And so on, drastically reducing the number of nodes needed
                    concat = gs.Node('Concat', 
                                    f'{node.name}_{count}', 
                                    dict(axis=concat_axis), 
                                    [inp for _ in range(base)],  # Repeat input 'base' times
                                    [out])
                    graph.nodes.append(concat)
                    count += 1
                    exponents.append(out)
                    logging.debug(f"Created power concat {concat.name} that expands input by factor of {base}")
                # Rebuild input list using calculated powers
                # This creates the same effect as the original concat but with fewer nodes
                # Using the base-N representation, we reconstruct the exact same number of
                # tensor copies but using our pre-computed power tensors
                #
                # Example: For num_inputs=100, base=24, coefficients=[4,4]:
                # - Add 4 copies of exponents[0] (original tensor)
                # - Add 4 copies of exponents[1] (each equal to 24 original tensors)
                # Total: 4 + 4*24 = 100 effective tensor copies with only 8 actual inputs
                for coeff, exp in zip(coefficients, exponents):
                    if coeff > 0:  # Only add non-zero coefficients
                        node.inputs.extend([exp for _ in range(coeff)])
                        logging.debug(f"Added {coeff} copies of power tensor to final concat")
                
                logging.debug(f"Optimized dimension expansion in {node.name}: {num_inputs} identical inputs → efficient representation with {len(node.inputs)} inputs")
            # For regular concat operations with too many inputs, use the recursive breakdown method
            break_concat_gt_base_to_2_concats(graph, node)
            logging.debug(f"Completed processing of Concat node {node.name}")
