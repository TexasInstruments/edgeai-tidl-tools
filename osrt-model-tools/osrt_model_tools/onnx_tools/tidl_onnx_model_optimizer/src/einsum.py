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
Module containing Einsum layer specific functions and optimizations

The Einsum operator is a flexible operator that can represent various matrix operations
through Einstein summation notation. This module provides transformations to replace
Einsum operations with more basic operations like MatMul, Transpose, and Reshape
for better compatibility with TIDL.
"""
import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np

def tidl_replace_einsum_with_matmul_and_basic_ops(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Replaces Einsum operations with a simplified combination of Reshape, Transpose, 
    and MatMul operations to improve compatibility with TIDL.
    
    This function specifically targets Einsum operations and attempts to decompose them
    into simpler operations that can be more easily mapped to hardware accelerators.
    
    Args:
        graph: The ONNX GraphSurgeon graph to be modified
        onnx_graph: The original ONNX graph (for reference)
        
    Returns:
        None: The function modifies the graph in-place
    """
    logging.debug("Starting Einsum replacement with basic operations optimization")
    
    # Find all Einsum nodes in the graph
    einsum_nodes = [node for node in graph.nodes if node.op == "Einsum"]
    logging.debug(f"Found {len(einsum_nodes)} Einsum nodes in the graph")
    
    for einsum_node in einsum_nodes:
        logging.debug(f"Processing Einsum node: {einsum_node.name}")
        
        # Get the Einsum equation from attributes
        equation = einsum_node.attrs.get("equation", "")
        logging.debug(f"Einsum equation: {equation}")
        
        # Validate equation format
        assert isinstance(equation, str), "Einsum equation must be a string"
        
        # Parse the equation into left-hand side (inputs) and right-hand side (output)
        lhs, rhs = equation.split('->')
        
        # Skip if right-hand side is empty
        if len(rhs) == 0:
            logging.debug(f"Skipping Einsum node {einsum_node.name} with empty right-hand side")
            continue
            
        # Parse right-hand side dimensions
        if rhs == rhs.split()[0]:
            rhs = [s for s in rhs]  # Convert single string to character list
        else:
            rhs = rhs.split()  # Already space-separated
            
        # Parse left-hand side operands
        operands = lhs.split(',')
        
        # Store original inputs and outputs for possible restoration
        _inp1, _inp2 = einsum_node.inputs
        _out = einsum_node.outputs[0]
        logging.debug(f"Input shapes: {_inp1.shape}, {_inp2.shape}, Output shape: {_out.shape}")
        
        # Only handle binary Einsum operations (with 2 inputs)
        if len(operands) != 2:
            logging.debug(f"Skipping Einsum node {einsum_node.name} with {len(operands)} operands (only binary operations supported)")
            continue
            
        # Process dimensions from each operand
        dims = {}  # Dictionary to map dimension labels to their sizes
        for i, operand in enumerate(operands):
            # Parse operand dimensions
            if operand == operand.split()[0]:
                operand = [s for s in operand]  # Convert to character list
            else:
                operand = operand.split()
                
            # Store parsed operand back
            operands[i] = operand
            
            # Get input tensor for this operand
            inp = einsum_node.inputs[i]
            
            # Map each dimension label to its size from the input shape
            for j, d in enumerate(operand):
                if d not in dims:
                    dims[d] = None
                    if inp.shape:
                        dims[d] = inp.shape[j]
                        logging.debug(f"Dimension {d} has size {dims[d]}")
        
        # Identify multiplication dimensions (dimensions that appear in inputs but not in output)
        mul_dims = [d for d in dims if d not in rhs] 
        
        # Only handle cases with exactly one multiplication dimension
        if len(mul_dims) != 1:
            logging.debug(f"Skipping Einsum node {einsum_node.name}: found {len(mul_dims)} multiplication dimensions, need exactly 1")
            continue
        # TODO Remove this logic to add support for more than 1 mul_dims
            
        # Ensure the multiplication dimension appears in both operands
        if not all(all(d in operand for d in mul_dims)  for operand in operands):
            logging.debug(f"Skipping Einsum node {einsum_node.name}: multiplication dimension {mul_dims[0]} not in all operands")
            continue
            
        # Unpack operands for clarity
        a, b = operands
        logging.debug(f"Operand dimensions - a: {a}, b: {b}")
        
        # Categorize dimensions:
        # 1. Same dimensions: appear in both operands but are not multiplication dimensions
        same_dims = [d for d in dims if d in a and d in b and d not in mul_dims]
        # 2. Dimensions unique to first operand
        diff_dims_a = [d for d in dims if d in a and d not in b and d not in mul_dims]
        # 3. Dimensions unique to second operand
        diff_dims_b = [d for d in dims if d not in a and d in b and d not in mul_dims]
        
        logging.debug(f"Dimension classification - same: {same_dims}, unique to a: {diff_dims_a}, unique to b: {diff_dims_b}, multiplication: {mul_dims}")
        
        # Find axes positions of the same dimensions in each operand
        same_dim_axes_a = [i for i, d in enumerate(a) if d in same_dims]
        same_dim_axes_b = [i for i, d in enumerate(b) if d in same_dims]
        
        # Ensure common dimensions are at the beginning of each operand in the same order
        if same_dim_axes_a != list(range(len(same_dims))) or same_dim_axes_b != list(range(len(same_dims))):
            logging.debug(f"Skipping Einsum node {einsum_node.name}: common dimensions are not at the beginning of operands")
            continue
        
        # Get input tensors
        inp1, inp2 = einsum_node.inputs
        
        # Handle first operand (a) - ensure multiplication dimension is last
        axes = [i for i, d in enumerate(a) if d in diff_dims_a]
        mul_axis = [i for i, d in enumerate(a) if d == mul_dims[0]][0]
        
        # If the multiplication axis isn't at the right position, insert a transpose
        if mul_axis != (len(a)-1):
            logging.debug(f"Adding Transpose node for first operand to move multiplication dimension to the end")
            
            # Create permutation that puts same dims first, then unique dims, then mul dim last
            perm = list(range(len(same_dims))) + axes + [mul_axis] 
            
            # Calculate output shape if input shape is available
            shape = None
            if inp1.shape:
                shape = [inp1.shape[p] for p in perm]
                
            # Create transpose node
            transpose_out = gs.Variable(f'{einsum_node.name}_tr1_out', inp1.dtype, shape)
            transpose = gs.Node('Transpose', 
                               f'{einsum_node.name}_tr1', 
                               dict(perm=perm), 
                               [inp1], 
                               [transpose_out])
                               
            # Update einsum input and dimension order
            einsum_node.inputs[0] = transpose_out
            a = [a[p] for p in perm]
            graph.nodes.append(transpose)
            logging.debug(f"Added transpose with perm={perm}, new operand a dimensions: {a}")
            
            
        # Handle second operand (b) - ensure multiplication dimension is in the right position
        axes = [i for i, d in enumerate(b) if d in diff_dims_b]
        mul_axis = [i for i, d in enumerate(b) if d == mul_dims[0]][0]
        
        # For MatMul compatibility, multiplication dimension should be after same dimensions
        # but before unique dimensions in the second operand
        if mul_axis != (len(b)-1-len(diff_dims_b)):
            logging.debug(f"Adding Transpose node for second operand to position multiplication dimension correctly")
            
            # Create permutation: same dims first, then mul dim, then unique dims
            perm = list(range(len(same_dims))) + [mul_axis] + axes
            
            # Calculate output shape if input shape is available
            shape = None
            if inp2.shape:
                shape = [inp2.shape[p] for p in perm]
                
            # Create transpose node
            transpose_out = gs.Variable(f'{einsum_node.name}_tr2_out', inp2.dtype, shape)
            transpose = gs.Node('Transpose', 
                               f'{einsum_node.name}_tr2', 
                               dict(perm=perm), 
                               [inp2], 
                               [transpose_out])
                               
            # Update einsum input and dimension order
            einsum_node.inputs[1] = transpose_out
            b = [b[p] for p in perm]
            graph.nodes.append(transpose)
            logging.debug(f"Added transpose with perm={perm}, new operand b dimensions: {b}")
        
        # Get updated inputs and output
        inp1, inp2 = einsum_node.inputs
        axes = [i for i, d in enumerate(a) if d in diff_dims_a]
        axes = [i for i, d in enumerate(b) if d in diff_dims_b]
        out = einsum_node.outputs[0]
        
        # If second operand has multiple axes in diff_dims_b, we need to flatten them
        # This is needed for proper MatMul operation
        if len(axes) > 1 and out.shape:
            logging.debug(f"Multiple unique dimensions in second operand, creating Reshape to flatten them")
            
            # Calculate new shape: keep non-axes dimensions and flatten axes dimensions
            shape = [dims[d] for i,d in enumerate(b) if i not in axes] + [np.prod([dims[d] for i,d in enumerate(b) if i in axes]).tolist()]
            
            # Create a combined dimension name for the flattened dimensions
            s = ''
            for axis in axes:
                s += b[axis]
                
            # Update dimensions list
            b = [d for i,d in enumerate(b) if i not in axes] + [s]
            dims[s] = shape[-1]
            
            # Create reshape node
            reshape_out = gs.Variable(f'{einsum_node.name}_rshp2_out', inp2.dtype, shape)
            shape_const = gs.Constant(f"{einsum_node.name}_shape2", np.array(shape).astype(np.int64))
            reshape = gs.Node('Reshape', 
                             f'{einsum_node.name}_rshp2', 
                             dict(), 
                             [inp2, shape_const], 
                             [reshape_out])
                             
            # Update einsum input
            einsum_node.inputs[1] = reshape_out
            graph.nodes.append(reshape)
            logging.debug(f"Added reshape to shape {shape}, new operand b dimensions: {b}")
            
            # Update diff_dims_b as we've combined multiple dimensions
            diff_dims_b = [d for d in dims if d not in a and d in b and d not in mul_dims]
        
        
        # Determine the dimensions of the output based on the transformations we've applied
        new_rhs = same_dims + diff_dims_a + diff_dims_b
        logging.debug(f"New output dimensions before flattening: {new_rhs}")
        
        # Flatten multi-character dimensions
        temp = []
        for d in new_rhs:
            if len(d) == 1:
                temp.append(d)
            elif len(d) > 1:
                if d == d.split()[0]:
                    d = [s for s in d]
                else:
                    d = d.split()
                temp.extend(d)
                
        # Calculate output shape based on dimension sizes
        shape = [dims[s] for s in temp]
        shape = shape if all(s for s in shape) else None
        logging.debug(f"Calculated intermediate output shape: {shape}")
        
        # If the output dimensions don't match what was requested in the original equation,
        # add a transpose to get the right dimension order
        if temp != rhs:
            logging.debug(f"Output dimensions don't match requested dimensions, adding transpose")
            logging.debug(f"Current: {temp}, Required: {rhs}")
            
            # Create transpose node to reorder dimensions
            trans_in = gs.Variable(f'{einsum_node.name}_tr_out', out.dtype, shape)
            perm = [temp.index(d) for d in rhs]
            transpose = gs.Node('Transpose', 
                               f'{einsum_node.name}_tr', 
                               dict(perm=perm), 
                               [trans_in], 
                               [out])
                               
            # Update einsum output
            graph.nodes.append(transpose)
            out = trans_in
            einsum_node.outputs[0] = trans_in
            logging.debug(f"Added transpose with perm={perm} to get final output order")
        
        # If any of the output dimensions have multiple characters, we need to reshape
        if any(len(d)>1 for d in new_rhs) and out.shape:
            logging.debug(f"Output has multi-character dimensions, adding reshape")
            
            # Calculate shape based on the pre-flattened dimensions
            shape1 = [dims[s] for s in new_rhs]
            logging.debug(f"Reshaping to shape: {shape1}")
            
            # Create reshape node
            reshape_in = gs.Variable(f'{einsum_node.name}_rshp_out', out.dtype, shape1)
            shape_const = gs.Constant(f'{einsum_node.name}_shape', np.array(shape).astype(np.int64))
            reshape = gs.Node('Reshape', 
                             f'{einsum_node.name}_rshp', 
                             {}, 
                             [reshape_in, shape_const], 
                             [out])
                             
            # Update einsum output
            einsum_node.outputs[0] = reshape_in
            graph.nodes.append(reshape)
            logging.debug(f"Added reshape to final output shape")
        
        # Check if we can convert to MatMul based on the dimensions
        axes_a = [i for i, d in enumerate(a) if d in diff_dims_a]
        axes_b = [i for i, d in enumerate(b) if d in diff_dims_b]
        
        # If both operands have exactly one unique dimension, we can convert to MatMul
        if len(axes_a) == 1 and len(axes_b) == 1:
            logging.debug(f"Converting Einsum node to MatMul")
            einsum_node.name += 'MatMul'  # Update node name
            einsum_node.op = 'MatMul'     # Change operator to MatMul
            einsum_node.attrs.clear()     # Clear attributes as they're not needed for MatMul
            logging.debug(f"Successfully converted Einsum to MatMul: {einsum_node.name}")
        else:
            # If we can't convert to MatMul, restore original connections
            logging.debug(f"Cannot convert to MatMul, restoring original Einsum node connections")
            einsum_node.inputs[0] = _inp1
            einsum_node.inputs[1] = _inp2  # Fixed bug: was using index 2
            einsum_node.outputs[0] = _out
            
    # Log completion of optimization
    logging.debug("Completed Einsum replacement optimization")
