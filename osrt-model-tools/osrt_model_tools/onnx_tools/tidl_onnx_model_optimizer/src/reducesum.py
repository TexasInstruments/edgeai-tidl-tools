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
Module containing ReduceSum layer specific functions and optimizations
"""
import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np



def tidl_convert_reducesum_to_matmul (graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Converts ReduceSum operations to equivalent implementations using MatMul and other operations.

    Transforms ReduceSum nodes into combinations of Transpose, MatMul, Reshape, and/or Squeeze
    operations for better hardware acceleration. Handles:
    - Input tensors of any dimension (2D and higher)
    - Reduction across one or two axes (with positive or negative indices)
    - keepdims=0 and keepdims=1 configurations
    - Different ONNX opset versions
    - Axes specified as attributes, inputs, or defaults

    Each transformation is tailored to the specific input dimensions and reduction pattern.
    """

    reduce_sums = [node for node in graph.nodes if node.op == "ReduceSum"]
    
    for idx, node in enumerate(reduce_sums):
        try:
            input_shape = node.inputs[0].shape
            ndims = len(input_shape)
            
            # Get axes
            if 'axes' in node.attrs:
                axes = node.attrs['axes']
            elif len(node.inputs) > 1:
                axes = node.inputs[1].values
            else:
                axes = list(range(ndims))
            
            # Normalize axes
            axes = [ax if ax >= 0 else ndims + ax for ax in axes]
            axes = sorted(axes)
            
            keepdims = node.attrs.get('keepdims', 1)
            dtype = getattr(node.inputs[0], 'dtype', np.float32)
            output_var = node.outputs[0]
            original_output_name = output_var.name  # Preserve original name
            
            # Validate
            if ndims < 2 or not axes:
                logging.info(f"Skipping {node.name}: invalid dimensions")
                continue
            
            # Check for dynamic shapes
            if None in input_shape:
                logging.warning(f"Skipping {node.name}: dynamic shape")
                continue
            
            # ===== SINGLE AXIS =====
            if len(axes) == 1:
                axis = axes[0]
                reduce_dim = input_shape[axis]
                ones = gs.Constant(f"{node.name}_ones_{idx}", np.ones((reduce_dim, 1), dtype=dtype))
                
                # If last axis - direct MatMul
                if axis == ndims - 1:
                    matmul_out = gs.Variable(f"{node.name}_matmul_out_{idx}", dtype=dtype)
                    graph.nodes.append(gs.Node("MatMul", f"{node.name}_matmul_{idx}",
                                               inputs=[node.inputs[0], ones],
                                               outputs=[matmul_out]))
                    
                    if keepdims:
                        matmul_out.name = original_output_name  # Use original name
                        final_out = matmul_out
                    else:
                        axes_const = gs.Constant(f"{node.name}_axes_{idx}", np.array([-1], dtype=np.int64))
                        final_out = gs.Variable(original_output_name, dtype=dtype)  # Use original name
                        graph.nodes.append(gs.Node("Squeeze", f"{node.name}_squeeze_{idx}",
                                                   inputs=[matmul_out, axes_const],
                                                   outputs=[final_out]))
                
                # Other axis - use transpose
                else:
                    perm = list(range(ndims))
                    perm[axis], perm[-1] = perm[-1], perm[axis]
                    
                    trans1_out = gs.Variable(f"{node.name}_trans1_out_{idx}", dtype=dtype)
                    graph.nodes.append(gs.Node("Transpose", f"{node.name}_trans1_{idx}",
                                               attrs={"perm": perm},
                                               inputs=[node.inputs[0]],
                                               outputs=[trans1_out]))
                    
                    matmul_out = gs.Variable(f"{node.name}_matmul_out_{idx}", dtype=dtype)
                    graph.nodes.append(gs.Node("MatMul", f"{node.name}_matmul_{idx}",
                                               inputs=[trans1_out, ones],
                                               outputs=[matmul_out]))
                    
                    trans2_out = gs.Variable(f"{node.name}_trans2_out_{idx}", dtype=dtype)
                    graph.nodes.append(gs.Node("Transpose", f"{node.name}_trans2_{idx}",
                                               attrs={"perm": perm},
                                               inputs=[matmul_out],
                                               outputs=[trans2_out]))
                    
                    if keepdims:
                        trans2_out.name = original_output_name  # Use original name
                        final_out = trans2_out
                    else:
                        axes_const = gs.Constant(f"{node.name}_axes_{idx}", np.array([axis], dtype=np.int64))
                        final_out = gs.Variable(original_output_name, dtype=dtype)  # Use original name
                        graph.nodes.append(gs.Node("Squeeze", f"{node.name}_squeeze_{idx}",
                                                   inputs=[trans2_out, axes_const],
                                                   outputs=[final_out]))
            
            # ===== MULTIPLE AXES =====
            else:
                axes_set = set(axes)
                keep_axes = [i for i in range(ndims) if i not in axes_set]
                reduce_axes = list(axes)
                perm = keep_axes + reduce_axes
                
                # Step 1: Transpose
                trans1_out = gs.Variable(f"{node.name}_trans1_out_{idx}", dtype=dtype)
                graph.nodes.append(gs.Node("Transpose", f"{node.name}_trans1_{idx}",
                                           attrs={"perm": perm},
                                           inputs=[node.inputs[0]],
                                           outputs=[trans1_out]))
                
                # Step 2: Reshape
                transposed_shape = [input_shape[i] for i in perm]
                num_keep = len(keep_axes)
                batch_shape = tuple(transposed_shape[:num_keep]) if num_keep > 0 else ()
                reduce_size = int(np.prod([input_shape[i] for i in axes]))
                
                reshape1_shape = batch_shape + (reduce_size,)
                if len(reshape1_shape) == 1:
                    reshape1_shape = (1,) + reshape1_shape
                
                shape1_const = gs.Constant(f"{node.name}_shape1_{idx}", np.array(reshape1_shape, dtype=np.int64))
                reshape1_out = gs.Variable(f"{node.name}_reshape1_out_{idx}", dtype=dtype)
                graph.nodes.append(gs.Node("Reshape", f"{node.name}_reshape1_{idx}",
                                           inputs=[trans1_out, shape1_const],
                                           outputs=[reshape1_out]))
                
                # Step 3: MatMul
                ones = gs.Constant(f"{node.name}_ones_{idx}", np.ones((reduce_size, 1), dtype=dtype))
                matmul_out = gs.Variable(f"{node.name}_matmul_out_{idx}", dtype=dtype)
                graph.nodes.append(gs.Node("MatMul", f"{node.name}_matmul_{idx}",
                                           inputs=[reshape1_out, ones],
                                           outputs=[matmul_out]))
                
                # Step 4: Reshape after MatMul
                if keepdims:
                    reshape2_shape = batch_shape + (1,) * len(axes)
                else:
                    reshape2_shape = batch_shape if batch_shape else (1,)
                
                shape2_const = gs.Constant(f"{node.name}_shape2_{idx}", np.array(reshape2_shape, dtype=np.int64))
                reshape2_out = gs.Variable(f"{node.name}_reshape2_out_{idx}", dtype=dtype)
                graph.nodes.append(gs.Node("Reshape", f"{node.name}_reshape2_{idx}",
                                           inputs=[matmul_out, shape2_const],
                                           outputs=[reshape2_out]))
                
                # Step 5: Transpose back (if keepdims)
                if keepdims:
                    perm_back = [0] * ndims
                    for i, p in enumerate(perm):
                        perm_back[p] = i
                    
                    final_out = gs.Variable(original_output_name, dtype=dtype)  # Use original name
                    graph.nodes.append(gs.Node("Transpose", f"{node.name}_trans2_{idx}",
                                               attrs={"perm": perm_back},
                                               inputs=[reshape2_out],
                                               outputs=[final_out]))
                else:
                    reshape2_out.name = original_output_name  # Use original name
                    final_out = reshape2_out
            
            # Reconnect consumers
            for consumer in output_var.outputs:
                for i, inp in enumerate(consumer.inputs):
                    if inp is output_var:
                        consumer.inputs[i] = final_out
            
            # Update graph outputs if needed
            for i, out in enumerate(graph.outputs):
                if out is output_var:
                    graph.outputs[i] = final_out
            
            logging.info(f"Converted {node.name} to MatMul")
            
        except Exception as e:
            logging.warning(f"Failed to convert {node.name}: {e}")
            import traceback
            traceback.print_exc()