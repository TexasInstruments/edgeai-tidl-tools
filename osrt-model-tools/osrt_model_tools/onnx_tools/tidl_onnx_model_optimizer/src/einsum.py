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
"""
import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np

def tidl_replace_einsum_with_basic_ops(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Replaces Einsum operations with equation 'bnc,bchw->bnhw' with a simplified combination of 
    Reshape, Transpose, and MatMul operations.
    """
    logging.debug("Starting Einsum replacement with basic operations optimization")
    
    # Find all Einsum nodes
    einsum_nodes = [node for node in graph.nodes if node.op == "Einsum"]
    logging.debug(f"Found {len(einsum_nodes)} Einsum nodes in the graph")
    
    for einsum_node in einsum_nodes:
        logging.debug(f"Processing Einsum node: {einsum_node.name}")
        # Check if this is the specific equation we want to replace
        equation = einsum_node.attrs.get("equation", "")
        logging.debug(f"Einsum equation: {equation}")
        if equation != "bnc,bchw->bnhw":
            logging.debug(f"Skipping Einsum {einsum_node.name}: equation '{equation}' does not match target 'bnc,bchw->bnhw'")
            continue
            
        try:
            logging.debug(f"Processing target equation 'bnc,bchw->bnhw' for node {einsum_node.name}")
            # Get the input tensors
            input1 = einsum_node.inputs[0]  # bnc
            input2 = einsum_node.inputs[1]  # bchw
            logging.debug(f"Input tensors: {input1.name} (bnc), {input2.name} (bchw)")
            
            # Get the output tensor
            output = einsum_node.outputs[0]  # bnhw
            logging.debug(f"Output tensor: {output.name} (bnhw)")
            
            # Get shapes if available
            if not all(hasattr(t, 'shape') and t.shape is not None for t in [input1, input2]):
                logging.debug(f"Skipping Einsum {einsum_node.name}: Unable to determine input shapes")
                print(f"Skipping Einsum {einsum_node.name}: Unable to determine input shapes")
                continue
                
            b, n, c = input1.shape
            b2, c2, h, w = input2.shape
            logging.debug(f"Input1 shape (bnc): [{b}, {n}, {c}]")
            logging.debug(f"Input2 shape (bchw): [{b2}, {c2}, {h}, {w}]")
            
            # Verify that batch size and channel dimensions match
            if b != b2 or c != c2:
                logging.debug(f"Skipping Einsum {einsum_node.name}: Dimension mismatch - b:{b}!={b2} or c:{c}!={c2}")
                print(f"Skipping Einsum {einsum_node.name}: Dimension mismatch")
                continue
            
            logging.debug("Creating replacement nodes for Einsum operation")
            # Step 1: Reshape input2 from [b,c,h,w] to [b,c,h*w]
            reshape2_output = gs.Variable(name=f"{einsum_node.name}_reshape2_output", 
                                         dtype=input2.dtype,
                                         shape=[b, c, h*w])
            
            reshape2_shape = gs.Constant(name=f"{einsum_node.name}_reshape2_shape", 
                                        values=np.array([b, c, h*w], dtype=np.int64))
            
            reshape2_node = gs.Node(
                op="Reshape",
                name=f"{einsum_node.name}_reshape2",
                inputs=[input2, reshape2_shape],
                outputs=[reshape2_output]
            )
            logging.debug(f"Created Reshape node: {reshape2_node.name} - shape [{b}, {c}, {h*w}]")
            
            # Step 3: MatMul [b,n,c] x [b,c,h*w] -> [b,n,h*w]
            matmul_output = gs.Variable(name=f"{einsum_node.name}_matmul_output", 
                                       dtype=output.dtype,
                                       shape=[b, n, h*w])
            
            matmul_node = gs.Node(
                op="MatMul",
                name=f"{einsum_node.name}_matmul",
                inputs=[input1, reshape2_output],
                outputs=[matmul_output]
            )
            logging.debug(f"Created MatMul node: {matmul_node.name} - [{b}, {n}, {c}] x [{b}, {c}, {h*w}] -> [{b}, {n}, {h*w}]")
            
            # Step 4: Final Reshape from [b,n,h*w] to [b,n,h,w]
            final_shape = gs.Constant(name=f"{einsum_node.name}_final_shape", 
                                     values=np.array([b, n, h, w], dtype=np.int64))
            
            final_reshape_node = gs.Node(
                op="Reshape",
                name=f"{einsum_node.name}_final_reshape",
                inputs=[matmul_output, final_shape],
                outputs=[output]  # Reuse the original output
            )
            logging.debug(f"Created final Reshape node: {final_reshape_node.name} - shape [{b}, {n}, {h}, {w}]")
            
            # Add all the new nodes to the graph
            graph.nodes.extend([
                reshape2_node,
                matmul_node, final_reshape_node
            ])
            logging.debug("Added new nodes to graph")
            
            # Disconnect the Einsum node
            einsum_node.outputs.clear()
            logging.debug(f"Successfully replaced Einsum node: {einsum_node.name}")
            print(f"Replaced Einsum node: {einsum_node.name}")
            
        except Exception as e:
            logging.debug(f"Exception occurred while replacing Einsum node {einsum_node.name}: {str(e)}")
            print(f"Error replacing Einsum node {einsum_node.name}: {str(e)}")
    
    logging.debug("Completed Einsum replacement with basic operations optimization")
