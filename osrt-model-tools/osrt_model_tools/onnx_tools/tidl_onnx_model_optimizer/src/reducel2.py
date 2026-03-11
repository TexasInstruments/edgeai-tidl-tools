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

def tidl_convert_reducel2_to_mul_reducesum_sqrt(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    '''
    Convert ReduceL2 Operartor to mul(x,x) -> ReduceSum -> sqrt
    
    ReduceL2 is replaced with a sequence of:
    1. Mul(input, input) - Square the input
    2. ReduceSum - Sum the squared values along the same axes as the original ReduceL2
    3. Sqrt - Take the square root of the sum
    '''
    # Get the opset version directly from the graph
    opset_version = graph.opset
    logging.debug(f"ONNX opset version: {opset_version}")
    
    # Identify and process ReduceL2 nodes in the graph
    reducel2_nodes = [node for node in graph.nodes if node.op == 'ReduceL2']
    logging.info(f"Found {len(reducel2_nodes)} ReduceL2 nodes to convert")
    
    for node in reducel2_nodes:
        if len(node.inputs) == 0 or node.inputs[0] is None:
            logging.warning(f"ReduceL2 node {node.name} has no inputs, skipping")
            continue
        
        # Get the input tensor
        input_tensor = node.inputs[0]
        
        # Get the ReduceL2 attributes and inputs
        axes = None
        axes_tensor = None
        keepdims = 1  # Default value as per ONNX spec
        
        # Check if axes is provided as a second input (opset >= 13)
        if len(node.inputs) > 1 and node.inputs[1] is not None:
            axes_tensor = node.inputs[1]
            if isinstance(axes_tensor, gs.Constant):
                axes = axes_tensor.values
                axes_tensor = None  # We'll create a new one based on opset version
        elif 'axes' in node.attrs:
            axes = node.attrs['axes']
        
        if 'keepdims' in node.attrs:
            keepdims = node.attrs['keepdims']
        
        # Create a unique name for the node or use the existing one
        node_name = node.name if node.name else f"reducel2_{id(node)}"
        
        # 1. Create Mul node to square the input (x*x)
        mul_output = gs.Variable(f"{node_name}_mul_output", dtype=input_tensor.dtype)
        mul_node = gs.Node(op="Mul", 
                           name=f"{node_name}_mul",
                           inputs=[input_tensor, input_tensor],
                           outputs=[mul_output])
        graph.nodes.append(mul_node)
        
        # 2. Create ReduceSum node based on opset version and axes source
        reducesum_output = gs.Variable(f"{node_name}_reducesum_output", dtype=input_tensor.dtype)
        reducesum_inputs = [mul_output]
        reducesum_attrs = {"keepdims": keepdims}
        
        # Handle axes based on opset version and source
        if opset_version >= 13:
            # For opset >= 13, axes should be passed as an input tensor if available
            if axes is not None:
                # Create a constant tensor for axes
                axes_constant = gs.Constant(f"{node_name}_axes", np.array(axes, dtype=np.int64))
                reducesum_inputs.append(axes_constant)
            elif axes_tensor is not None:
                # Use the original variable tensor
                reducesum_inputs.append(axes_tensor)
        else:
            # For opset < 13, axes should be in attributes if available
            if axes is not None:
                reducesum_attrs["axes"] = axes
            elif axes_tensor is not None and isinstance(axes_tensor, gs.Constant):
                # Extract values from tensor and put in attributes
                reducesum_attrs["axes"] = axes_tensor.values
            elif axes_tensor is not None:
                # For variable axes tensor, pass it as an input
                reducesum_inputs.append(axes_tensor)
        
        # Create the ReduceSum node
        reducesum_node = gs.Node(op="ReduceSum",
                                name=f"{node_name}_reducesum",
                                attrs=reducesum_attrs,
                                inputs=reducesum_inputs,
                                outputs=[reducesum_output])
        graph.nodes.append(reducesum_node)
        
        # 3. Create Sqrt node
        sqrt_node = gs.Node(op="Sqrt",
                           name=f"{node_name}_sqrt",
                           inputs=[reducesum_output],
                           outputs=node.outputs)
        graph.nodes.append(sqrt_node)
        
        # 4. Disconnect the original ReduceL2 node
        node.inputs.clear()
        node.outputs.clear()
        
