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
Module containing Gather layer specific functions and optimizations
"""


import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np
from .common import has_unk_axis



def tidl_convert_gather_scalar_to_1d(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    When Gather has single scalar index = t, convert to:
    Gather with 1D index [t] + Reshape to remove extra dimension
    """
    nodes = graph.nodes
    tensors = graph.tensors()

    gather_nodes = [node for node in graph.nodes if node.op == "Gather"]

    for node in gather_nodes:

        if not isinstance(node.inputs[1], gs.Constant):
            continue

        if has_unk_axis(node.inputs[0]):
            continue

        inp, idx = node.inputs[0], node.inputs[1]

        # Skip if input has less than 2 dimensions
        if len(inp.shape) < 2:
            logging.info(f"The Gather node {node.name} is currently not supported for conversion!")
            continue

        # Skip if index not in original tensors (likely already converted)
        if idx.name not in tensors:
            logging.debug(f"Skipping {node.name} - index {idx.name} not in original tensors")
            continue
        
        # Try to get the index values
        try:
            gather_indices = np.array(tensors[idx.name].values, dtype=np.int64)
        except (KeyError, AttributeError) as e:
            logging.debug(f"Skipping {node.name} - cannot access index values: {e}")
            continue
        
        # Check if it's a scalar (0-dimensional)
        if len(gather_indices.shape) == 0:
            axis = node.attrs.get('axis', 0)
            
            # Handle negative index
            index_value = int(gather_indices.item())
            if index_value < 0:
                index_value = index_value + inp.shape[axis]
            
            # Validate index is in bounds
            if index_value < 0 or index_value >= inp.shape[axis]:
                logging.warning(f"Index {index_value} out of bounds for axis {axis} with size {inp.shape[axis]} in node {node.name}")
                continue
            
            # Skip axis=0 (batch dimension)
            if axis == 0:
                logging.debug(f"Skipping Gather node {node.name} with axis=0 (batch dimension not supported)")
                continue
            
            logging.debug(f"Converting Gather {node.name} from scalar index {gather_indices} to 1D index [{index_value}] + Reshape")
            
            # Create new 1D index constant [index_value]
            new_indices = gs.Constant(
                name=f"{node.name}_indices_1d",
                values=np.array([index_value], dtype=np.int64)
            )
            
            # Create intermediate output for Gather
            input_dtype = node.inputs[0].dtype
            gather_out = gs.Variable(
                name=f"{node.name}_gather_out",
                dtype=input_dtype
            )
            
            # Create new Gather node with 1D indices
            new_gather = gs.Node(
                name=f"{node.name}_converted",  # Different name to avoid conflicts
                op="Gather",
                inputs=[node.inputs[0], new_indices],
                outputs=[gather_out],
                attrs={'axis': axis}
            )
            
            logging.debug(f"Adding Gather {new_gather.name} with 1D indices {new_indices.values} on axis {axis}")
            
            # Calculate output shape (remove dimension at 'axis')
            # Input shape after 1D Gather: keeps all dims but axis has size 1
            # We need to remove that dimension
            new_shape = list(inp.shape)
            new_shape = new_shape[:axis] + new_shape[axis + 1:]
            
            # Validate new shape
            if any(d is None or d < 0 for d in new_shape):
                logging.warning(f"Invalid reshape dimensions: {new_shape} for node {node.name}")
                continue
            
            # Create Reshape shape constant
            reshape_shape = gs.Constant(
                name=f"{node.name}_reshape_shape",
                values=np.array(new_shape, dtype=np.int64)
            )
            
            # Create Reshape node
            reshape = gs.Node(
                name=f"{node.name}_reshape",
                op="Reshape",
                inputs=[gather_out, reshape_shape],
                outputs=node.outputs
            )
            
            logging.debug(f"Adding Reshape {reshape.name} to shape {new_shape}")
            
            # Add nodes to graph
            graph.nodes.append(new_gather)
            graph.nodes.append(reshape)
            
            # Clear out original node outputs (marks it for removal)
            node.outputs.clear()
