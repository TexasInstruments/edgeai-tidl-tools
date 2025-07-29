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
Module containing GatherElements layer specific functions and optimizations
"""
import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np



def tidl_replace_tile_gatherelements_with_reshape_gather (graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Reads an ONNX model, finds Tile+GatherElements patterns, and replaces them with Reshape+Gather.
    """
    
    # Find all Tile nodes
    tile_nodes = [node for node in graph.nodes if node.op == "Tile"]
    
    for tile_node in tile_nodes:
        # Check if this Tile node feeds into a GatherElements node
        if not tile_node.outputs or not tile_node.outputs[0].outputs:
            continue
            
        gather_elements_nodes = [n for n in tile_node.outputs[0].outputs if n.op == "GatherElements"]
        
        if not gather_elements_nodes:
            continue
            
        # We found a Tile->GatherElements pattern
        for gather_elements_node in gather_elements_nodes:
            try:
                # Get the original input to the Tile
                input_tensor = tile_node.inputs[0]
                
                # Get the data tensor for the GatherElements (first input)
                data_tensor = gather_elements_node.inputs[0]
                
                # Get the GatherElements axis
                gather_axis = gather_elements_node.attrs.get("axis", 0)
                
                # Get the output of the GatherElements
                gather_output = gather_elements_node.outputs[0]
                
                # Determine the shape needed for Reshape
                # For a typical case where Tile expands a dimension that needs to be flattened
                if hasattr(input_tensor, 'shape') and input_tensor.shape is not None:
                    # For simplicity, let's flatten to the dimension needed for Gather
                    # This would need to be adjusted based on your specific use case
                    if len(input_tensor.shape) > 0:
                        # Calculate the flattened shape - removing the last dimension if it's 1
                        original_shape = list(input_tensor.shape)
                        new_shape = original_shape[1:2]
                        
                        # Create shape constant for Reshape
                        shape_constant = gs.Constant(name=f"{tile_node.name}_shape", 
                                                   values=np.array(new_shape, dtype=np.int64))
                        
                        # Create Reshape node
                        reshape_output = gs.Variable(name=f"{tile_node.name}_reshape_output", 
                                                   dtype=input_tensor.dtype,shape=new_shape)
                        
                        reshape_node = gs.Node(
                            op="Reshape",
                            name=f"{tile_node.name}_Reshape",
                            inputs=[input_tensor, shape_constant],
                            outputs=[reshape_output]
                        )
                        
                        # Create Gather node
                        gather_node = gs.Node(
                            op="Gather",
                            name=f"{gather_elements_node.name}_Gather",
                            inputs=[data_tensor, reshape_output],
                            outputs=[gather_output],  # Reuse the original output
                            attrs={"axis": gather_axis}
                        )
                        
                        # Add new nodes to the graph
                        graph.nodes.append(reshape_node)
                        graph.nodes.append(gather_node)
                        
                        # Disconnect the old nodes
                        tile_node.outputs[0].outputs.clear()
                        
                        # Mark the old nodes for removal
                        tile_node.outputs.clear()
                        gather_elements_node.outputs.clear()
                        
                        print(f"Replaced Tile+GatherElements pattern: {tile_node.name} -> {gather_elements_node.name}")
                        
            except Exception as e:
                print(f"Error replacing pattern for {tile_node.name}: {str(e)}")