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
Module containing Tile layer specific functions and optimizations
"""
import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np


def tidl_convert_tile_to_expand_for_size1_dims(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """Convert Tile to Expand only when repeating size-1 dimensions."""
    
    nodes = graph.nodes
    
    for node in nodes:
        if node.op == "Tile":
            try:
                if len(node.inputs) != 2:
                    continue
                
                input_tensor = node.inputs[0]
                repeats_tensor = node.inputs[1]
                
                # Get input shape
                input_shape = input_tensor.shape
                if not input_shape or any(d is None for d in input_shape):
                    continue
                
                # Get repeats
                if isinstance(repeats_tensor, gs.Constant):
                    repeats = repeats_tensor.values
                elif hasattr(repeats_tensor, 'values') and repeats_tensor.values is not None:
                    repeats = repeats_tensor.values
                else:
                    continue
                
                repeats = np.array(repeats).flatten().astype(np.int64)
                if np.all(repeats==1):
                    out_tensor = node.outputs[0]
                    for out_node in list(out_tensor.outputs):
                        index = out_node.inputs.index(out_tensor)
                        out_node.inputs[index] = input_tensor
                    continue
                
                if len(repeats) != len(input_shape):
                    continue
                
                # Check: can only repeat size-1 dimensions
                can_convert = True
                for dim_size, repeat_count in zip(input_shape, repeats):
                    if repeat_count > 1 and dim_size != 1:
                        can_convert = False
                        break
                
                if not can_convert:
                    continue
                
                # Calculate target shape
                target_shape = [int(d * r) for d, r in zip(input_shape, repeats)]
                
                # Create shape constant
                shape_constant = gs.Constant(
                    name=f"{node.name}_shape",
                    values=np.array(target_shape, dtype=np.int64)
                )
                
                # Create Expand node
                expand_node = gs.Node(
                    op="Expand",
                    name=f"{node.name}_expand",
                    inputs=[input_tensor, shape_constant],
                    outputs=node.outputs
                )
                
                graph.nodes.append(expand_node)
                
                for output in expand_node.outputs:
                    output.inputs.clear()
                    output.inputs.append(expand_node)
                
                node.outputs.clear()
                
                logging.debug(f"Converted Tile to Expand: {node.name}, shape {input_shape} -> {target_shape}")
                
            except Exception as e:
                logging.warning(f"Failed to convert Tile node {node.name}: {e}")
                continue