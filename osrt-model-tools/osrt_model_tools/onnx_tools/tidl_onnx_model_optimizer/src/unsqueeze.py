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
Module containing Unsqueeze layer specific functions and optimizations
"""
import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np
from .common import has_unk_axis

def tidl_convert_unsqueeze_to_reshape (graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Converting the unsqueeze layer to reshape to support it in TIDL
    """
    nodes = graph.nodes
    
    for node in nodes:
        if (node.op == "Unsqueeze") and (not has_unk_axis(node.inputs[0])):
            if hasattr(node, 'attrs') and ('axes' in node.attrs):
                print(node.attrs['axes'])
                axes = node.attrs['axes']
                axes = np.array(axes)
            elif len(node.inputs)> 1 and isinstance(node.inputs[1], gs.Constant):
                axes = node.inputs[1].values
            else:
                logging.info(f"axes is not present in inputs for node {node.name}")
                continue
            inp = node.inputs[0]
            inp, axes = node.inputs[0], node.inputs[1].values
            orig_shape = inp.shape
            axes = np.where(axes<0, axes+len(orig_shape) + len(axes), axes)
            axes.sort()
            new_shape = np.array([], dtype=np.int64)
            i = 0
            j = 0
            while (i <= max(axes)) or (j < len(orig_shape)):
                if i in axes:
                    new_shape = np.append(new_shape, 1)
                else:
                    new_shape = np.append(new_shape, orig_shape[j])
                    j += 1
                i += 1
                
            reshp_shape = gs.Constant(name= f"{node.name}_Reshape_shape",
                                          values= new_shape)

            reshp = gs.Node(name= f"{node.name}_Reshape", op= "Reshape",
                            inputs= [inp, reshp_shape], outputs= node.outputs)

            logging.debug(f"Adding Reshape {reshp.name} to replace unsqueeze with new_shape as :  "
                            f"{new_shape}")
            
            graph.nodes.append(reshp)

            # clear out original node outputs and remove
            node.outputs.clear()
            
            
def tidl_eliminate_unsqueeze(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Eliminate Unsqueeze nodes when the input is a constant initializer (gs.Constant)
    and axes are provided as a constant input
    Replace the Unsqueeze node with a new constant with the correct shape.
    """
    nodes_to_remove = []

    for node in graph.nodes:
        if node.op != "Unsqueeze":
            continue

        inp = node.inputs[0]
        # Only handle constant initializer input
        if not isinstance(inp, gs.Constant):
            logging.debug(f"Skipping Unsqueeze node '{node.name}': input is not a constant.")
            continue

        if len(node.inputs) < 2 or not isinstance(node.inputs[1], gs.Constant):
            logging.debug(f"Skipping Unsqueeze node '{node.name}': axes input is not a constant or missing.")
            continue
        axes = node.inputs[1].values
        if axes is None:
            logging.debug(f"Skipping Unsqueeze node '{node.name}': axes values are None.")
            continue
        axes = axes.tolist() if hasattr(axes, "tolist") else list(axes)

        orig_shape = list(inp.values.shape)
        num_axes = len(axes)
        output_rank = num_axes + len(orig_shape)

        # handle negative axes
        axes = [a + output_rank if a < 0 else a for a in axes]
        axes = sorted(axes)

        # Generate new dims (all 0s, will fill in below)
        new_dims = [0] * output_rank
        for axis in axes:
            if axis >= len(new_dims):
                logging.warning(f"UnsqueezeElimination cannot remove node due to invalid axes: {node.name}")
                break
            new_dims[axis] = 1
        else:
            # Fill in the non-unsqueezed dims from the input shape
            begin = 0
            for i in range(output_rank):
                if new_dims[i] == 0:
                    new_dims[i] = orig_shape[begin]
                    begin += 1

            logging.debug(f"Eliminating Unsqueeze node '{node.name}': reshaping constant from {orig_shape} to {new_dims}.")
            # Create new constant with new shape
            new_const = gs.Constant(name=f"{inp.name}_unsqueeze_elim", values=inp.values.reshape(new_dims))
            # Replace all outputs of the node with the new constant
            for out in node.outputs:
                logging.debug(f"Redirecting output '{out.name}' of node '{node.name}' to new constant '{new_const.name}'.")
                out.inputs = [new_const]
            nodes_to_remove.append(node)
            logging.info(f"Eliminated Unsqueeze node: {node.name}")

    for node in nodes_to_remove:
        logging.info(f"Removing node: {node.name}")
        graph.nodes.remove(node)