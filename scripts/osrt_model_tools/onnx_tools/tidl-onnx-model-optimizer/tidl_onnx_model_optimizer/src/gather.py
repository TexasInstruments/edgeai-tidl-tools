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



def tidl_modify_gather(graph: gs.Graph, onnx_graph: onnx.GraphProto, args: dict):
    """
    Wrapper function to modify gather layers to satisfy TIDL constraints
    """
    if args['convert_gather_with_single_index_to_slice']:
        logging.debug("Running convert_gather_with_single_index_to_slice")
        tidl_convert_gather_with_single_index_to_slice(graph, onnx_graph)


def tidl_convert_gather_with_single_index_to_slice(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    When Gather has single index = t, can be converted to
    Slice [t, t+1] on the same axis
    """
    nodes = graph.nodes
    tensors = graph.tensors()

    for node in nodes:
        if node.op == "Gather":
            inp, idx = node.inputs[0], node.inputs[1]
            # check if single index
            gather_indices = np.array(tensors[idx.name].values, dtype= np.int64)
            if len(gather_indices.shape) == 0:

                axis = node.attrs['axis']
                # add Slice
                slice_out = gs.Variable(name= f"{node.name}_Slice_out", dtype= np.float32)
                starts = np.reshape(gather_indices, (1,))
                slice_starts = gs.Constant(name= f"{node.name}_Slice_starts",
                                           values= starts)
                ends = starts + 1
                slice_ends = gs.Constant(name= f"{node.name}_Slice_ends",
                                           values= ends)
                slice_axes = gs.Constant(name= f"{node.name}_Slice_axes",
                                           values= np.array([axis], dtype= np.int64))
                slc = gs.Node(name= f"{node.name}_Slice", op= "Slice",
                                inputs= [node.inputs[0], slice_starts, slice_ends, slice_axes],
                                outputs= [slice_out])


                logging.debug(f"Adding Slice {slc.name} with axes {slice_axes.values}, "
                              f"starts {slice_starts.values} and ends {slice_ends.values}")
                graph.nodes.append(slc)

                # add reshape to fix extra singular dim from slice
                new_shape = list(inp.shape)
                new_shape = np.array(new_shape[:axis] + new_shape[axis + 1: ],
                                     dtype= np.int64)
                reshp_shape = gs.Constant(name= f"{node.name}_Reshape_shape",
                                          values= new_shape)

                reshp = gs.Node(name= f"{node.name}_Reshape", op= "Reshape",
                                inputs= [slice_out, reshp_shape], outputs= node.outputs)

                logging.debug(f"Adding Reshape {reshp.name} to reshape Sliced output to "
                              f"{new_shape}")
                graph.nodes.append(reshp)


                # clear out original node outputs and remove
                node.outputs.clear()
