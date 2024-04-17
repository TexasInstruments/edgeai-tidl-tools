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
Module containing Concat layer specific functions and optimizations
"""


import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np



def tidl_modify_concat(graph: gs.Graph, onnx_graph: onnx.GraphProto, args: dict):
    """
    Wrapper function to modify resize layers to satisfy TIDL constraints
    """
    if args['convert_concat_axis_width_to_channel']:
        logging.debug("Performing convert_concat_axis_width_to_channel")
        tidl_convert_concat_axis_width_to_channel(graph, onnx_graph)

    graph.cleanup().toposort()


def tidl_convert_concat_axis_width_to_channel (graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Convert axis of a Concat layer from width to channel,
    adding proper Transposes before and after for retaining
    functionalilty
    """
    tensors = graph.tensors()

    for node in graph.nodes:
        # check if concat
        if node.op == "Concat":
            if  (   # must be width axis
                    (node.attrs['axis'] == -1) or    \
                    (node.attrs['axis'] == (len(node.inputs[0].shape) -1))
                ):

                valid_input = True
                for inp in node.inputs:
                    # need at least channel dim
                    if len(inp.shape) < 3:
                        logging.critical(f"{inp.name} input to {node.name} has no channel dim"
                                         "Unable to convert axis to channel")
                        valid_input = False

                # skip this node if not valid input
                if not valid_input:
                    continue

                # change axis of concat
                # ASSUMES:: same shape for all inputs, take from first one
                node.attrs['axis'] = len(node.inputs[0].shape) - 3

                ## modify inputs to the concat
                for idx, inp in enumerate(node.inputs):
                    ## constant inputs
                    if isinstance(inp, gs.Constant):
                        concat_const_tensor = np.array(tensors[inp.name].values, dtype=np.float32)
                        # get indices array
                        perm = list(range(len(concat_const_tensor.shape)))
                        # swap channel and width
                        temp = perm[-1]
                        perm[-1] = perm[-3]
                        perm[-3] = temp
                        # transpose const input
                        logging.debug(f"Transposing constant input {inp.name} to {node.name}:"
                                      f"perm= {tuple(perm)}")
                        tr_concat_const_tensor = np.transpose(concat_const_tensor, tuple(perm))
                        node.inputs[idx] = gs.Constant(name=f'{inp.name}_transposed',
                                                       values=tr_concat_const_tensor)

                    ## variable inputs
                    else:
                        # get indices array
                        perm = list(range(len(inp.shape)))
                        # swap channel and width
                        temp = perm[-1]
                        perm[-1] = perm[-3]
                        perm[-3] = temp

                        # create transpose layer with this permutation
                        transpose_out = gs.Variable(name=f'{inp.name}_transposed', dtype= np.float32)
                        transpose_node = gs.Node(name=f'transpose_{inp.name}', op= 'Transpose',
                                                 attrs= {"perm":perm}, inputs=[inp],
                                                 outputs=[transpose_out])
                        logging.debug(f"Adding node {transpose_node.name} with:"
                                      f"perm= {tuple(perm)}")
                        graph.nodes.append(transpose_node)

                        # feed new input to concat
                        node.inputs[idx] = transpose_out

                ## modify outputs from concat
                for idx, outp in enumerate(node.outputs):
                    # get indices array
                    perm = list(range(len(outp.shape)))
                    # swap channel and width
                    temp = perm[-1]
                    perm[-1] = perm[-3]
                    perm[-3] = temp

                    # create transpose layer with this permutation
                    transpose_in = gs.Variable(name=f'{outp.name}_transposed', dtype= np.float32)
                    transpose_node = gs.Node(name=f'transpose_{outp.name}', op= 'Transpose',
                                                attrs= {"perm":perm}, inputs=[transpose_in],
                                                outputs=[outp])
                    logging.debug(f"Adding node {transpose_node.name} with:"
                                    f"perm= {tuple(perm)}")
                    graph.nodes.append(transpose_node)

                    # feed new input to concat
                    node.outputs[idx] = transpose_in

