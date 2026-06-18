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
Module containing SpacetoDepth specific functions and optimizations
"""


import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np



def tidl_convert_space2depth_to_reshp_tr_reshp (graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Convert space to depth node to reshape->transpose->reshape nodes
    """

    nodes = graph.nodes

    for node in nodes:
        if node.op == "SpaceToDepth":
            inp = node.inputs[0]
    
            inp_shape = inp.shape
            if len(inp_shape) != 4:
                logging.warning(f"The input to Depth to Space was a tensor of shape {inp_shape}, expected a tensor of size 4.")
                continue
    
            if hasattr(node, 'attrs'):
                if 'blocksize' in node.attrs and node.attrs['blocksize'] is not None:
                    block_size = node.attrs['blocksize']
                else:
                    logging.info(f"Block size is not present in depth2space node {node.name}, skipping the conversion")


            new_shape = np.array([inp_shape[0], inp_shape[1], inp_shape[1]//block_size, block_size, inp_shape[2]//block_size, block_size], dtype= np.int64)
            reshp_shape1 = gs.Constant(name= f"{node.name}_Reshape_shape_1", values= new_shape)
            reshape_out_1 = gs.Variable(name=f"{node.name}_Reshape_out_1", dtype=np.float32)
            reshape_node1 = gs.Node(name=f"{node.name}_Reshape_1", op="Reshape", inputs=[inp, reshp_shape1], outputs=[reshape_out_1])
            graph.nodes.append(reshape_node1)
            logging.debug(f"Adding Reshape node {reshape_node1.name} instead of SpaceToDepth layer {node.name}")

            transpose_out = gs.Variable(name=f"{node.name}_transpose_out", dtype=np.float32)
            permidx = [0, 3, 5, 1, 2, 4]
            transpose_node = gs.Node(op="Transpose", name=f"{node.name}_transpose_node",
                                    attrs={"perm": permidx}, inputs=[reshape_out_1],
                                    outputs=[transpose_out])
            
            graph.nodes.append(transpose_node)
            logging.debug(f"Adding Transpose node {transpose_node.name} instead of SpaceToDepth layer {node.name}")

            new_shape = np.array((inp_shape[0], inp_shape[1]*(block_size*block_size), inp_shape[2]//block_size, inp_shape[3]//block_size), dtype= np.int64)
            reshp_shape2 = gs.Constant(name= f"{node.name}_Reshape_shape_2", values= new_shape)
            reshape_node2 = gs.Node(name=f"{node.name}_Reshape_2", op="Reshape", inputs=[transpose_out, reshp_shape2], outputs=node.outputs)
            graph.nodes.append(reshape_node2)
            logging.debug(f"Adding Reshape node {reshape_node2.name} instead of SpaceToDepth layer {node.name}")
            
            node.outputs.clear()


