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
Module containing DepthToSpace specific functions and optimizations
"""


import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np
from .common import find_in_layer


def tidl_insert_1x1_conv_before_depthtospace (graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Only depthtospace operations preceeded by a conv is supported
    """

    nodes = graph.nodes

    for node in nodes:
        if node.op == "DepthToSpace":
            inp = node.inputs[0]
    
            inp_shape = inp.shape
            if len(inp_shape) != 4:
                logging.warning(f"The input to Depth to Space was a tensor of shape {inp_shape}, expected a tensor of size 4.")
                continue
    
            # check if there is conv node before already, then dont do anything
            prev_layer = find_in_layer(node, 0) 
            if prev_layer is not None and prev_layer.op == "Conv" and prev_layer.attrs.get("kernel_shape") == [1, 1]:
                logging.info(f"There is a 1x1 conv before the depth to space layer {node.name}, not adding another.")
                continue
    
            weights = np.expand_dims(np.eye(inp_shape[1], inp_shape[1], dtype=np.float32), (-1, -2))
    
            conv_weights = gs.Constant(name=f"{node.name}_conv_weights", values=weights)
            conv_out = gs.Variable(name=f"{node.name}_Conv_out", dtype=np.float32)
            conv_node = gs.Node(name=f"{node.name}_Conv", op="Conv", attrs={"kernel_shape": [1, 1], "strides": [1, 1]}, inputs=[inp, conv_weights], outputs=[conv_out])
            logging.debug(f"Adding Conv node {conv_node.name} before DepthToSpace layer {node.name}")
            graph.nodes.append(conv_node)
    
            node.inputs[0] = conv_out

