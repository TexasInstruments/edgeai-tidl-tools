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
Module containing Batchnorm specific functions and optimizations
"""


import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np



def tidl_convert_batchnorm_input_to_4D (graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Only 4D batchnorm (NCHW) with batchnorm on the channel is supported
    """

    nodes = graph.nodes

    for node in nodes:
        if node.op == "BatchNormalization":
            inp = node.inputs[0]
            dim  = list(inp.shape)
            # Check if less than 4D
            if len(dim) < 4:
                # add reshape with 1's appended at the end
                num_ones = 4 - len(dim)
                new_shape = np.array(dim + [1]*num_ones, dtype= np.int64)

                reshp_shape = gs.Constant(name= f"{node.name}_Reshape_4D_shape",
                              values= new_shape)
                reshp_out = gs.Variable(name= f"{node.name}_Reshape_4D_out_1", dtype= np.float32)
                reshp = gs.Node(name= f"{node.name}_Reshape_4D", op= "Reshape",
                                inputs= [inp, reshp_shape], outputs= [reshp_out])

                logging.debug(f"Adding Reshape node {reshp.name} to convert input to shape {tuple(new_shape)}")
                graph.nodes.append(reshp)

                # change input to batchnorm
                node.inputs[0] = reshp_out

                # redirected output
                bn_out = gs.Variable(name= f"{node.name}_Reshape_4D_out_2", dtype= np.float32)

                # add reshape to convert back to original
                original_shape = np.array(dim, dtype= np.int64)
                reshp_shape = gs.Constant(name= f"{node.name}_Reshape_Original_shape",
                              values= original_shape)
                reshp_out = gs.Variable(name= f"{node.name}_Reshape_Original_out", dtype= np.float32)
                reshp = gs.Node(name= f"{node.name}_Reshape_Original", op= "Reshape",
                                inputs= [bn_out, reshp_shape], outputs= node.outputs)

                logging.debug(f"Adding Reshape node {reshp.name} to convert input to shape {tuple(original_shape)}")
                graph.nodes.append(reshp)

                # change output of bn
                node.outputs = [bn_out]
