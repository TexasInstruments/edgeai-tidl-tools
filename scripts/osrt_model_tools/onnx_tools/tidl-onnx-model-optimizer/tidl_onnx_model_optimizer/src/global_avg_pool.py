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
Module containing Global average pooling layer specific functions and optimizations
"""


import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np


# threshold to consider global avg pooling as large
LARGE_GLOBAL_AVG_POOLING_THRESHOLD = 1024

def tidl_convert_large_global_avg_pooling_to_matmul (graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Global average pooling with large HxW values might be unoptimal,
    converting the input with a reshape from HxW to 1xHW and doing MatMul
    with a const tensor of dim HWx1 and value of 1/HW
    """

    nodes = graph.nodes

    for node in nodes:
        if node.op == "GlobalAveragePool":
            dim  = node.inputs[0].shape
            # check if large enough to convert
            if len(dim) < 2:
                logging.critical(f"GlobalAveragePooling {node.name} does not have at least 2 dimension"
                                 "in inputs, cannot convert")
                continue

            pool_size = dim[-1]*dim[-2]
            if (pool_size) < LARGE_GLOBAL_AVG_POOLING_THRESHOLD:
                logging.debug(f"GlobalAveragePooling {node.name} is not large enough with HxW as "
                              f"{dim[-2]}x{dim[-1]} => does not require conversion")
                continue

            # convert input from CxHxW to 1xCxHW (flatten inner dimensions)
            if len(dim) >= 3:
                new_shape = np.array(dim[:-3] + [1, dim[-3], pool_size], dtype= np.int64)
            # case of HxW handled
            else:
                new_shape = np.array([1, pool_size], dtype= np.int64)
            # add reshape
            reshp_out = gs.Variable(name= f"{node.name}_inp_Reshape_out", dtype= np.float32)
            reshp_shape = gs.Constant(name= f"{node.name}_inp_Reshape_shape", values= new_shape)
            reshp = gs.Node(name= f"{node.name}_inp_Reshape", op= "Reshape",
                            inputs= [node.inputs[0], reshp_shape], outputs= [reshp_out])

            logging.debug(f"Adding Reshape node {reshp.name} to convert input to shape {tuple(new_shape)}")
            graph.nodes.append(reshp)

            # setup const tensor
            matmul_const_tensor = np.array([1/pool_size] * pool_size, dtype= np.float32)
            matmul_const_tensor = np.reshape(matmul_const_tensor, (pool_size, 1))
            logging.debug(f"MatMul has Constant tensor with value {1/pool_size} "
                          f"and shape {matmul_const_tensor.shape}")

            # add matmul
            matmul_const_inp = gs.Constant(name= f"{node.name}_MatMul_const", values= matmul_const_tensor)
            matmul_output = gs.Variable(name= f"{node.name}_MatMul_out", dtype= np.float32)
            matmul = gs.Node(name= f"{node.name}_MatMul", op= "MatMul",
                             inputs= [reshp_out, matmul_const_inp], outputs= [matmul_output])
            logging.debug(f"Adding MatMul {matmul.name}")
            graph.nodes.append(matmul)

            # convert matmul output from D2xCx1 to D2xCx1x1 to match output
            new_shape = np.array(node.outputs[0].shape, dtype= np.int64)
            # add reshape
            reshp_out = gs.Variable(name= f"{node.name}_matmul_out_Reshape_out", dtype= np.float32)
            reshp_shape = gs.Constant(name= f"{node.name}_matmul_out_Reshape_shape", values= new_shape)
            reshp = gs.Node(name= f"{node.name}_matmul_out_Reshape", op= "Reshape",
                            inputs= [matmul.outputs[0], reshp_shape], outputs= node.outputs)

            logging.debug(f"Adding Reshape node {reshp.name} to convert input to shape {tuple(new_shape)}")
            graph.nodes.append(reshp)

            # clear node outputs
            node.outputs.clear()
