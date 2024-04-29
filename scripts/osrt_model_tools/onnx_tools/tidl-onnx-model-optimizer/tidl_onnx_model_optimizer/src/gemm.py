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
Module containing Gemm layer specific functions and optimizations
"""
import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np


def tidl_modify_gemm(graph: gs.Graph, onnx_graph: onnx.GraphProto, args: dict):
    """
    Wrapper function to modify Gemm layers to satisfy TIDL constraints
    """
    if args['convert_gemm_to_matmul_and_add']:
        logging.debug("Running convert_gemm_to_matmul_and_add")
        tidl_convert_gemm_to_matmul_and_add(graph, onnx_graph)


def tidl_convert_gemm_to_matmul_and_add (graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Convert Gemm layer with constant B input to Matmul and
    Gemm bias (if exists) to a following add layer
    """

    nodes = graph.nodes
    tensors = graph.tensors()

    for node in nodes:
        if node.op == "Gemm" and\
        isinstance(node.inputs[1], gs.Constant):		# check if B is constant input
            # check attributes
            if 'alpha' in node.attrs.keys() and node.attrs['alpha'] != 1:
                logging.critical(f"Gemm node {node.name} has unsupported alpha != 1, skipping change")
                continue

            if 'beta' in node.attrs.keys() and node.attrs['beta'] != 1:
                logging.critical(f"Gemm node {node.name} has unsupported beta != 1, skipping change")
                continue

            is_tranposed = node.attrs['transB'] if 'transB' in node.attrs.keys() else 0



            # extract weights and bias
            weights = np.array(tensors[node.inputs[1].name].values, dtype=np.float32)
            bias = None
            if len(node.inputs) > 2:	# bias exists
                bias = np.array(tensors[node.inputs[2].name].values, dtype=np.float32)

            if is_tranposed:
                # swap last two indices
                weight_dim_indices = list(range(len(weights.shape)))
                temp = weight_dim_indices[-1]
                weight_dim_indices[-1] = weight_dim_indices[-2]
                weight_dim_indices[-2] = temp

                weights = np.transpose(weights, tuple(weight_dim_indices))
                logging.debug(f"transB is set to True, tranposing weights with perm = {tuple(weight_dim_indices)}")

            # add MatMul node
            if bias is not None:
                matmul_out = gs.Variable(name= f"{node.name}_MatMul_out", dtype= np.float32)
            else:
                matmul_out = node.outputs[0]

            matmul_wts = gs.Constant(name= f"{node.name}_MatMul_weights", values=weights)
            matmul = gs.Node(name= f"{node.name}_MatMul", op= "MatMul",
                             inputs= [node.inputs[0], matmul_wts], outputs= [matmul_out])
            logging.debug(f"Adding MatMul node {matmul.name} with weights from {node.name}")
            graph.nodes.append(matmul)

            # add Add if bias is not None
            if bias is not None:
                # create add
                # add_out = gs.Variable(name= f"{node.name}_Bias_Add_out", dtype= np.float32)
                add_out = node.outputs[0]
                add_wts = gs.Constant(name= f"{node.name}_Bias_Add_constant", values= bias)
                add = gs.Node(name= f"{node.name}_Bias_Add", op= "Add",
                              inputs= [matmul_out, add_wts], outputs= [add_out])
                logging.debug(f"Adding Add node {add.name} with bias from {node.name}")

                graph.nodes.append(add)

            # clear this node's output
            node.outputs.clear()
