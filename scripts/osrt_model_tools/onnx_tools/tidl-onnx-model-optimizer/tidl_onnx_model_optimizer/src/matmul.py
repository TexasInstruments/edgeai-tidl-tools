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
Module containing MatMul layer specific functions and optimizations
"""
import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np


def tidl_modify_matmul(graph: gs.Graph, onnx_graph: onnx.GraphProto, args: dict):
    """
    Wrapper function to modify Gemm layers to satisfy TIDL constraints
    """
    if args['convert_matmul_to_conv_1x1s1']:
        logging.debug("Running convert_matmul_to_conv_1x1s1")
        tidl_convert_matmul_to_conv_1x1s1(graph, onnx_graph)

def tidl_convert_matmul_to_conv_1x1s1 (graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Function to convert MatMul layer to Convolution with kernel 1x1, stride 1x1
    """
    nodes = graph.nodes
    tensors = graph.tensors()

    for node in nodes:
        if node.op == "MatMul" and\
        isinstance(node.inputs[1], gs.Constant):
            # extract weights of MatMul
            weights = np.array(tensors[node.inputs[1].name].values, dtype=np.float32)

            # convert to kernel weights
            # swap last two indices
            weight_dim_indices = list(range(len(weights.shape)))
            temp = weight_dim_indices[-1]
            weight_dim_indices[-1] = weight_dim_indices[-2]
            weight_dim_indices[-2] = temp
            weights = np.transpose(weights, tuple(weight_dim_indices))

            # convert from MxN to MxNx1x1 for kernel
            kernel_shape = tuple(list(weights.shape) + [1, 1])
            kernel = np.reshape(weights, newshape=kernel_shape)

            logging.debug(f"Converting {node.name} weights to Conv kernel weights,"
                          f"tranposed with perm = {tuple(weight_dim_indices)} and reshaped "
                          f"to shape {kernel_shape}")

            inp = node.inputs[0]
            inp_dim = list(inp.shape)
            new_inp = None   # new input to be passed for added nodes
            # input must be converted to 4D as Conv needs 4D input
            if len(inp_dim) < 4:
                # HxW input
                # convert to 1xWxHx1
                # add a Tranpose (WxH) and Reshape (1xWxHx1)
                # if H == 1, we can skip the Transpose
                if len(inp_dim) == 2:
                    h, w = inp_dim[-2], inp_dim[-1]

                    # not special case: H != 1
                    if h != 1:
                        tr_out = gs.Variable(name= f"{inp.name}_Transpose_out", dtype= np.float32)
                        tr_attr = {'perm': [1, 0]}
                        tr = gs.Node(name= f"{inp.name}_Transpose", op= "Transpose", attrs= tr_attr,
                                     inputs= [inp], outputs= [tr_out])

                        # add transpose
                        logging.debug(f"Adding Transpose {tr.name} for converting input from "
                                      f"{h}x{w} to {w}x{h}")
                        graph.nodes.append(tr)
                        # forward the input with new added node
                        new_inp = tr_out
                    else:
                        # report no transpose
                        logging.debug(f"Skipping Transpose for converting input from "
                                      f"{h}x{w} to {w}x{h} as h = 1")
                        new_inp = inp

                    # add reshape
                    reshp_out = gs.Variable(name= f"{inp.name}_Reshape_out", dtype= np.float32)
                    reshp_shape = gs.Constant(name= f"{inp.name}_Reshape_shape",
                                              values= np.array([1, w, h, 1], dtype= np.int64))
                    reshp = gs.Node(name= f"{inp.name}_Reshape", op= "Reshape",
                                    inputs= [new_inp, reshp_shape], outputs= [reshp_out])

                    logging.debug(f"Adding Reshape {reshp.name} for converting input to "
                                  f"shape {[1, w, h, 1]}")
                    graph.nodes.append(reshp)

                    #forward input
                    new_inp = reshp_out
                else:
                    logging.critical("Input to matmul with 3 dimension is not supported for"
                                     "conversion currently")
                    continue
            # only one transpose will suffice
            else:
                tr_out = gs.Variable(name= f"{inp.name}_Transpose_out", dtype= np.float32)
                # set perm array
                perm = list(range(len(inp_dim)))
                perm = perm[:-3] + [perm[-1]] + perm[-3:-1]
                tr_attr = {'perm': perm}

                tr = gs.Node(name= f"{inp.name}_Transpose", op= "Transpose", attrs= tr_attr,
                                inputs= [inp], outputs= [tr_out])

                # add transpose
                logging.debug(f"Adding Transpose {tr.name} for converting input from "
                                f"{tuple(inp_dim)} to {tuple(perm)}")
                graph.nodes.append(tr)

                # forward input
                new_inp = tr_out

            # add Conv
            conv_attrs = {
                'kernel_shape': [1, 1],
                'strides': [1, 1],
            }

            conv_out = gs.Variable(name= f"{node.name}_Conv_out", dtype= np.float32)
            conv_kernel = gs.Constant(name= f"{node.name}_Conv_weights", values= kernel)
            conv = gs.Node(name= f"{node.name}_Conv", op= "Conv", attrs= conv_attrs,
                           inputs= [new_inp, conv_kernel], outputs= [conv_out])

            # add node
            logging.debug(f"Added Conv node {conv.name} with kernel 1x1, stride 1")
            graph.nodes.append(conv)

            # forward input
            new_inp = conv_out
            # restore output to proper shape
            if len(inp_dim) < 4:
                # HxW input
                # converted to 1xWxHx1
                # output is now 1xMxHx1
                # add Reshape to MxH and transpose to HxM
                # if H == 1, we can skip the Transpose and change reshape to 1xM directly
                if len(inp_dim) == 2:
                    h, m = inp_dim[-2], kernel_shape[0]

                    # add reshape
                    if h != 1:
                        new_shape = [m, h]
                        reshp_out = gs.Variable(name= f"{node.name}_Reshape_out", dtype= np.float32)
                        reshp_out_list = [reshp_out]
                        # forward input
                        new_inp = reshp_out
                    else:
                        new_shape = [1, m]
                        reshp_out_list = node.outputs

                    reshp_shape = gs.Constant(name= f"{node.name}_Reshape_shape",
                                              values= np.array(new_shape, dtype= np.int64))
                    reshp = gs.Node(name= f"{node.name}_Reshape", op= "Reshape",
                                    inputs= [new_inp, reshp_shape], outputs= reshp_out_list)

                    logging.debug(f"Adding Reshape {reshp.name} for converting output to "
                                  f"shape {new_shape}")
                    graph.nodes.append(reshp)

                    # not special case: H != 1
                    if h != 1:
                        tr_out = gs.Variable(name= f"{node.name}_Transpose_out", dtype= np.float32)
                        tr_attr = {'perm': [1, 0]}
                        tr = gs.Node(name= f"{node.name}_Transpose", op= "Transpose", attrs= tr_attr,
                                     inputs= [new_inp], outputs= node.outputs)

                        # add transpose
                        logging.debug(f"Adding Transpose {tr.name} for converting output from "
                                      f"{h}x{m} to {m}x{h}")
                        graph.nodes.append(tr)

                        # forward input
                        new_inp = tr_out
                    else:
                        # report no transpose
                        logging.debug(f"Skipping Transpose for converting output from "
                                      f"{h}x{m} to {m}x{h} as h = 1")

            # only one transpose will suffice
            else:
                # set perm array
                perm = list(range(len(inp_dim)))
                perm = perm[:-3] + perm[-2:] + [perm[-3]]
                tr_attr = {'perm': perm}

                tr = gs.Node(name= f"{inp.name}_Transpose", op= "Transpose", attrs= tr_attr,
                                inputs= [new_inp], outputs= node.outputs)

                # add transpose
                logging.debug(f"Adding Transpose {tr.name} for converting input"
                                f"to shape {tuple(perm)}")
                graph.nodes.append(tr)

            # clear this node
            node.outputs.clear()
