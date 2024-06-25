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
from .common import id_generator, find_out_layer, is_single_const_single_var_input

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
                        tr_out = gs.Variable(name= f"{inp.name}_Conv_Transpose_out_{id_generator.get_id()}", dtype= np.float32)
                        tr_attr = {'perm': [1, 0]}
                        tr = gs.Node(name= f"{inp.name}_Conv_Transpose__{id_generator.get_id()}", op= "Transpose", attrs= tr_attr,
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
                    reshp_out = gs.Variable(name= f"{inp.name}_Conv_Reshape_out_{id_generator.get_id()}", dtype= np.float32)
                    reshp_shape = gs.Constant(name= f"{inp.name}_Conv_Reshape_shape_{id_generator.get_id()}",
                                              values= np.array([1, w, h, 1], dtype= np.int64))
                    reshp = gs.Node(name= f"{inp.name}_Conv_Reshape_{id_generator.get_id()}", op= "Reshape",
                                    inputs= [new_inp, reshp_shape], outputs= [reshp_out])

                    logging.debug(f"Adding Reshape {reshp.name} for converting input to "
                                  f"shape {[1, w, h, 1]}")
                    graph.nodes.append(reshp)
                    #forward input
                    new_inp = reshp_out
                elif len(inp_dim) == 3:
                    tr_out = gs.Variable(name= f"{inp.name}_Conv_Transpose_out_{id_generator.get_id()}", dtype= np.float32)
                    # set perm array
                    perm = list(range(len(inp_dim)))
                    perm = [perm[-1]] + perm[-3:-1]
                    tr_attr = {'perm': perm}
                    tr = gs.Node(name= f"{inp.name}_Conv_Transpose_{id_generator.get_id()}", op= "Transpose", attrs= tr_attr,
                                    inputs= [inp], outputs= [tr_out])
                    # add transpose
                    logging.debug(f"Adding Transpose {tr.name} for converting input from "
                                    f"{tuple(inp_dim)} to {tuple(perm)}")
                    graph.nodes.append(tr)
                    # forward input
                    new_inp = tr_out
                    # as this is 3D input, need to be converted to 4D for Conv layer
                    # add reshape
                    reshp_out = gs.Variable(name= f"{inp.name}_Conv_Reshape_out_{id_generator.get_id()}", dtype= np.float32)
                    reshp_shape = gs.Constant(name= f"{inp.name}_Conv_Reshape_shape_{id_generator.get_id()}",
                                              values= np.array([1, inp_dim[-1]] + inp_dim[-3:-1],
                                                               dtype= np.int64))
                    reshp = gs.Node(name= f"{inp.name}_Conv_Reshape_{id_generator.get_id()}", op= "Reshape",
                                    inputs= [new_inp, reshp_shape], outputs= [reshp_out])

                    logging.debug(f"Adding Reshape {reshp.name} for converting input to "
                                  f"shape {reshp_shape.values}")
                    graph.nodes.append(reshp)
                    #forward input
                    new_inp = reshp_out

                else:
                    logging.critical("Input to matmul with unsupported number of dimension "
                                     "is not supported for conversion currently")
                    continue
            # only one transpose will suffice
            else:
                tr_out = gs.Variable(name= f"{inp.name}_Conv_Transpose_out_{id_generator.get_id()}", dtype= np.float32)
                # set perm array
                perm = list(range(len(inp_dim)))
                perm = perm[:-3] + [perm[-1]] + perm[-3:-1]
                tr_attr = {'perm': perm}

                tr = gs.Node(name= f"{inp.name}_Conv_Transpose_{id_generator.get_id()}", op= "Transpose", attrs= tr_attr,
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

            conv_out = gs.Variable(name= f"{node.name}_Conv_out_{id_generator.get_id()}", dtype= np.float32)
            conv_kernel = gs.Constant(name= f"{node.name}_Conv_weights_{id_generator.get_id()}", values= kernel)
            conv = gs.Node(name= f"{node.name}_Conv_{id_generator.get_id()}", op= "Conv", attrs= conv_attrs,
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
                        reshp_out = gs.Variable(name= f"{node.name}_Conv_Reshape_out_{id_generator.get_id()}", dtype= np.float32)
                        reshp_out_list = [reshp_out]
                    else:
                        new_shape = [1, m]
                        reshp_out_list = node.outputs

                    reshp_shape = gs.Constant(name= f"{node.name}_Conv_Reshape_shape_{id_generator.get_id()}",
                                              values= np.array(new_shape, dtype= np.int64))
                    reshp = gs.Node(name= f"{node.name}_Conv_Reshape_{id_generator.get_id()}", op= "Reshape",
                                    inputs= [new_inp, reshp_shape], outputs= reshp_out_list)
                    logging.debug(f"Adding Reshape {reshp.name} for converting output to "
                                  f"shape {new_shape}")
                    graph.nodes.append(reshp)
                    # forward input
                    new_inp = reshp_out

                    # not special case: H != 1
                    if h != 1:
                        tr_attr = {'perm': [1, 0]}
                        tr = gs.Node(name= f"{node.name}_Conv_Transpose_{id_generator.get_id()}", op= "Transpose", attrs= tr_attr,
                                     inputs= [new_inp], outputs= node.outputs)

                        # add transpose
                        logging.debug(f"Adding Transpose {tr.name} for converting output from "
                                      f"{h}x{m} to {m}x{h}")
                        graph.nodes.append(tr)
                    else:
                        # report no transpose
                        logging.debug(f"Skipping Transpose for converting output from "
                                      f"{h}x{m} to {m}x{h} as h = 1")
                elif len(inp_dim) == 3:
                    # add reshape to remove the additional dimension
                    c, h, m = inp_dim[-3], inp_dim[-2], kernel_shape[0]
                    new_shape = [m, c, h]
                    reshp_out = gs.Variable(name= f"{node.name}_Conv_Reshape_out_{id_generator.get_id()}", dtype= np.float32)
                    reshp_shape = gs.Constant(name= f"{node.name}_Conv_Reshape_shape_{id_generator.get_id()}",
                                              values= np.array(new_shape, dtype= np.int64))
                    reshp = gs.Node(name= f"{node.name}_Conv_Reshape_{id_generator.get_id()}", op= "Reshape",
                                    inputs= [new_inp, reshp_shape], outputs= [reshp_out])

                    logging.debug(f"Adding Reshape {reshp.name} for converting output to "
                                  f"shape {new_shape}")
                    graph.nodes.append(reshp)
                    # forward input
                    new_inp = reshp_out

                    # add transpose
                    tr_attr = {'perm': [1, 2, 0]}
                    tr = gs.Node(name= f"{node.name}_Conv_Transpose_{id_generator.get_id()}", op= "Transpose", attrs= tr_attr,
                                    inputs= [new_inp], outputs= node.outputs)
                    # add transpose
                    logging.debug(f"Adding Transpose {tr.name} for converting output with "
                                  f"perm {tr_attr['perm']}")
                    graph.nodes.append(tr)



            # only one transpose will suffice
            else:
                # set perm array
                perm = list(range(len(inp_dim)))
                perm = perm[:-3] + perm[-2:] + [perm[-3]]
                tr_attr = {'perm': perm}

                tr = gs.Node(name= f"{inp.name}_Conv_Transpose_{id_generator.get_id()}", op= "Transpose", attrs= tr_attr,
                                inputs= [new_inp], outputs= node.outputs)

                # add transpose
                logging.debug(f"Adding Transpose {tr.name} for converting input"
                                f"to shape {tuple(perm)}")
                graph.nodes.append(tr)

            # clear this node
            node.outputs.clear()

def tidl_push_matmul_channel_in_height (graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Matmul layers with one input broadcasted across channel and other input with
    small plane size can have the channel and height axis merged
    """
    matmul_nodes = [node for node in graph.nodes if node.op == "MatMul"]

    for node in matmul_nodes:
        if  len(node.inputs) == 2 and \
            isinstance(node.inputs[0], gs.Variable) and \
            isinstance(node.inputs[1], gs.Constant) :
            # extract inputs
            var_inp, const_inp = node.inputs[0], node.inputs[1]
            var_inp_shape = var_inp.shape
            # check if bias is there
            bias = find_matmul_bias(node, graph)
            # check if it is broadcast channelwise
            if len(var_inp.shape) == 3 and len(const_inp.shape) == 2:
                logging.debug(f"MatMul Node {node.name} has channel-wise broadcast of "
                              f"{var_inp.shape} x {const_inp.shape}, "
                              f"Pushing channel dim {var_inp_shape[0]} to height in the input")
                c, h, w = var_inp_shape[0], var_inp_shape[1], var_inp_shape[2]
                # add reshape before
                reshp_shape = gs.Constant(name= f"{var_inp.name}_Reshape_shape_{id_generator.get_id()}",
                                        values= np.array([1, c*h, w], dtype= np.int64))
                reshp_out = gs.Variable(name= f"{var_inp.name}_Reshaped_out_{id_generator.get_id()}",
                                        dtype= np.float32)
                reshp = gs.Node(name= f"{var_inp.name}_Reshape_{id_generator.get_id()}", op= "Reshape",
                                inputs= [var_inp, reshp_shape], outputs= [reshp_out])
                # add node
                graph.nodes.append(reshp)
                logging.debug(f"Adding node {reshp.name} for reshaping input to {reshp_shape.values}")

                # redirect node input
                node.inputs[0] = reshp_out
                if bias is not None:
                    logging.debug(f"Found bias {bias.name}, adding reshape after bias")
                    # clear bias input shape
                    node.outputs[0].shape = None
                    # forward node to bias
                    node = bias
                # new output for matmul
                matmul_out = gs.Variable(name= f"{node.name}_Reshaped_out_{id_generator.get_id()}",
                                         dtype= np.float32)
                # add reshape after
                reshp_shape = gs.Constant(name= f"{node.name}_Reshape_shape_{id_generator.get_id()}",
                                        values= np.array([c, h, const_inp.shape[1]], dtype= np.int64))
                reshp = gs.Node(name= f"{node.name}_Reshape_{id_generator.get_id()}", op= "Reshape",
                                inputs= [matmul_out, reshp_shape],
                                outputs= node.outputs)
                # add node
                graph.nodes.append(reshp)
                logging.debug(f"Adding node {reshp.name} for reshaping input to {reshp_shape.values}")

                # redirect node output
                node.outputs[0] = matmul_out


def find_matmul_bias (matmul: gs.Node, graph: gs.Graph) -> gs.Node|None:
    """
    Given a MatMul node and a graph, return node if it is a bias
    after the MatMul
    checks:
    1. if an Add is after MatMul
    2. Add has constant input
    3. Constant input of Add has appropriate dimension
    like, if MatMul has HxM * MxN then Add must have
    constant input N
    """
    assert matmul.op == "MatMul", "Not MatMul node, cannot find bias for this node"


    if  len(matmul.inputs) == 2 and \
        isinstance(matmul.inputs[0], gs.Variable) and \
        isinstance(matmul.inputs[1], gs.Constant):
        # search for bias add
        for node in graph.nodes:
            if  find_out_layer(matmul, 0) == node:
                if node.op == "Add" and is_single_const_single_var_input(node):
                    # check for dimension match
                    matmul_const_inp_shape = matmul.inputs[1].shape
                    add_const_inp_shape = node.inputs[1].shape if \
                        isinstance(node.inputs[1], gs.Constant) else node.inputs[0].shape
                    if matmul_const_inp_shape[-1] == add_const_inp_shape[-1]:
                        return node
                #endif
            #endif
        #endfor
    #endif
    return None
