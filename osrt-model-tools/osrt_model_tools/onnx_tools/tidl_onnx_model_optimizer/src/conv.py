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
Module containing Conv layer specific functions and optimizations
"""
import logging
import numpy as np
import onnx_graphsurgeon as gs
import onnx
import numpy as np
import copy
from .common import find_in_layers, find_out_layers


def tidl_convert_conv_large_pad_to_smaller_kernel (graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Convolution layer with large kernels and small inputs might
    be unsupported when pad is greater than the input dimension.
    This can be converted to Conv with smaller kernel and less
    pad for support
    """
    conv_nodes = [node for node in graph.nodes if node.op == "Conv"]
    tensors = graph.tensors()

    for conv in conv_nodes:
        # check if conversion is needed
        if 'pads' not in conv.attrs.keys(): # pads must be defined
            continue
        elif 'strides' not in conv.attrs.keys():
            continue
        elif len(conv.attrs['pads']) != 4:      # all 4 pad values are needed
            logging.debug(f"{conv.name} does not have 4 pad values, "
                          "cannot check for conversion")
            continue

        if 'dilations' not in conv.attrs.keys():
            dilation_h, dilation_w = 1, 1
        else:
            dilation_h, dilation_w = conv.attrs['dilations'][-2], conv.attrs['dilations'][-1]

        pad_t, pad_l, pad_b, pad_r = conv.attrs['pads'][0], conv.attrs['pads'][1], \
                                conv.attrs['pads'][2], conv.attrs['pads'][3]
        stride_h, stride_w = conv.attrs['strides'][-2], conv.attrs['strides'][-1]
        inp, weights = conv.inputs[0], conv.inputs[1]
        weight_tensor = np.array(tensors[weights.name].values, dtype= np.float32)
        bias = None
        outp = conv.outputs[0]
        if len(conv.inputs) > 2:
            bias = conv.inputs[2]

        if  bias is not None and \
            (len(bias.shape) != 1 or bias.shape[-1] != outp.shape[-3]):
            # if bias exists must be shape of only out channels
            logging.debug(f"{conv.name} has bias {bias.name} with shape not "
                          "supported for conversion, only bias with shape = "
                          "out_channels will be accepted")
            continue

        h, w = inp.shape[-1], inp.shape[-2]
        if  (w < pad_l) or (w < pad_r) or \
            (h < pad_t) or (h < pad_b):
            logging.debug(f"{conv.name} has pads {conv.attrs['pads']} > feature sizes "
                          f"({h}, {w}) --- converting kernel")

            # get original kernel shape
            if 'kernel_shape' in conv.attrs.keys():
                kernel_shape = conv.attrs['kernel_shape']
            else:
                kernel_shape = [weights.shape[-2], weights.shape[-1]]

            # calculate steps := how many times kernel can be placed on
            # the original data with pads
            w_padded = inp.shape[-1] + pad_l + pad_r
            h_padded = inp.shape[-2] + pad_t + pad_b

            # calculate effective kernel size based on dilation
            eff_kernel_size_w = kernel_shape[1] + (kernel_shape[1]-1)*(dilation_w-1)
            eff_kernel_size_h = kernel_shape[0] + (kernel_shape[0]-1)*(dilation_h-1)

            h_steps = w_padded - eff_kernel_size_w + 1        # horizontal steps
            v_steps = h_padded - eff_kernel_size_h + 1        # vertical steps
            # get relevant places in kernel

            top_left_index = tuple([int((eff_kernel_size_h - pad_t - 1)/dilation_h), int((eff_kernel_size_w - pad_l - 1)/dilation_w)])
            bottom_right_index = tuple([int((eff_kernel_size_h - pad_b - 1)/dilation_h), int((eff_kernel_size_w - pad_r - 1)/dilation_w)])
            
            # top_left_index = tuple([pad_t - v_steps + 1, pad_l - h_steps + 1])
            # bottom_right_index = tuple([pad_t + h - 1, pad_l + w - 1])
            reduced_weight_tensor = weight_tensor[:, :, 
                                                  top_left_index[0]: bottom_right_index[0] + 1, 
                                                  top_left_index[1]: bottom_right_index[1] + 1]
            reduced_kernel_shape = [reduced_weight_tensor.shape[-2],
                                    reduced_weight_tensor.shape[-1]]

            logging.debug(f"Reduced kernel shape from {kernel_shape} to {reduced_kernel_shape}")
            logging.debug(f"Slicing weights in positions top-left {top_left_index}"
                          f" and bottom-right {bottom_right_index}")

            out_h, out_w = outp.shape[-2], outp.shape[-1]
            reduced_pad_h =   ((out_h - 1) * stride_h - h + reduced_kernel_shape[0])//2
            reduced_pad_w =   ((out_w - 1) * stride_w - w + reduced_kernel_shape[1])//2
            reduced_pads = [reduced_pad_h, reduced_pad_w, reduced_pad_h, reduced_pad_w]

            logging.debug(f"Reducing pads to {reduced_pads}")

            # change the conv
            logging.debug(f"Changing {conv.name} input weights and attributes")
            conv.attrs['pads'] = np.array(reduced_pads, dtype= np.int64)
            conv.attrs['kernel_shape'] = np.array(reduced_kernel_shape, dtype= np.int64)
            conv.inputs[1] = gs.Constant(name= f"{weights.name}_reduced",
                                         values=reduced_weight_tensor)
            # bias need not change


def tidl_convert_conv_7x7_stride4_to_stride1(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Segformer model has a convolution layer with 7x7 kernel and 4 stride, converting the layer to the one
    with a stride of 1 using combination of maxpool and conv
    """

    for node in graph.nodes:
        if node.op == 'Conv':
            if node.attrs['kernel_shape'] == [7, 7] and node.attrs['strides'] == [4, 4]:
                node.attrs['strides'] = [1, 1]
                node.outputs[0].shape = None

                next_nodes = list(node.outputs[0].outputs)

                maxpool_output1 = gs.Variable(node.name.replace('Conv','MaxPool_out1'), dtype=np.float32)
                maxpool_output2 = gs.Variable(node.name.replace('Conv','MaxPool_out2'), dtype=np.float32)

                new_maxpool1 = gs.Node(op="MaxPool", name=node.name.replace('Conv', 'MaxPool1'),
                                                inputs=node.outputs,
                                            outputs=[maxpool_output1])
                new_maxpool1.attrs = dict(kernel_shape=[1, 1], strides=[2, 2])
                graph.nodes.append(new_maxpool1)
                logging.debug(f"Adding max pool node {new_maxpool1.name} in conv 7x7 conversion")

                new_maxpool2 = gs.Node(op="MaxPool", name=node.name.replace('Conv', 'MaxPool2'),
                                                inputs=[maxpool_output1],
                                            outputs=[maxpool_output2])
                new_maxpool2.attrs = dict(kernel_shape=[1, 1], strides=[2, 2])
                graph.nodes.append(new_maxpool2)
                logging.debug(f"Adding max pool node {new_maxpool2.name} in conv 7x7 conversion")

                for next_node in next_nodes:
                    index = next_node.inputs.index(node.outputs[0])
                    next_node.inputs[index] = maxpool_output2
                logging.debug(f"Changed the inputs of the conv {node.name}'s next layers in conv 7x7 conversion")


                    
def tidl_convert_conv_even_filter_to_odd(graph: gs.Graph, onnx_graph: onnx.GraphProto, zero_points={'Conv_Name_Fake_Example': -0.001}):
    '''
    Even-sized convolution kernels are not supported in TIDL
    Replace even-sized kernels with next-size up odd kernels, with padding handled appropriately. Additional filter weights are the zero_points
    Only square convolutions are supported here, but should be trivial to extend to length/height wise (Nx1 or 1xN) filters 

    :param zero_points: On a per-layer basis, the zero-point for asymmetric quantization. This is a dictionary where key is the layer name, and value is the zero-point for that layer (assumed same for all layers, i.e. no grouping)
    
    Some tricks are required here due to Conv layer implementation in TIDL being 'SAME' only. 
    This requires padding be handled outside the layer itself (due to asymmetric pads). 
    Asymmetric quantization is not well supported for these layers, since the zero-point is unknown until calibration. The zero-point param fills the additional convolution weights. This feature is untested
    '''

    conv_nodes = [node for node in graph.nodes if node.op == "Conv"]

    for conv in conv_nodes:
        kernel_shape = conv.attrs['kernel_shape']
        pads = conv.attrs['pads']
        weight_tensor = conv.inputs[1]

        conv_input = conv.inputs[0]

        MAX_SUPPORTED_CONV_KERNEL = 7 #7x7 is largest validated layer size
        if kernel_shape[0] % 2 == 0 and kernel_shape[0] < MAX_SUPPORTED_CONV_KERNEL and kernel_shape[1] == kernel_shape[0]:
            logging.debug('Promoting conv node (%s) size (%d x %d) to next size up' % (conv.name, kernel_shape[0], kernel_shape[1]))

            new_size = kernel_shape[0] + 1
            new_shape = [new_size, new_size]

            zero_p = zero_points.get(conv.name, 0)
            
            new_weights_shape = [*weight_tensor.shape[:2], *new_shape]

            # is it correct to put the zero point here or only in the layer padding
            new_weights = np.full(new_weights_shape, zero_p, dtype=np.float32)
            # We will pad left and top side of the filter weights with the fill_value / zero-point as we increase the spatial dimensions by 1
            new_weights[:,:,1:,1:] = weight_tensor.values

            new_weights_tensor = gs.Constant(weight_tensor.name, new_weights)
            conv.inputs[1] = new_weights_tensor


            conv.attrs['kernel_shape'] = new_shape
            logging.debug('  New conv kernel shape: ')


            pad_name = 'Pad/' + conv.name

            pads = copy.copy(pads)
            pads[0] += 1 # x1 (height) +1  to account for larger filter
            pads[1] += 1 # x2 (width) +1 to account for larger filter
            all_pads = np.asarray([0,0, pads[0], pads[1], 0, 0, pads[2], pads[3] ]) #incorporate all dimensions: depending on opset, may not support axis specification 
            pads_tensor = gs.Constant(pad_name + '_pads', np.asarray(all_pads, np.int64))
            fill_value_tensor = gs.Constant(pad_name + '_fill', np.asanyarray([zero_p], dtype=np.float32))


            conv.attrs['pads'] = [0,0,0,0]

            pad_attrs = {
                'mode' : 'constant'
            }
            pad_inputs = [conv_input, pads_tensor, fill_value_tensor]
            pad_outputs = [gs.Variable(pad_name+'_output', dtype=conv_input.dtype)]

            logging.debug('  Adding Pad layer with dimensions (%d,%d,%d,%d) and resetting conv pads to 0\'s' % (pads[0], pads[1], pads[2], pads[3]))

            pad_node = gs.Node('Pad', pad_name, pad_attrs, pad_inputs, pad_outputs)

            conv.inputs[0] = pad_outputs[0]
            graph.nodes.append(pad_node)
            #topographical sort should be run after this function completes to reorder the new Pad nodes


def tidl_convert_tr_conv_stride_n_tr_to_matmul(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Models having transpose -> conv(stride n) -> transpose need to be converted to 
    reshape -> transpose -> reshape -> matmul -> add -> reshape
    """
    for node in graph.nodes:
        if node.op == 'Conv':
            in_layers =  find_in_layers(node)
            out_layers = find_out_layers(node)
            if not(len(in_layers) == 1 and len(out_layers) == 1):
                logging.debug(f"convert_tr_conv_stride_n_tr_to_matmul is supported only in the case of transpose->conv->tranpose,\
                              skipping conversion of {node.name}")
                continue
            if not (in_layers[0].op == 'Transpose' and out_layers[0].op == 'Transpose'):
                logging.debug(f"convert_tr_conv_stride_n_tr_to_matmul is supported only in the case of transpose->conv->tranpose,\
                              skipping conversion of {node.name}")
                continue
            strides = node.attrs['strides']
            kernel_shape = node.attrs['kernel_shape']

            if not (kernel_shape[0] == kernel_shape[1] == strides[0] == strides[0]):
                logging.debug(f"Kernel shape should match the strides of the layer {node.name}, skipping")
                continue
            group = node.attrs['group']
            dilations = node.attrs['dilations']
            if group != 1 or dilations[0] != 1:
                logging.debug(f"The rule is currently not applicable for group or dilation > 1, have it in {node.name}, skipping")
                continue
            if isinstance(node.inputs[1], gs.Constant):
                conv_weights = node.inputs[1].values
            else:
                logging.debug(f"Weights of the conv node {node.name} are not a constant, skipping the conversion")
                continue  
            if len(node.inputs) > 2:
                conv_bias = node.inputs[2].values
            else:
                logging.info(f"The conv node {node.name} does not have a bias node, will not add an extra add layer")
                conv_bias = None

            input_shape = in_layers[0].inputs[0].shape
            if len(input_shape) != 4:
                logging.info(f"The input shape of the conv node {node.name} is not 4, skipping")
                continue
            if (input_shape[1] % kernel_shape[0] != 0) or (input_shape[2] % kernel_shape[1] != 0):
                logging.info(f"The input shape of the conv node {node.name} are not divisible by kernel, skipping")
                continue

            reshape_out_node_1 = gs.Variable(node.name + "rehsp_1_out")
            reshape_1_shape = np.array([input_shape[0], input_shape[1]//kernel_shape[0], kernel_shape[0], input_shape[2]//kernel_shape[1], \
                                        kernel_shape[1], input_shape[3]], dtype = np.int64)
            reshp_node_1 = gs.Node(op="Reshape", name=node.name + "rehsp_1", \
                                   inputs=[in_layers[0].inputs[0], gs.Constant(node.name + "rehsp_1_shape", reshape_1_shape)], \
                                    outputs=[reshape_out_node_1])
            graph.nodes.append(reshp_node_1)
            logging.debug(f"Added node {reshape_out_node_1.name}")

            transpose_out_node = gs.Variable(node.name + "transpose_1_out")
            transpose_node = gs.Node(op="Transpose", name=node.name + "Transpose", \
                                     attrs= {'perm': [0,1,3,2,4,5]}, inputs=[reshape_out_node_1],\
                                        outputs = [transpose_out_node])
            graph.nodes.append(transpose_node)
            logging.debug(f"Added node {transpose_node.name}")

            reshape_out_node_2 = gs.Variable(node.name + "rehsp_2_out")
            reshape_2_shape = np.array([input_shape[0], (input_shape[1]//kernel_shape[0])*(input_shape[2]//kernel_shape[1]), \
                                        kernel_shape[0]*kernel_shape[1]*input_shape[3]], dtype = np.int64)
            reshp_node_2 = gs.Node(op="Reshape", name=node.name + "rehsp_2", \
                                   inputs=[transpose_out_node, gs.Constant(node.name + "rehsp_2_shape", reshape_2_shape)], \
                                    outputs=[reshape_out_node_2])
            graph.nodes.append(reshp_node_2)
            logging.debug(f"Added node {reshp_node_2.name}")
            
            matmul_out = gs.Variable(node.name + 'matmul_out')
            matmul_weights = np.reshape(np.transpose(conv_weights, (2,3,1,0)), (kernel_shape[0]*kernel_shape[1]*input_shape[3], -1))
            matmul_node = gs.Node(op="MatMul", name = node.name + "matmul", \
                                   inputs=[reshape_out_node_2, gs.Constant(node.name + "matmul_weights", matmul_weights)], \
                                    outputs=[matmul_out])
            graph.nodes.append(matmul_node)
            logging.debug(f"Added node {matmul_node.name}")

            if conv_bias is not None:
                add_out = gs.Variable(node.name + 'add_out')
                add_bias = np.expand_dims(conv_bias, axis=(0,1))
                add_node = gs.Node(op="Add", name = node.name + "add", \
                                   inputs = [matmul_out, gs.Constant(node.name + "add_bias", np.array(add_bias, dtype=np.float32))], outputs = [add_out])
                graph.nodes.append(add_node)
                logging.debug(f"Added node {add_node.name}")
            else:
                add_out = matmul_out
                logging.debug(f"Bias node is not present in the conv node {node.name}, not adding the add node")

            reshape_3_shape = np.array([input_shape[0], input_shape[1]//kernel_shape[0], input_shape[2]//kernel_shape[1], -1], dtype = np.int64)
            reshp_node_3 = gs.Node(op="Reshape", name=node.name + "rehsp_3", \
                                   inputs=[add_out, gs.Constant(node.name + "rehsp_3_shape", reshape_3_shape)], \
                                    outputs=[out_layers[0].outputs[0]])
            graph.nodes.append(reshp_node_3)
            logging.debug(f"Added node {reshp_node_3.name}")

            node.outputs.clear()
            in_layers[0].outputs.clear()
            out_layers[0].outputs.clear()
