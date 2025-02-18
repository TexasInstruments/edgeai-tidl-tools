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
Module containing Resize layer specific functions and optimizations
"""

import logging
import onnx_graphsurgeon as gs
import onnx
from onnx import numpy_helper
import numpy as np
from .common import *


def tidl_convert_resize_params_size_to_scale(graph: gs.Graph,
                                             onnx_graph: onnx.GraphProto) :
    """
    If some resize layer has size defined instead of scale,
    change to scale accordingly
    ------------------------------------------------------
    Assumes that the inputs are in following order
    [Variable Input, roi, scales, sizes]
    """
    tensors = graph.tensors()

    for node in graph.nodes:
        # check if satisfy criteria
        if node.op == "Resize":
            # if not 4 inputs, do not consider
            if len(node.inputs) == 4:
                var, roi, scales, sizes = node.inputs[0], node.inputs[1], node.inputs[2], node.inputs[3]
                # if sizes is not empty and scales are empty and both are constant
                if (not np.any(scales.shape)) and np.any(sizes.shape) \
                    and isinstance(sizes, gs.Constant):
                    reshaped_sizes = np.array(tensors[sizes.name].values, dtype=np.float32)
                    in_sizes = np.array(var.shape, dtype=np.float32)
                    scale_params = reshaped_sizes/in_sizes
                    #roi initializer
                    if roi.shape is None:
                        roi_const = np.array([], dtype=np.float32)
                        initializer_name = f"{node.name}.roi_const_name"
                        roi_const_name = numpy_helper.from_array(roi_const, name=initializer_name)
                        onnx_graph.initializer.append(roi_const_name)
                        roi = gs.Constant(name=initializer_name, values=roi_const) 
                    # check scale parameter values
                    for t in scale_params:
                        if np.log2(t) != int(np.log2(t)):
                            logging.warning(f"{node.name} has scale not as power of "
                                            f"2 which is not supported by TIDL")
                    scale_name = f"{node.name}.scales"
                    scales_updated = gs.Constant(name=scale_name, values=scale_params)
                    node.inputs = [var, roi, scales_updated]
                    logging.debug(f"Updating resize node {node.name} inputs from "
                                f"sizes to scale {scale_params}")
            elif len(node.inputs) == 3:
                var, roi, scales = node.inputs[0], node.inputs[1], node.inputs[2]
                #roi initializer
                if roi.shape is None:
                    roi_const = np.array([], dtype=np.float32)
                    initializer_name = f"{node.name}.roi_const_name"
                    roi_const_name = numpy_helper.from_array(roi_const, name=initializer_name)
                    onnx_graph.initializer.append(roi_const_name)
                    roi = gs.Constant(name=initializer_name, values=roi_const) 
                node.inputs = [var, roi, scales]
            # endif
        # endif
    # endfor

def tidl_convert_resize_params_size_to_scale_dynamic_batch(graph: gs.Graph,
                                             onnx_graph: onnx.GraphProto) :
    '''
    Commonly see pattern from training export where a resize uses static dimensions for all but the batch
    This shows as a dual path from the previous node
        1) directly to Resize -- this is the actual tensor
        2) Shape->Slice->Concat -- this isolates the shape, and is variable. 

        Node_A
         |2   \ 1
       Shape   \ 
         |     |
       Slice   |
         |     |
       Concat  |
         |    /
         Resize
            |

    We want to parse path 2 and eliminate it. It is strictly unnecessary for static batch sizes, which TIDL requires anyway
    '''
    
    resize_nodes = [node for node in graph.nodes if node.op == "Resize"]


    num_resizes = len(resize_nodes)
    i=0
    while i < num_resizes:

        resize_n  = resize_nodes[i]
        logging.debug(f'Analyzing Resize node {resize_n.name} for dynamic sizes')

        if len(resize_n.inputs) <= 3 : 
            ## if len is 3, then it's using 'scales' already.
            ## if less than 3, it is not a valid Resize node, and we'll skip anyway
            i += 1; continue;


        input_shape = resize_n.inputs[0].shape
        expected_output_shape = np.zeros((4), dtype=np.int64)

        input_nodes = find_in_layers(resize_n)
        original_input_node = input_nodes[0]

        ## we will isolate the 'sizes path' comng out of the node, starting from concat and working our way up
        sizes_path_node_concat = None
        for n in input_nodes:
            if n.op == 'Concat' and n.outputs[0] == resize_n.inputs[3]:
                sizes_path_node_concat = n

        if sizes_path_node_concat is None: 
            #didn't find a concat node, so skip this one..
            logging.debug(f'Could not find concat node preceding {resize_n.name} in sizes path; skipping')
            i += 1; continue;

        concat_inputs = sizes_path_node_concat.inputs
        
        HW_values  = extract_constant_values(concat_inputs[1], graph)

        if HW_values is None or len(HW_values) != 2: 
            logging.warning(F'Did not find constant values in the concat node, and the path is more complex to parse than expected. Ignore this node {resize_n.name}')
            i += 1; continue;

        logging.debug(f'Found height and width values for the resize tensor output: {HW_values}')
        expected_output_shape[-2:] = HW_values

        input_nodes = find_in_layers(sizes_path_node_concat)

        sizes_path_node_slice = input_nodes[0]
        slice_inputs = sizes_path_node_slice.inputs

        if sizes_path_node_slice.op != 'Slice': 
            logging.warning('Could not find slice node in sizes path; skipping')
            i += 1; continue;

        slice_start = extract_constant_values(slice_inputs[1], graph)
        slice_end = extract_constant_values(slice_inputs[2], graph)

        input_nodes = find_in_layers(sizes_path_node_slice)
        sizes_path_node_shape = input_nodes[0]
        if sizes_path_node_shape.op != 'Shape': 
            logging.warning('Could not find Shape node in sizes path; skipping')
            i += 1; continue;
        ## If we reach here, the whole dynamic shapes path has been found

        input_nodes = find_in_layers(sizes_path_node_shape)
        original_node_check = input_nodes[0]

        if original_node_check != original_input_node:
            logging.warning(f"Failure to match source of Shape node to original input to resize {original_input_node.name} vs. {original_node_check.name}")
            i += 1; continue;

        ## extract batch and channel info
        NC_values = input_shape[slice_start[0]:slice_end[0]]
        expected_output_shape[0:2] = NC_values

        scales = expected_output_shape / input_shape
        scales = scales.astype(np.float32)

        scales_tensor = gs.Constant(name= f"{resize_n.name}/scales",
                                         values=scales)
        resize_n.inputs[2] = scales_tensor
        resize_n.inputs.pop(3) # remove 'sizes'

        ## remove nodes along dynamic resize path
        sizes_path_node_concat.outputs.clear()
        sizes_path_node_shape.outputs.clear()
        sizes_path_node_slice.outputs.clear()
        graph.cleanup().toposort()

        ## Unfortunately, graph has to be modified here. 
        ## We are replacing dynamic shapes with static, so shape_inference must rerun
        ## This require the onnx_graph be recreated, shape_inference, and gs.Graph be regenerated 
        temp_model = gs.export_onnx(graph)
        temp_model = onnx.shape_inference.infer_shapes(temp_model, check_type= True, strict_mode= True)
        onnx_graph = temp_model.graph
        graph = gs.import_onnx(temp_model)
        resize_nodes = [node for node in graph.nodes if node.op == "Resize"]

        assert(len(resize_nodes) == num_resizes) #check that nothing was removed, and we can keep looping according to num_resizes

        logging.debug("Successfully found and replaced sizes dynamic path with static scales for node %d" % i)
        i += 1
        # break
    return graph

def tidl_remove_unity_resize(graph: gs.Graph,
                            onnx_graph: onnx.GraphProto):
    '''
    Some models have an effectively null resize node that scales by a factor of 1 in all dimensions
    Such a node is often an export artifact -- a layer added by a model format converter
    This is node effectively unity, but it will be processed nonetheless. It should therefore be removed

    This optimization rule assumes that 'scales' are used to dictate output dimensions, as opposed to 'sizes'
    '''

    tensors = graph.tensors()
    nodes_to_remove = []
    for node in graph.nodes:

        if node.op == "Resize":
            inputs = node.inputs
            if len(inputs) >= 3:
                X, roi, scales = inputs[0:3]
            else: 
                continue
            Y = node.outputs[0]
            attrs = node.attrs

            if X.shape == Y.shape and all(map(lambda x: x==1, scales.values)):
                #ensure it's not using ROI, which is only with crop-and-resize mode
                if node.attrs['coordinate_transformation_mode'] == 'tf_crop_and_resize':
                    logging.warning("Detected Resize node as using ROI... skipping")
                    continue

                logging.debug("Removing unity Resize node %s" % node.name)
            
                out_nodes = find_out_layers(node)

                for o_node in out_nodes:
                    for i, net in enumerate(o_node.inputs):
                        if net == Y:
                            o_node.inputs[i] = X

                #node will be removed by cleanup since it has only unused outputs
    #

