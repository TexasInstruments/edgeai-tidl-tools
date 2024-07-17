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
Getting the layer names for all the layers that are present between the specified start and end node of onnx graph
"""

import os
import sys
import logging
import onnx_graphsurgeon as gs
import onnx

from tidl_onnx_model_optimizer.src.common import format_logger, find_out_layers, is_end_node


def get_all_nodes(node, end_nodes, deny_list_nodes):
    # recursive function to find all the deny list nodes
    if node.name in end_nodes:
        add_in_list = True
        if node.name not in deny_list_nodes:
            deny_list_nodes.append(node.name)   
            logging.debug(f"Adding {node.name} to deny list.")
        return add_in_list, deny_list_nodes

    elif is_end_node(node):
        add_in_list = False
        return add_in_list, deny_list_nodes

    node_outputs = find_out_layers(node)
    add_in_list = False
    for n_id in node_outputs:
        add_in_list_here, deny_list_nodes = get_all_nodes(n_id, end_nodes, deny_list_nodes)
        # to add the intermediate nodes if one node has a branch which need not be included in deny list
        add_in_list = add_in_list or add_in_list_here 
        if add_in_list and add_in_list_here and (n_id.name not in deny_list_nodes):
            deny_list_nodes.append(n_id.name)
            logging.debug(f"Adding {n_id.name} to deny list.")
            
    return add_in_list, deny_list_nodes


def get_all_node_names (model_path, start_end_layers={}, verbose=False, **kwargs):
    """
    Main function
    ---------------------------------------------------------
    Inputs
    ---------------------------------------------------------
    model_path:             path to input ONNX model
    start_end_layers:       dictionary of the start and end layers, between which (including start 
                            and end node) needs to be added to deny list
                            if "None" is passed in the end node (values of dict), then the model output nodes
                            are assumed as the end nodes
     ---------------------------------------------------------------
    Output
    ---------------------------------------------------------------
    nodes:                  comma separated string of all the nodes that need to be added in the deny list
    """
    args = {}
    args['log_level'] = "debug" if verbose else "info"
    # format logger
    format_logger(args['log_level'])
    
     # check for valid path
    if not os.path.isfile(model_path):
        logging.error(f"File {model_path} not found")
        sys.exit(-1)
    
    model = onnx.load(model_path)

    graph = gs.import_onnx(model)
    model_outputs = [node.inputs[0].name for node in graph.outputs]
    
    deny_list_nodes = []
    for node in graph.nodes:
        if node.name in start_end_layers.keys():
            end_layers = start_end_layers[node.name]
            if end_layers is None:
                end_layers = model_outputs
            _, deny_list_nodes = get_all_nodes(node, end_layers, deny_list_nodes)
            deny_list_nodes.append(node.name)
            logging.debug(f"Adding {node.name} to deny list.")
    
    comma_separated_deny_nodes = ', '.join(deny_list_nodes)
    logging.info(f"Deny list for the specification is {comma_separated_deny_nodes}, with {len(deny_list_nodes)} nodes.")
    
    return comma_separated_deny_nodes

