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
Module containing reshape->transpose->reshape layer specific functions and optimizations
"""
import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np
import copy
from .common import find_out_layer

def tidl_optimize_reshp_tr_reshp(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Optimize the reshape transpose reshape layer combination such that the intermediate dimensions length do not exceed 
    beyond 6, clubbing based on the transpose perm values.
    """
    nodes = graph.nodes
    iteration = 0
    for node in nodes:
        if (node.op == "Reshape") and (find_out_layer(node, 0).op == "Transpose") and (find_out_layer(find_out_layer(node, 0), 0).op == "Reshape"):
            # found reshape transpose reshape combination
            input_shape = node.inputs[0].shape

            if input_shape is None or len(input_shape) >= 7:
                logging.info(f"Skipping optimization for {node.name} as input dimensions exceed 6 or shape unavailable")
                continue
            tr_node = find_out_layer(node, 0)
            resp_2_node = find_out_layer(find_out_layer(node, 0), 0)
            output_shape = resp_2_node.outputs[0].shape
            if output_shape is None or len(output_shape) >= 7:
                logging.info(f"Skipping optimization for {resp_2_node.name} as output dimensions exceed 6 or shape unavailable")
                continue

            tr_perms = tr_node.attrs['perm'] if 'perm' in tr_node.attrs else None
            if tr_perms is None:
                logging.info(f"tr_perms do not exist for {tr_node.name}.")
                continue

            perms_consecutive = []
            temp = [tr_perms[0]]
            for i in range(1, len(tr_perms)):
                if tr_perms[i] == tr_perms[i - 1] + 1:
                    temp.append(tr_perms[i])
                else:
                    perms_consecutive.append(temp)
                    temp = [tr_perms[i]]
 
            perms_consecutive.append(temp)
            # sort the perms_consecutive based on smallest element in each list
            old_perms_consecutive = copy.deepcopy(perms_consecutive)
            perms_consecutive.sort(key=lambda x: min(x))

            reshp_1_shape_old = node.inputs[1].values

            reshp_1_shape_new = []
            idx = 0
            for group in perms_consecutive:
                group_size = np.prod([reshp_1_shape_old[elem] for elem in group])
                reshp_1_shape_new.append(group_size)
                idx += len(group)

            while idx < len(reshp_1_shape_old):
                reshp_1_shape_new.append(reshp_1_shape_old[idx])
                idx += 1

            # few reshapes have the same name of shape which causes issue. 
            node.inputs[1] = gs.Constant(node.inputs[1].name + "rehsp_1_shape" + str(iteration) , np.array(reshp_1_shape_new))

            min_old_perms_consecutive = [min(group) for group in old_perms_consecutive]

            # Ensure min_old_perms_consecutive contains all elements from 0 to len(min_old_perms_consecutive)
            for num in range(len(min_old_perms_consecutive)):
                if num not in min_old_perms_consecutive:
                    for idx in range(len(min_old_perms_consecutive)):
                        if min_old_perms_consecutive[idx] > num:
                            min_old_perms_consecutive[idx] = num
                            break


            tr_node.attrs['perm'] = min_old_perms_consecutive

            logging.debug(f"Converted the nodes {node.name} and {tr_node.name}  to support less than 6 variable inputs.")

            iteration += 1
            node.outputs[0].shape = None
            tr_node.outputs[0].shape = None








            

