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
Module containing Slice layer specific functions and optimizations
"""
import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np


def tidl_expand_slice_across_multiple_axis (graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Convert the Slice across multiple axis to multiple slices in series
    """
    nodes = graph.nodes
    node_iter = 0
    
    for node in nodes:
        if (node.op == "Slice") and isinstance(node.inputs[1], gs.Constant) and (node.inputs[1].values.size > 1): 
            # slice is across multiple axis
            node_name = node.inputs[1].name if node.name=='' else node.name 
            
            len_inputs = len(node.inputs)
            if len_inputs<4: # all_axes isn't defined
                logging.warning(f"All Axes isn't defined in a multi-axis slice node : {node_name}")
                continue
            
            all_starts = node.inputs[1].values
            len_axes = len(all_starts)
            all_ends = node.inputs[2].values
            all_axes = node.inputs[3].values
            if len_inputs > 4: # steps are provided
                all_steps = node.inputs[4].values
            else:
                all_steps = np.ones(len_axes, dtype=np.int64)
   
            prev_slice_node = None
            for i in range(len_axes):
                curr_start = gs.Constant(name= node_name + "_start_" + str(node_iter) + "_" + str(i), values=np.array([all_starts[i]]))
                curr_end = gs.Constant(name= node_name + "_end_" + str(node_iter) + "_" + str(i), values=np.array([all_ends[i]]))
                curr_axis = gs.Constant(name= node_name + "_axis_" + str(node_iter) + "_" + str(i), values=np.array([all_axes[i]]))
                curr_step = gs.Constant(name= node_name + "_step_" + str(node_iter) + "_" + str(i), values=np.array([all_steps[i]]))
                interim_output = gs.Variable(name= node_name + "_out_" + str(node_iter) + "_" + str(i), dtype=np.float32)
                
                node_output = node.outputs[0] if i==(len_axes-1) else interim_output
                node_input = node.inputs[0] if i==0 else prev_slice_node.outputs[0]

                slice_node = gs.Node(name= node_name + "_" + str(node_iter) + "_" + str(all_axes[i]), op= "Slice",
                        inputs= [node_input, curr_start, curr_end, curr_axis, curr_step], outputs = [node_output])
                
                logging.debug(f"Adding Node {slice_node.name}")
                graph.nodes.append(slice_node)
                prev_slice_node = slice_node 
                
            node.outputs.clear()
            node_iter += 1