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
        if (node.op == "Slice"):
            if len(node.inputs)<2:
                # Slice-1 implementation (old opset)
                len_starts = len(node.attrs['starts'])
                if len_starts > 1:
                    logging.warning(f"Slice-1 implementation is not supported as of now. We suggest using higher opset version.")
                    continue
            elif isinstance(node.inputs[1], gs.Constant):
                len_starts = node.inputs[1].values.size
            else:
                continue
            
            if len_starts > 1:
                # slice is across multiple axis
                node_name = node.inputs[1].name if node.name=='' else node.name 
                
                len_inputs = len(node.inputs)
                if len_inputs<4: # all_axes isn't defined
                    logging.warning(f"All Axes isn't defined in a multi-axis slice node : {node_name}. Skipping the conversion")
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


def tidl_convert_2_dimension_slice_to_maxpool (graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Convert the Slice present in 2 dimensions to a maxpool layer. A slice with steps=[2,2] with axes=[2,3] can be converted 
    to maxpool with a stride of 2. There could be 2 different possibilities i.e. NCHW format
    and NHWC format. In the second part, we will have to introduce transpose -> maxpool -> transpose. 
    """
    nodes = graph.nodes
    node_iter = 0
    
    for node in nodes:
        if (node.op == "Slice") and len(node.inputs) > 3 and isinstance(node.inputs[3], gs.Constant) and (node.inputs[3].values.size == 2): 
            # slice has axes input and axes is across 2 axis
            if len(node.inputs) < 5:
                # does not have access to the steps input , continuing
                continue

            input_shape = node.inputs[0].shape
            if input_shape is None or len(input_shape) != 4:
                logging.warning(f"Shape Inference is not done, or the dimensions are not equal to 4")
                continue

            # Normalize negative axes to positive values
            axes = node.inputs[3].values.copy()
            axes = np.where(axes < 0, axes + len(input_shape), axes)
            

            if abs(axes[0] - axes[1]) == 1 and (node.inputs[4].values[0] == node.inputs[4].values[1]):
                # slice axes are consecutive and both axes has the same step
                stride = node.inputs[4].values[0]
                # stride == 2 is supported in TIDL, if greater than that, it will be broken into 2 by other optimizations
                node_name = node.name if node.name else f"slice_to_maxpool_{node_iter}"

                end_axes = sorted(axes)[-1]
                if end_axes != (len(input_shape) - 1):
                    # would require a transpose before and after the maxpool
                    transpose_output = gs.Variable(name=f"{node_name}_transpose_out")
                    maxpool_output = gs.Variable(name=f"{node_name}_maxpool_out")
                    perm1 = [0, 3, 1, 2] if end_axes==2 else [2, 3, 0, 1]
                    perm2 = [0, 2, 3, 1] if end_axes==2 else [2, 3, 0, 1]

                    transpose1_node = gs.Node(
                        name=f"{node_name}_transpose1",
                        op="Transpose",
                        attrs={"perm": perm1},
                        inputs=[node.inputs[0]],
                        outputs=[transpose_output]
                    )
                    logging.debug(f"Adding Node {transpose1_node.name}")

                    # MaxPool2D
                    maxpool_node = gs.Node(
                        name=f"{node_name}_maxpool",
                        op="MaxPool",
                        attrs={
                            "kernel_shape": [1, 1],
                            "strides": [stride, stride]
                        },
                        inputs=[transpose_output],
                        outputs=[maxpool_output]
                    )
                    logging.debug(f"Adding Node {maxpool_node.name}")
                    
                    # NCHW to NHWC transpose
                    transpose2_node = gs.Node(
                        name=f"{node_name}_transpose2",
                        op="Transpose",
                        attrs={"perm": perm2},
                        inputs=[maxpool_output],
                        outputs=[node.outputs[0]]
                    )
                    logging.debug(f"Adding Node {transpose2_node.name}")
                    
                    graph.nodes.extend([transpose1_node, maxpool_node, transpose2_node])
                #
                else:
                    maxpool_node = gs.Node(
                        name=f"{node_name}_maxpool",
                        op="MaxPool",
                        attrs={
                            "kernel_shape": [1, 1],
                            "strides": [stride, stride],
                            "pads": [0, 0, 0, 0]
                        },
                        inputs=[node.inputs[0]],
                        outputs=[node.outputs[0]]
                    )
                    logging.debug(f"Adding Node {maxpool_node.name}")
                    graph.nodes.append(maxpool_node)
                #
            #
            node.outputs.clear()
            node_iter += 1
        #
    #


def tidl_add_slice_step_axis (graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    TIDL has some bug with supporting slice when step axis is not present, adding the default value
    """
    nodes = graph.nodes
    node_iter = 0
    
    for node in nodes:
        if (node.op == "Slice") and isinstance(node.inputs[1], gs.Constant) and len(node.inputs)<5: 
            # slice is found, however, it has missing axes and/or step
            node_name = node.inputs[1].name if node.name=='' else node.name 
            len_inputs = len(node.inputs)
            node_input = node.inputs[0]
            slice_start =  node.inputs[1]
            slice_end =  node.inputs[2]
            slice_axes = node.inputs[3]
            slice_steps = gs.Constant(name= node_name + "_step_" + str(node_iter), values=np.ones(slice_start.values.shape[0], dtype=np.int64))
            
            slice_node = gs.Node(name= node_name + "_" + str(node_iter), op= "Slice",
                inputs= [node_input, slice_start, slice_end, slice_axes, slice_steps], outputs = [node.outputs[0]])
            
            logging.debug(f"Adding Node {slice_node.name}")
            graph.nodes.append(slice_node)
                
            node.outputs.clear()
            node_iter += 1