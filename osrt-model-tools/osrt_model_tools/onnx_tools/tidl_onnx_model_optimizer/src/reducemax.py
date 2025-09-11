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
Module containing ReduceMax layer specific functions and optimizations
"""
import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np



def tidl_convert_reducemax_width_to_height (graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    The ReduceMax layer is replaced with the cascaded multiple layers, e.g.,
    "Transpose + ReduceMax + Transpose + Squeeze".
    """
    logging.debug("Starting tidl_convert_reducemax_width_to_height optimization")
    count = 0
    for node in graph.nodes:
        if node.op == 'ReduceMax':
            logging.debug(f"Processing ReduceMax node: {node.name}")
            
            # Input and output tensors
            input_tensor = node.inputs
            if not input_tensor:
                logging.debug(f"No input tensors found for node {node.name}, skipping")
                continue
                
            # Get shape and dtype information first to determine numdims
            dtype = input_tensor[0].dtype
            shape = input_tensor[0].shape
            if shape is None:
                logging.debug(f"Shape is None for node {node.name}, skipping")
                continue
                
            numdims = len(shape)
            logging.debug(f"Input tensor shape: {shape}, dtype: {dtype}, numdims: {numdims}")
            
            # Get attributes
            if 'axes' in node.attrs:
                axes = node.attrs['axes']
                logging.debug(f"Found axes in node attributes: {axes}")
            elif len(input_tensor) > 1:
                axes = input_tensor[1].values
                logging.debug(f"Found axes in input tensor: {axes}")
            else:
                axes = np.arange(0, numdims, 1)
                logging.debug(f"Using default axes: {axes}")

            try:
                keepdims = node.attrs['keepdims']
            except:
                logging.debug(f"keepdims for {node.name} node does not exist. Set keepdims to 1")
                keepdims = 1
            # keepdims = node.attrs.get('keepdims', 1)
            
            if axes is None:
                logging.debug(f"axes for {node.name} is none, skipping node")
                continue
                
            # Convert to list if it's not already
            if isinstance(axes, int):
                axes = [axes]
                logging.debug(f"Converted axes to list: {axes}")
            
            # Only handle specific cases where width reduction is being performed
            if not ((numdims == 4 and axes[0] == 3) or 
                   (numdims == 3 and axes[0] == 2) or
                   (numdims == 2 and axes[0] == 1) or
                   axes[0] == -1):
                logging.debug(f"Node {node.name} does not match width reduction criteria (numdims={numdims}, axes={axes}), skipping")
                continue
                
            # Define permutation for transpose
            if numdims == 4:
                # NCHW -> NCWH (swap H and W)
                permidx = [0, 1, 3, 2]
                shape_outshape = (shape[0], shape[1], shape[3], shape[2])
                shape_outreducemax = (shape[0], shape[1], shape[3], 1)
                logging.debug(f"4D case: permidx={permidx}, shape_outshape={shape_outshape}")
            elif numdims == 3:
                # CHW -> CWH
                permidx = [0, 2, 1]
                shape_outshape = (shape[0], shape[2], shape[1])
                shape_outreducemax = (shape[0], 1, shape[1])
                logging.debug(f"3D case: permidx={permidx}, shape_outshape={shape_outshape}")
            elif numdims == 2:
                # HW -> WH
                permidx = [1, 0]
                shape_outshape = (shape[1], shape[0])
                shape_outreducemax = (shape[1], 1)
                logging.debug(f"2D case: permidx={permidx}, shape_outshape={shape_outshape}")
            
            idx = count
            count += 1
            logging.debug(f"Starting transformation for node {node.name} with index {idx}")
            
            # 1. Transpose
            var_outshape = [gs.Variable(f"rm_transpose_out.{idx}",
                                      dtype=dtype, shape=shape_outshape)]
            transpose1 = gs.Node(op="Transpose", name=f"rm_transpose.{idx}.1",
                                attrs={"perm": permidx}, inputs=input_tensor,
                                outputs=var_outshape)
            graph.nodes.append(transpose1)
            logging.debug(f"Adding Node {transpose1.name}")
            
            # 2. ReduceMax (now reducing along height which was originally width)
            var_outreduce = [gs.Variable(f"rm_reducemax_out.{idx}", 
                                        dtype=dtype, shape=shape_outreducemax)]
            reduce_max_node = gs.Node(op="ReduceMax", name=f"rm_reducemax.{idx}",
                                    attrs={'axes': [-2], 'keepdims': 1},
                                    inputs=[var_outshape[0]],
                                    outputs=var_outreduce)
            graph.nodes.append(reduce_max_node)
            logging.debug(f"Adding Node {reduce_max_node.name}")
            
            # 3. Transpose back
            var_out_tr2 = [gs.Variable(f"rm_transpose_out2.{idx}", dtype=dtype)]
            transpose2 = gs.Node(op="Transpose", name=f"rm_transpose.{idx}.2",
                                attrs={"perm": permidx}, 
                                inputs=var_outreduce,
                                outputs=var_out_tr2)
            graph.nodes.append(transpose2)
            logging.debug(f"Adding Node {transpose2.name}")
            
            # 4. Squeeze if keepdims is 0
            if keepdims == 0:
                logging.debug(f"keepdims=0, adding Squeeze node for {node.name}")
                # Create a constant for the axes to squeeze
                if graph.opset < 13:
                    squeeze_node = gs.Node(op="Squeeze", name=f"rm_squeeze.{idx}",
                                          attrs={"axes": axes},
                                          inputs=[var_out_tr2[0]],
                                          outputs=node.outputs)
                    logging.debug(f"Using opset < 13, squeeze with axes attribute")
                else:
                    squeeze_axes = gs.Constant(f"squeeze_axes.{idx}", 
                                             values=np.array(axes, dtype=np.int64))
                    squeeze_node = gs.Node(op="Squeeze", name=f"rm_squeeze.{idx}",
                                          inputs=[var_out_tr2[0], squeeze_axes],
                                          outputs=node.outputs)
                    logging.debug(f"Using opset >= 13, squeeze with axes input")
                graph.nodes.append(squeeze_node)
                logging.debug(f"Adding Node {squeeze_node.name}")
            else:
                # Connect transpose output directly to original outputs
                logging.debug(f"keepdims=1, connecting transpose output directly to original outputs")
                transpose2.outputs = node.outputs
                
            # Remove the original ReduceMax node
            logging.debug(f"Removing original ReduceMax node {node.name}")
            node.outputs.clear()
    
    logging.debug(f"Completed tidl_convert_reducemax_width_to_height optimization. Processed {count} ReduceMax nodes")
