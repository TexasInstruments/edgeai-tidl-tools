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
Module containing ReduceMean layer specific functions and optimizations
"""
import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np



def tidl_convert_reducemean_to_matmul (graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    The ReduceMean layer is replaced with the cascaded multiple layers, e.g.,
    "Reshape + MatMul + Reshape". Assume that
    1. The number of dimes of the input tensor to ReduceMean is 4 (B, C, H, W)
    2. The attribute, "axes" of ReduceMean should be [2], [3] or [2, 3]
    """
    reduce_means = [node for node in graph.nodes if node.op == "ReduceMean"]

    for idx, reduce_mean in enumerate(reduce_means):
        # Check if this ReduceMean is part of a LayerNorm structure
        is_layernorm = False
        for output in reduce_mean.outputs[0].outputs:
            if output.op == "Sub" or output.op == "Add":  # LayerNorm has Sub after ReduceMean, or add incase of second one
                for sub_output in output.outputs[0].outputs:
                    if sub_output.op == "Pow" or sub_output.op == "Mul" or sub_output.op == "Sqrt":  # Followed by Pow or Mul
                        is_layernorm = True
                        break
                if is_layernorm:
                    break
        
        if is_layernorm:
            logging.info(f"Skipping ReduceMean optimization as it appears to be part of LayerNorm")
            continue

        input_tensor = reduce_mean.inputs
        input_shape  = input_tensor[0].shape

        # input tensor dim. Should be 3D or 4D
        numdims  = len(input_shape)

        # axes can either be input or attribute
        if 'axes' in reduce_mean.attrs:
            axes = reduce_mean.attrs['axes']
        elif len(input_tensor) > 1:
            axes = input_tensor[1].values
        else:
            axes = np.arange(0, numdims, 1)

        try:
            keepdims = reduce_mean.attrs['keepdims']
        except:
            logging.debug(f"keepdims for {reduce_mean.name} node does not exist. Set keepdims to 1")
            keepdims = 1

        if len(input_shape) < 2:
            logging.info(f"The input tensor to ReduceMean {reduce_mean.name} should be a 2D, 3D or 4D tensor, skipping") 
            continue

        if len(axes) > 2:
            logging.info(f"The length of Attribute axes of {reduce_mean.name} should be 1 or 2, skipping") 
            continue

        if numdims == 4:
            if not(axes[0] == 2 or axes[0] == -2 or axes[0] == 3 or axes[0] == -1):
                logging.info(f"Attribute axes for {reduce_mean.name} should be 2 or 3, skipping")
                continue
                
        elif numdims == 3:
            if not(axes[0] == 1 or axes[0] == -2 or axes[0] == 2 or axes[0] == -1):
                logging.info(f"Attribute axes for {reduce_mean.name} should be 1 or 2, skipping")
                continue

        elif numdims == 2:
            if not(axes[0] == 0 or axes[0] == -2 or axes[0] == 1 or axes[0] == -1):
                logging.info(f"Attribute axes for {reduce_mean.name} should be 0 or 1, skipping")
                continue

        dtype = reduce_mean.inputs[0].dtype if (isinstance(reduce_mean.inputs[0], gs.Variable) and hasattr(reduce_mean.inputs[0], 'dtype')) \
                    else np.float32

        if numdims == 4:
            B, C, H, W = input_shape
        elif numdims == 3:
            C, H, W = input_shape
        elif numdims == 2:
            H, W = input_shape

        if len(axes) == 1:

            if (numdims == 4 and axes[0] == 2) or \
               (numdims == 3 and axes[0] == 1) or \
               (numdims == 2 and axes[0] == 0) or axes[0] == -2:

                if numdims == 4:
                    shape_outshape  = (B, C, W, H)
                    shape_outmatmul = (B, C, W, 1)
                    permidx = [0, 1, 3, 2]
                elif numdims == 3:
                    shape_outshape  = (C, W, H)
                    shape_outmatmul = (C, W, 1)
                    permidx = [0, 2, 1]
                elif numdims == 2:
                    shape_outshape  = (W, H)
                    shape_outmatmul = (W, 1)
                    permidx = [1, 0]

                # 1. Transpose
                var_outshape   = [gs.Variable(f"rm_transpose_out.{idx}",
                                              dtype=dtype, shape=shape_outshape)]
                transpose1 = gs.Node(op="Transpose", name=f"rm_transpose.{idx}.1",
                                     attrs={"perm": permidx}, inputs=input_tensor,
                                     outputs=var_outshape)
                graph.nodes.append(transpose1)
                logging.debug(f"Adding Node {transpose1.name}")

                # 2. MatMul
                const_dim = H
                values  = np.ones(shape=(const_dim, 1), dtype=dtype) / const_dim
                const_inmatmul = gs.Constant(f"in_rm_matmul.{idx}", values=values)
                var_outmatmul  = [gs.Variable(f"out_rm_matmul.{idx}",
                                              dtype=dtype, shape=shape_outmatmul)]

                matmul = gs.Node(op="MatMul", name=f"rm_matmul.{idx}",
                                 inputs=[var_outshape[0], const_inmatmul],
                                 outputs=var_outmatmul)
                graph.nodes.append(matmul)
                logging.debug(f"Adding Node {matmul.name}")

                # 3. Transpose or Reshape
                if keepdims == 1:
                    transpose2 = gs.Node(op="Transpose", name=f"rm_transpose.{idx}.2",
                                         attrs={"perm": permidx}, inputs=var_outmatmul,
                                         outputs=reduce_mean.outputs)
                    graph.nodes.append(transpose2)
                    logging.debug(f"Adding Node {transpose2.name}")
                else:
                    if graph.opset < 13:
                        squeeze = gs.Node(op="Squeeze", name=f"rm_squeeze.{idx}",
                                        attrs={"axes": [-1]}, inputs=var_outmatmul,
                                        outputs=reduce_mean.outputs)
                    else:
                        axes = gs.Constant(f'rm_squeeze.{idx}_axes', values= np.array([-1], dtype=np.int64))
                        squeeze = gs.Node(op="Squeeze", name=f"rm_squeeze.{idx}",
                                        inputs=var_outmatmul + [axes],
                                        outputs=reduce_mean.outputs)
                    graph.nodes.append(squeeze)
                    logging.debug(f"Adding Node {squeeze.name}")

            elif (numdims == 4 and axes[0] == 3) or \
                 (numdims == 3 and axes[0] == 2) or \
                 (numdims == 2 and axes[0] == 1) or axes[0] == -1:

                if numdims == 4:
                    shape_outmatmul = (B, C, H, 1)
                elif numdims == 3:
                    shape_outmatmul = (C, H, 1)
                elif numdims == 2:
                    shape_outmatmul = (H, 1)

                # 1. MatMul
                const_dim = W
                values  = np.ones(shape=(const_dim, 1), dtype=dtype) / const_dim
                const_inmatmul = gs.Constant(f"in_rm_matmul.{idx}", values=values)

                if keepdims == 1:
                    var_outmatmul = reduce_mean.outputs
                else:
                    var_outmatmul  = [gs.Variable(f"out_rm_matmul.{idx}",
                                                  dtype=dtype, shape=shape_outmatmul)]

                matmul = gs.Node(op="MatMul", name=f"rm_matmul.{idx}",
                                 inputs=[input_tensor[0], const_inmatmul], outputs=var_outmatmul)
                graph.nodes.append(matmul)
                logging.debug(f"Adding Node {matmul.name}")

                # 2. Reshape
                if keepdims == 0:
                    if graph.opset < 13:
                        squeeze = gs.Node(op="Squeeze", name=f"rm_squeeze.{idx}",
                                        attrs={"axes": [-1]}, inputs=var_outmatmul,
                                        outputs=reduce_mean.outputs)
                    else:
                        axes = gs.Constant(f'rm_squeeze.{idx}_axes', values= np.array([-1], dtype=np.int64))
                        squeeze = gs.Node(op="Squeeze", name=f"rm_squeeze.{idx}",
                                        inputs=var_outmatmul + [axes],
                                        outputs=reduce_mean.outputs)
                    graph.nodes.append(squeeze)
                    logging.debug(f"Adding Node {squeeze.name}")

        elif len(axes) == 2:

            if numdims == 4:
                shape_outshape  = (B, 1, C, H*W)
                shape_outmatmul = (B, 1, C, 1)
                if keepdims == 1:
                    shape_output = (B, C, 1, 1)
                else:
                    shape_output = (B, C)
            elif numdims == 3:
                shape_outshape  = (1, C, H*W)
                shape_outmatmul = (1, C, 1)
                if keepdims == 1:
                    shape_output = (C, 1, 1)
                else:
                    shape_output = (C)
            elif numdims == 2:
                shape_outshape  = (1, H*W)
                shape_outmatmul = (1, 1)


            # 1. Reshape node
            newshape       = np.array(shape_outshape, dtype=np.int64)
            const_newshape = gs.Constant(f"rm_reshape_shape.{idx}.1", values=newshape)
            var_outshape   = [gs.Variable(f"rm_reshape_out.{idx}",
                                          dtype=dtype, shape=shape_outshape)]

            reshape1 = gs.Node(op="Reshape", name=f"rm_reshape.{idx}.1",
                               inputs=[input_tensor[0], const_newshape] , outputs=var_outshape)
            graph.nodes.append(reshape1)
            logging.debug(f"Adding Node {reshape1.name}")

            if numdims == 2:
                if keepdims != 1:
                    logging.info(f"Attribute keepdims should be 1 for 2D tensor for {reduce_mean.name}, skipping")
                    continue

                # 2. MatMul
                const_dim      = H*W
                values         = np.ones(shape=(const_dim, 1), dtype=dtype) / const_dim
                const_inmatmul = gs.Constant(f"in_rm_matmul.{idx}", values=values)

                var_outmatmul = reduce_mean.outputs                
                matmul = gs.Node(op="MatMul", name=f"rm_matmul.{idx}",
                                 inputs=[var_outshape[0], const_inmatmul], outputs=var_outmatmul)
                graph.nodes.append(matmul)
                logging.debug(f"Adding Node {matmul.name}")
            else:
                # 2. MatMul
                const_dim      = H*W
                values         = np.ones(shape=(const_dim, 1), dtype=dtype) / const_dim
                const_inmatmul = gs.Constant(f"in_rm_matmul.{idx}", values=values)
                var_outmatmul  = [gs.Variable(f"out_rm_matmul.{idx}",
                                              dtype=dtype, shape=shape_outmatmul)]

                matmul = gs.Node(op="MatMul", name=f"rm_matmul.{idx}",
                                 inputs=[var_outshape[0], const_inmatmul], outputs=var_outmatmul)
                graph.nodes.append(matmul)
                logging.debug(f"Adding Node {matmul.name}")

                # 3. Reshape: Output shape (newshape) depends on numdims and keepdims
                newshape = np.array(shape_output, dtype=np.int64)
                const_newshape = gs.Constant(f"rm_reshape_shape.{idx}.2", values=newshape)

                reshape2 = gs.Node(op="Reshape", name=f"rm_reshape.{idx}.2",
                                   inputs=[var_outmatmul[0], const_newshape],
                                   outputs=reduce_mean.outputs)
                graph.nodes.append(reshape2)
                logging.debug(f"Adding Node {reshape2.name}")

        # remove ReduceMean node by clearing its outputs
        reduce_mean.outputs.clear()
