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


def tidl_modify_reducemean(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Wrapper function to modify ReduceMean layers to satisfy TIDL constraints
    """
    tidl_convert_reducemean(graph)


def tidl_convert_reducemean(graph: gs.Graph):
    """
    The ReduceMean layer is replaced with the cascaded multiple layers, e.g.,
    "Reshape + MatMul + Reshape". Assume that 
    1. The number of dimes of the input tensor to ReduceMean is 4 (B, C, H, W)
    2. The attribute, "axes" of ReduceMean should be [2], [3] or [2, 3]    
    """
    reduce_means = [node for node in graph.nodes if node.op == "ReduceMean"]

    for idx, reduce_mean in enumerate(reduce_means):

        input_tensor = reduce_mean.inputs
        input_shape = input_tensor[0].shape

        axes     = reduce_mean.attrs['axes']
        keepdims = reduce_mean.attrs['keepdims']

        assert len(input_shape) == 4, "The input tensor to ReduceMean should be a 4D tensor"
        assert len(axes) <= 2, "The length of Attribute axes should be 1 or 2"
        assert axes[0] == 2 or axes[0] == 3, "Attribute axes should be 2 or 3"

        B, C, H, W = input_shape

        if len(axes) == 1:

            if axes[0] == 2:
                # 1. Transpose
                var_outshape   = [gs.Variable(f"rm_transpose_out.{idx}",
                                              dtype=np.float32, shape=(B, C, W, H))]
                transpose1 = gs.Node(op="Transpose", name=f"rm_transpose.{idx}.1",
                                     attrs={"perm": [0,1,3,2]}, inputs=input_tensor,
                                     outputs=var_outshape)
                graph.nodes.append(transpose1)

                # reset B, C, H, W
                B, C, H, W = transpose1.outputs[0].shape

                # 2. MatMul
                const_dim = W
                values  = np.ones(shape=(const_dim, 1), dtype=np.float32) / const_dim
                const_inmatmul = gs.Constant(f"in_rm_matmul.{idx}", values=values)
                var_outmatmul  = [gs.Variable(f"out_rm_matmul.{idx}", 
                                              dtype=np.float32, shape=(B, C, H, 1))]

                matmul = gs.Node(op="MatMul", name=f"rm_matmul.{idx}",
                                 inputs=[var_outshape[0], const_inmatmul],
                                 outputs=var_outmatmul)
                graph.nodes.append(matmul)

                # 3. Transpose or Reshape
                if keepdims == 1:
                    transpose2 = gs.Node(op="Transpose", name=f"rm_transpose.{idx}.2",
                                         attrs={"perm": [0,1,3,2]}, inputs=var_outmatmul,
                                         outputs=reduce_mean.outputs)
                    graph.nodes.append(transpose2)
                else:
                    squeeze = gs.Node(op="Squeeze", name=f"rm_squeeze.{idx}",
                                      attrs={"axes": 3}, inputs=var_outmatmul,
                                      outputs=reduce_mean.outputs)
                    graph.nodes.append(squeeze)
            elif axes[0] == 3:
                # 1. MatMul
                const_dim = W
                values  = np.ones(shape=(const_dim, 1), dtype=np.float32) / const_dim
                const_inmatmul = gs.Constant(f"in_rm_matmul.{idx}", values=values)

                if keepdims == 1:
                    var_outmatmul = reduce_mean.outputs
                else:
                    var_outmatmul  = [gs.Variable(f"out_rm_matmul.{idx}",
                                                  dtype=np.float32, shape=(B, C, H, 1))]

                matmul = gs.Node(op="MatMul", name=f"rm_matmul.{idx}",
                                 inputs=[input_tensor[0], const_inmatmul], outputs=var_outmatmul)
                graph.nodes.append(matmul)

                # 2. Reshape
                if keepdims == 0:
                    squeeze = gs.Node(op="Squeeze", name=f"rm_squeeze.{idx}",
                                      attrs={"axes": 3}, inputs=var_outmatmul,
                                      outputs=reduce_mean.outputs)
                    print(squeeze)
                    graph.nodes.append(squeeze)

        elif len(axes) == 2:
            # 1. Reshape node
            newshape       = np.array([B, 1, C, H*W], dtype=np.int64)
            const_newshape = gs.Constant(f"rm_reshape_shape.{idx}.1", values=newshape)
            var_outshape   = [gs.Variable(f"rm_reshape_out.{idx}",
                                          dtype=np.float32, shape=(B, 1, C, H*W))]

            reshape1 = gs.Node(op="Reshape", name=f"rm_reshape.{idx}.1",
                               inputs=[input_tensor[0], const_newshape] , outputs=var_outshape)
            graph.nodes.append(reshape1)

            # 2. MatMul
            const_dim      = H*W
            values         = np.ones(shape=(const_dim, 1), dtype=np.float32) / const_dim
            const_inmatmul = gs.Constant(f"in_rm_matmul.{idx}", values=values)
            var_outmatmul  = [gs.Variable(f"out_rm_matmul.{idx}",
                                          dtype=np.float32, shape=(B, 1, C, 1))]

            matmul = gs.Node(op="MatMul", name=f"rm_matmul.{idx}",
                             inputs=[var_outshape[0], const_inmatmul], outputs=var_outmatmul)
            graph.nodes.append(matmul)

            # 3. Reshape
            if keepdims == 1:
                newshape = np.array([B, C, 1, 1], dtype=np.int64)
            else:
                newshape = np.array([B, C], dtype=np.int64)
            const_newshape = gs.Constant(f"rm_reshape_shape.{idx}.2", values=newshape)

            reshape2 = gs.Node(op="Reshape", name=f"rm_reshape.{idx}.2",
                               inputs=[var_outmatmul[0], const_newshape],
                               outputs=reduce_mean.outputs)
            graph.nodes.append(reshape2)

        # remove ReduceMean node by clearing its outputs
        reduce_mean.outputs.clear()
