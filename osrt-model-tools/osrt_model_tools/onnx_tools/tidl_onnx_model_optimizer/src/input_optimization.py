# Copyright (c) {2025 - 2026} Texas Instruments Incorporated
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

import logging
import onnx_graphsurgeon as gs
import numpy as np
import numbers
from onnx import TensorProto

def tidl_add_input_normalization(graph, onnx_graph,  input_mean=None,input_scale=None):
    if  input_mean is None and input_scale is None:
        logging.warning("Both input_mean and input_scale are provided None! Not adding input normalization!")
        return
    inputs = graph.inputs
    if input_scale is None:
        input_scale = [[0.0078125,0.0078125,0.0078125]]
        logging.info(f"Input scale not provided, defaulting to {input_scale}")
    if input_mean is None:
        input_mean = [[128.0, 128.0, 128.0]]
        logging.info(f"Input mean not provided, defaulting to {input_mean}")
    num_norms = min(len(inputs), len(input_scale))
    if isinstance(input_mean[0], numbers.Number):
        input_mean = [input_mean]
    if isinstance(input_scale[0], numbers.Number):
        input_scale = [input_scale]
    for i in range(num_norms):
        inp = inputs[i]
        scale = input_scale[i]
        mean = input_mean[i]
        mean = [x * -1 for x in mean]
        final_shape = [1]*len(inp.shape)
        for axis, dim in enumerate(inp.shape):
            if dim == len(mean):
                final_shape[axis] = -1
                break
        new_inp = gs.Variable(inp.name + "_net_in", shape=inp.shape, dtype=np.uint8)
        graph.inputs[i] = new_inp
        cast_out = gs.Variable(f'{inp.name}_cast_out', shape=inp.shape, dtype=inp.dtype)
        cast_node = gs.Node("Cast", f'{inp.name}_cast', inputs=[new_inp], outputs=[cast_out], attrs=dict(to=TensorProto.FLOAT))
        graph.nodes.append(cast_node)
        
        mean = np.array(mean, dtype=np.float32)
        mean = mean.reshape(final_shape)
        mean = gs.Constant(f'{inp.name}_bias', mean)
        scale = np.array(scale, dtype=np.float32)
        scale = scale.reshape(final_shape)
        scale = gs.Constant(f'{inp.name}_scale', scale)
        
        add_out = gs.Variable(f'{inp.name}_add_out', shape=inp.shape, dtype=inp.dtype)
        add_node = gs.Node("Add", f'{inp.name}_add', inputs=[cast_out, mean], outputs=[add_out])
        graph.nodes.append(add_node)
        
        mul_node = gs.Node("Mul", f'{inp.name}_mul', inputs=[add_out, scale], outputs=[inp])
        logging.info(f"Adding input normalization for input {inp.name}")
        graph.nodes.append(mul_node)
    
    for i, out in enumerate(graph.outputs):
        output_node = out.inputs[0]
        if output_node.op != 'ArgMax':
            continue
        cast_out = gs.Variable(f'{out.name}_cast_out', shape=out.shape, dtype=np.uint8)
        cast_node = gs.Node("Cast", f'{out.name}_cast', inputs=[out], outputs=[cast_out], attrs=dict(to=TensorProto.UINT8))
        logging.info(f"Adding Cast node for output {out.name}")
        graph.nodes.append(cast_node)
        graph.outputs[i] = cast_out
    
    graph.cleanup().toposort()

