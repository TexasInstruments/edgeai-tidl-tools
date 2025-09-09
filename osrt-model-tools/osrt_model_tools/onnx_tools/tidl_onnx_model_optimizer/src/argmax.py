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
import onnx

def tidl_change_argmax_keepdims_to_1(graph:gs.Graph,onnx_graph: onnx.GraphProto):
    '''
    Changes the keepdims parameter from 0 to 1 and axis to -3 along with appropiate shape adjustments.
    '''
    nodes = [node for node in graph.nodes if node.op == 'ArgMax']
    for i, node in enumerate(nodes):
        keep_dims = node.attrs['keepdims'] 
        axis = node.attrs['axis']
        inp = node.inputs[0]
        inp_dim = len(inp.shape)

        if axis == max(0,inp_dim-3):
            axis = node.attrs['axis'] = -3
        if keep_dims == 1 and axis == -3:
            continue
        node1 = None
        shape = node.inputs[0].shape
        old_axis = axis
        if ((0>axis>-3) or ((inp_dim-2)<=axis<inp_dim) and axis>=0) and (inp_dim+(-axis if axis<=0 else (inp_dim-axis)))<=6:
            num1s = 3+axis if axis<0 else 3-(inp_dim-axis)
            new_shape = list(shape.copy())
            new_shape.extend( [1 for _ in range(num1s)])
            if len(inp.inputs) and (inp_node := inp.inputs[0]).op == 'Reshape':
                shape1 = inp_node.inputs[1].copy()
                shape1.name+=f'_{i}'
                shape1.values = np.array(new_shape.copy())
                inp.shape = new_shape
                inp_node.inputs[1] = shape1
                node1 = gs.Node(op="Reshape", inputs=[], outputs=[])
            else:
                node1_out = gs.Variable(f'{node.name}_rehsape1_out', inp.dtype, new_shape)
                new_shape = gs.Constant(name=f"{node.name}_shape1", values=np.array(new_shape))
                node1 = gs.Node(op="Reshape", inputs=[inp, new_shape], outputs=[node1_out])
                shape = new_shape
                graph.nodes.append(node1)
                logging.debug(f"Adding Node {node1.name}")
            axis = node.attrs['axis'] = -3
        elif axis != -3:
            perm = list(range(inp_dim))
            perm[axis], perm[-3] = perm[-3], perm[axis]
            new_shape = list(shape.copy())
            new_shape[axis], new_shape[-3] = new_shape[-3], new_shape[axis]
            node1_out = gs.Variable(f'{node.name}_transpose1_out', inp.dtype, new_shape)
            node1 = gs.Node(op="Transpose", inputs=[inp], outputs=[node1_out])
            node1.attrs['perm'] = perm
            graph.nodes.append(node1)
            logging.debug(f"Adding Node {node1.name}")
            axis = node.attrs['axis'] = -3
        node.inputs[0] = node1_out if (node1 is not None) and (len(node1.inputs)) else inp
        out = node.outputs[0]
        old_shape = out.shape
        if keep_dims != 1 or node1 is not None:
            new_shape = list(node.inputs[0].shape)
            new_shape[axis] = 1
            node.attrs['keepdims'] =1
        
        if (node1 is not None and node1.op=='Transpose'):
            node2_in = gs.Variable(f'{node.name}_transpose2_in', out.dtype, new_shape.copy())
            new_shape[old_axis], new_shape[-3] = new_shape[-3], new_shape[old_axis]
            node2_out = node.outputs[0] if keep_dims == 1 else gs.Variable(f'{node.name}_transpose2_out', out.dtype, new_shape)
            node2 = gs.Node(op="Transpose", inputs=[node2_in], outputs=[node2_out])
            node2.attrs['perm'] = perm
            graph.nodes.append(node2)
            logging.debug(f"Adding Node {node2.name}")
            if keep_dims != 1:
                final_shape = gs.Constant(name=f"{node.name}_shape", values=np.array(old_shape))
                reshape_node = gs.Node(op="Reshape", inputs=[node2_out, final_shape], outputs=[node.outputs[0]])
                graph.nodes.append(reshape_node)
                logging.debug(f"Adding Node {reshape_node.name}")
            node.outputs[0] = node2_in
        elif (node1 is not None and node1.op == 'Reshape') or (keep_dims!=1):
            reshape_in = gs.Variable(f'{node.name}_reshape_in', out.dtype, new_shape)
            final_shape = gs.Constant(name=f"{node.name}_shape", values=np.array(old_shape))
            reshape_node = gs.Node(op="Reshape", inputs=[reshape_in, final_shape], outputs=[node.outputs[0]])
            graph.nodes.append(reshape_node)
            node.outputs[0] = reshape_in            
            logging.debug(f"Adding Node {reshape_node.name}")