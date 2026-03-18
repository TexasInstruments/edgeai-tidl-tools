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
Module containing Gemm layer specific functions and optimizations
"""
import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np


def tidl_convert_gemm_to_matmul_and_add (graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Convert Gemm layer with constant B input to Matmul and
    Gemm bias (if exists) to a following add layer
    """
    def add_transpose_for(tensor):
        if isinstance(tensor, gs.Constant):
            perm = list (range(len(tensor.shape)))
            perm[-2],perm[-1] = perm[-1],perm[-2]
            tensor.values = np.transpose(tensor.values,(perm))
            return tensor
        if isinstance(tensor, gs.Variable):
            inp_node = tensor.inputs[0]
            if inp_node.op == 'Constant':
                value = inp_node.attrs.get('value', None)
                if value: 
                    return gs.Constant(name = tensor.name, values = np.transpose(value,(-2,-1))) 
            shape = tensor.shape
            shape[-2:] = shape[-2:][::-1]
            perm = list(range(len(shape)))
            perm[-2:]= perm[-2:][::-1]
            t_out = gs.Variable(name = f"{tensor.name}_t_out",dtype=tensor.dtype,shape=shape)
            if t_out.name in graph.tensors():
                return graph.tensors()[t_out.name]
            trans_node = gs.Node('Transpose',name=f"{tensor.name}_t",inputs=[tensor],outputs=[t_out], attrs=dict(perm=perm))
            graph.nodes.append(trans_node)
            logging.debug(f"Added transpose for {tensor.name}")
            return t_out
        logging.debug(f"Skipping transpose for {tensor.name} as it is of unsupported type ({tensor.__class__.__name__})")
        return tensor
        
    gemm_nodes = [node for node in graph.nodes if node.op == 'Gemm']
    for node in gemm_nodes:
        A,B = node.inputs[0:2]
        C = node.inputs[2] if len(node.inputs)==3 else None
        
        alpha = node.attrs.get('alpha',1.0)
        beta = node.attrs.get('beta',1.0)
        
        transA = node.attrs.get('transA',0)
        transB = node.attrs.get('transB',0)
        
        failed = False
        if transA:
            if A.shape and len(A.shape)>=2:
                A = add_transpose_for(A)
            else:
                failed = True
        
        if transB:
            if B.shape and len(B.shape)>=2:
                B = add_transpose_for(B)
            else:
                failed =True
        if failed :
            continue
        
        if alpha == 0:
            logging.critical(f"alpha == 0 not supported for changing at node {node.name}")
            continue
        
        if alpha != 1:
            if isinstance(A, gs.Constant):
                A.values = A.values * alpha
            elif isinstance(B, gs.Constant):
                B.values = B.values * alpha
            else:
                mul_out = gs.Variable(name = f"{node.name}_mul_out",dtype=A.dtype,shape=A.shape)
                alpha_out = gs.Constant(name = f"{node.name}_alpha",values = alpha)
                mul_node = gs.Node('Mul',name=f"{node.name}_mul",inputs=[A,alpha_out],outputs=[mul_out],)
                A = mul_out
                graph.nodes.append(mul_node)
                logging.debug(f"Added mul node {mul_node.name} for node {node.name} and alpha {alpha}")
        shape = list(A.shape[:-1])+list(B.shape[-1:]) if A.shape and B.shape else None
        matmul_out = gs.Variable(name = f"{node.name}_matmul_out",dtype=A.dtype,shape=shape) if beta!=0 and C is not None else node.outputs[0]
        matmul_node = gs.Node('MatMul',name=f"{node.name}_matmul",inputs=[A,B],outputs=[matmul_out],)
        graph.nodes.append(matmul_node)
        logging.debug(f"Added matmul node {matmul_node.name} for node {node.name}")
        
        if C is not None and beta != 0 :
            if beta != 1:
                if isinstance(C, gs.Constant):
                    C.values = C.values * beta
                if isinstance(C, gs.Variable):
                    mul_out = gs.Variable(name = f"{node.name}_mul_out",dtype=C.dtype,shape=C.shape)
                    beta_out = gs.Constant(name = f"{node.name}_beta",values = beta)
                    mul_node = gs.Node('Mul',name=f"{node.name}_mul",inputs=[C,beta_out],outputs=[mul_out],)
                    graph.nodes.append(mul_node)
                    logging.debug(f"Added mul node {mul_node.name} for node {node.name} and beta {beta}")
                    C = mul_out
            add_node = gs.Node('Add',name=f"{node.name}_add",inputs=[matmul_out,C],outputs=[node.outputs[0]],)
            graph.nodes.append(add_node)
            logging.debug(f"Added Add node {add_node.name} for node {node.name}")
        node.outputs.clear()
        graph.nodes.remove(node)
        