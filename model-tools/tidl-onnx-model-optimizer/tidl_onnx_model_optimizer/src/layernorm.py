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
Module containing Layer normalization layer specific functions and optimizations
"""
import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np


def tidl_expand_layernorm_to_component_ops(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    The LayerNormalization-17 from ONNX is not supported by TIDL. We can expand
    this layer to it's fundamental operators to make it supported in TIDL
    """
    tensors = graph.tensors()
    layernorms = [node for node in graph.nodes if node.op == "LayerNormalization"]

    for node in layernorms:
        # extract attibutes
        axis = node.attrs['axis']
        epsilon = node.attrs['epsilon']

        # extract scale and bias
        scale = node.inputs[1]
        scale_tensor = np.array(tensors[scale.name].values, dtype= np.float32)
        bias = None
        bias_tensor = None
        if len(node.inputs) > 2:
            bias = node.inputs[2]
            bias_tensor = np.array(tensors[bias.name].values, dtype= np.float32)
        else:
            bias_tensor = np.zeros((node.outputs[0].shape[-1], ), dtype= np.float32)


        # add component ops
        logging.debug(f"Adding component nodes for LayerNormalization {node.name}")
        # reducemean 1
        reducemean1_op = gs.Variable(name= f"{node.name}_ReduceMean1_out",
                                     dtype= np.float32)
        reducemean1 = gs.Node(name= f"{node.name}_ReduceMean1", op= "ReduceMean",
                              attrs= {"axes": np.array([axis], dtype= np.int64)},
                              inputs= [node.inputs[0]], outputs= [reducemean1_op])

        logging.debug(f"Adding Node {reducemean1.name}")
        graph.nodes.append(reducemean1)

        # sub
        sub_op = gs.Variable(name= f"{node.name}_Sub_out", dtype= np.float32)
        sub = gs.Node(name= f"{node.name}_Sub", op= "Sub",
                      inputs= [node.inputs[0], reducemean1_op], outputs= [sub_op])
        logging.debug(f"Adding Node {sub.name}")
        graph.nodes.append(sub)

        # pow
        pow_op = gs.Variable(name= f"{node.name}_Pow_out", dtype= np.float32)
        pow_exp = gs.Constant(name= f"{node.name}_Pow_exponent",
                              values= np.array(2, dtype= np.float32))
        pow_2 = gs.Node(name= f"{node.name}_Pow", op= "Pow",
                      inputs= [sub_op, pow_exp], outputs= [pow_op])
        logging.debug(f"Adding Node {pow_2.name}")
        graph.nodes.append(pow_2)

        # reducemean 2
        reducemean2_op = gs.Variable(name= f"{node.name}_ReduceMean2_out",
                                     dtype= np.float32)
        reducemean2 = gs.Node(name= f"{node.name}_ReduceMean2", op= "ReduceMean",
                              attrs= {"axes": np.array([axis], dtype= np.int64)},
                              inputs= [pow_op], outputs= [reducemean2_op])
        logging.debug(f"Adding Node {reducemean2_op.name}")
        graph.nodes.append(reducemean2)

        # add
        add_op = gs.Variable(name= f"{node.name}_Add_out", dtype= np.float32)
        add_const = gs.Constant(name= f"{node.name}_Add_const",
                                values= np.array(epsilon, dtype= np.float32))
        add = gs.Node(name= f"{node.name}_Add", op= "Add",
                      inputs= [reducemean2_op, add_const], outputs= [add_op])

        logging.debug(f"Adding Node {add.name}")
        graph.nodes.append(add)

        # sqrt
        sqrt_op = gs.Variable(name= f"{node.name}_Sqrt_out", dtype= np.float32)
        sqrt = gs.Node(name= f"{node.name}_Sqrt", op= 'Sqrt', inputs= [add_op],
                       outputs= [sqrt_op])
        logging.debug(f"Adding Node {sqrt.name}")
        graph.nodes.append(sqrt)

        # div
        div_op = gs.Variable(name= f"{node.name}_Div_out", dtype= np.float32)
        div = gs.Node(name= f"{node.name}_Div", op= "Div",
                      inputs= [sub_op, sqrt_op], outputs= [div_op])
        logging.debug(f"Adding Node {div.name}")
        graph.nodes.append(div)

        # mul with scale
        scaled_op = gs.Variable(name= f"{node.name}_Scaled_out", dtype= np.float32)
        scale_const = gs.Constant(name= f"{node.name}_Scale", values= scale_tensor)
        mul = gs.Node(name= f"{node.name}_Scale_Mul", op= "Mul",
                      inputs= [div_op, scale_const], outputs= [scaled_op])
        logging.debug(f"Adding Node {mul.name}")
        graph.nodes.append(mul)

        # add with bias
        bias_const = gs.Constant(name= f"{node.name}_Bias", values= bias_tensor)
        bias_add = gs.Node(name= f"{node.name}_Bias_Add", op= "Add",
                      inputs= [scaled_op, bias_const], outputs= node.outputs)
        logging.debug(f"Adding Node {bias_add.name}")
        graph.nodes.append(bias_add)

        # remove original node
        node.outputs.clear()
