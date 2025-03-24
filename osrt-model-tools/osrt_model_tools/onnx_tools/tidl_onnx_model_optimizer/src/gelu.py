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
Module containing gelu layer specific functions and optimizations
"""
import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np
from .common import find_in_layer, find_out_layer

def tidl_convert_tanhgelu_to_erfgelu(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Convert the tanh gelu 0.5*(x*(1 + tanh(0.7978845834732056*(x + 0.044714998453855515*x^3)))) to a gelu layer
    """
    nodes = graph.nodes
    
    for node in nodes:
        if (node.op == "Tanh"):
            # checking whether tanh gelu structure is satisfied
            if len(node.inputs) != 1:
                continue
            tanh_node = node
            add_node = find_in_layer(find_in_layer(tanh_node, 0), 0)
            if add_node.op != "Add":
                logging.debug(f"tanh node {node.name} not a part of gelu, skipping the conversion")
                continue
            mul_node_1 = find_in_layer(add_node, 0)
            mul_node_2 = find_in_layer(add_node, 1)

            if not (isinstance(mul_node_1, gs.Node) and isinstance(mul_node_2, gs.Node)):
                logging.debug(f"Either {mul_node_1.name} or {mul_node_2.name} is not a node, skipping the conversion")
                continue

            const_mul_node = mul_node_1
            inp_gelu_branch_node = mul_node_2
            if mul_node_1.op != "Mul":
                inp_gelu_branch_node = mul_node_1
                const_mul_node = mul_node_2
                if mul_node_2.op != "Mul":
                    logging.debug(f"Neither input of add node {add_node.name} is a mul node , skipping the conversion")
                    continue

            if find_in_layer(const_mul_node, 0).op != "Mul":
               logging.debug(f"Input of mul node {const_mul_node.name} is not a mul or pow node , skipping the conversion")
               continue 

            const_add_node = find_out_layer(tanh_node, 0)
            for inp in const_add_node.inputs:
                if isinstance(inp, gs.Constant):
                    if not np.isclose(inp.values, 1):
                        logging.debug(f"Node {inp.name} does not have value of 1, skipping the conversion")
                    #
                elif not isinstance(inp, gs.Variable):
                    logging.debug(f"Node {inp.name} is not a variable, skipping the conversion")
                #
            #

            logging.debug(f"The gelu has been identified for the layer starting from {inp_gelu_branch_node.name}.")

            inpt_erf_node = gs.Variable(name = tanh_node.name + "_erf_inpt", dtype= np.float32)
            erf_node = gs.Node(name = tanh_node.name + "_erf", op = "Erf",
                               inputs = [inpt_erf_node], outputs = [tanh_node.outputs[0]])
            logging.debug(f"Adding Node {erf_node.name}")
            graph.nodes.append(erf_node)
            tanh_node.outputs.clear()

            div_const_inpt = gs.Constant(name = tanh_node.name + "_div_inpt", values = np.array(1.4142135381698608, dtype=np.float32))
            div_node = gs.Node(name = tanh_node.name + "_div", op = "Div",
                               inputs = [inp_gelu_branch_node.outputs[0] ,div_const_inpt], outputs = [inpt_erf_node])
            logging.debug(f"Adding Node {div_node.name}")
            graph.nodes.append(div_node)

            logging.debug("Removing the following nodes fromn the graph: ")    
            logging.debug(find_in_layer(tanh_node, 0).name)
            find_in_layer(tanh_node, 0).outputs.clear() # constant mul node 2
            logging.debug(const_mul_node.name)
            const_mul_node.outputs.clear()
            logging.debug(add_node.name)
            add_node.outputs.clear()
            logging.debug(find_in_layer(find_in_layer(const_mul_node, 0), 1).name)
            find_in_layer(find_in_layer(const_mul_node, 0), 1).outputs.clear() # x^2 mul node
            logging.debug(find_in_layer(const_mul_node, 0).name)
            find_in_layer(const_mul_node, 0).outputs.clear() # x^3 mul node

            logging.info(f"The gelu conversion for layer starting from {inp_gelu_branch_node.name} would induce slight error in \
                         output bit-wise comparision, however, the accuracy should not be impacted.")




        