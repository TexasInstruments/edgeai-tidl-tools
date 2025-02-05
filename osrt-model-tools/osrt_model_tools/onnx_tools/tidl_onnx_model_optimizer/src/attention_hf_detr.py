# Copyright (c) {2023 - 2024} Texas Instruments Incorporated
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
"""Attention block detection and optimization"""

import logging
from typing import List
from abc import ABC, abstractmethod
import numpy as np
import onnx_graphsurgeon as gs
import onnx
import copy

from .common import bordered
from .common import find_in_layers, find_node_idx, find_out_layers, is_ancestor
from .common import find_in_layer, find_out_layer
from .common import remove_node



def tidl_detr_optimize_attention (graph: gs.Graph, onnx_graph: onnx.GraphProto):
    logging.info(f"Currently under development, not being used")
    return
    """
    Wrapper function to re-arrange and optimize self-attention block for Transformers
    """
    # create attention objects from the graph
    attention_blocks = tidl_find_attention_block (graph, onnx_graph)
    if len(attention_blocks) > 0:
        logging.info(f"Attention blocks detected: {len(attention_blocks)}, processing for optimization")
    for idx, att in enumerate(attention_blocks):
        logging.debug("\n"+bordered(f"Attention Block {idx}"))
        att.optimize(graph)




class Attention (ABC):
    """
    Single Attention object containing the positions of the nodes in this
    specific attention block in a graph
    -------------------------------------------------------
    This is the abstract class containing absolutely necessary elements of
    Attention blocks. For specific optmizations we use specific derived classes
    """
    def __init__(self):
        ### layers in main attention formula
        ### This must be present ###
        self.matmul_qkt = -1      # QK^T
        self.softmax = -1       # softmax (QK^T/sqrt(d)) = z
        self.matmul_qktv = -1     # zV
        ### -------------------- ###
        # needed params for optimization
        self.num_heads = -1
        self.head_dim = -1
        self.window = -1

        # this flag detects is standard optmization is possible
        # this depends on the
        self.optimize_ok = False




    def printable_attention (self, graph: gs.Graph) -> str:
        """
        Return a string of identified nodes for each specific attention block
        """
        nodes = graph.nodes
        return (f"[heads: {self.num_heads}, headDim: {self.head_dim}]::"
            f"[{nodes[self.matmul_qkt].name}, {nodes[self.softmax].name}, "
            f"{nodes[self.matmul_qktv].name}]")



    @abstractmethod
    def optimize (self, graph: gs.Graph):
        """
        Run optimization for a single attention block
        """
        logging.debug("Abstract Attention class has nothing to optimize")


class DeTRLikeAttention (Attention):
    """
    Torch-like as well as HF-like attention structure needs to be optimized.
    #TODO Improve on the Swin Add optimization and adjust of levit type arch (no reshape)
    #TODO Add support for DETR and Segformer Transformation as well
    """
    def __init__(self):
        super().__init__()
        
        self.split_qkv = -1

        self.qk_add_split = -1

        self.q_t = -1
        self.k_t = -1
        self.v_t = -1
        
        self.q_reshape = -1
        self.k_reshape = -1
        self.v_reshape = -1
        
        self.q_add = -1
        self.k_add = -1
        self.v_add = -1
        
        # nodes to remove output shape after optimization is done
        self.remove_output_shape_list = list()
        
    def optimize(self, graph: gs.Graph):
        """
        Torch-like attention specific optmization
        """
        logging.debug(f"Optimizing attention block {self.printable_attention(graph)}")
        # run various optimizations
        # set optmize to true to start processing
        self.optimize_ok = True
        # populate layers for optimization
        self.populate_structure_specific_layers(graph)
        # process optmizing changes
        self.remove_q_and_v_transpose(graph)
        self.adjust_k_transpose(graph)
        
        self.remove_q_and_k_and_v_reshape(graph)
        
        self.add_transpose_and_reshape_before_split(graph)
        
        self.adjust_split(graph)
        
        self.adjust_q_and_k_and_v_add(graph)
        
        # remove stale data
        self.remove_outdated_output_shapes(graph)

        
    def populate_structure_specific_layers (self, graph: gs.Graph):
        """
        After attention object is created with basic necessary layers
        and the layer where Q,K,V split happens is consolidated,
        this function identifies layers which are specific to deit-like
        attention structures before the layer that splits Q,K,V
        """
        if not self.optimize_ok:
            return

        nodes = graph.nodes
        if self.split_qkv == -1:
            return
        # fill up layers before Q, K, V split
        # traceback input considering only single input

        # Add in QK branch
        for split_out in find_out_layers(nodes[split_qkv]):
            # find a layer with add having a constant input
            if split_out.op=="Add":
                for split_out_inp in split_out.inputs:
                    if isinstance(split_out_inp, gs.Constant):
                        self.qk_add_split = find_node_idx(split_out, graph)
            if split_out.op=="MatMul":
                self.v_matmul = find_node_idx(split_out, graph)
                    #
                #
            #
        #
        if self.qk_add_split == -1:
            # no add layer with constant inp, basically the generic attention block
            # for future 
            self.qk_add_split = self.split_qkv


        # Transpose Q, K

        matmul_qkt = nodes[self.matmul_qkt]
        matmul_qkt_in_nodes = find_in_layers(matmul_qkt)
        for in_node in matmul_qkt_in_nodes:
            if in_node.op == "Transpose":
                pass
            if t_node.attrs['perm'] == [0,2,1,3]:
                self.q_t = find_node_idx(t_node, graph)
            elif t_node.attrs['perm'] == [0,2,3,1]:
                self.k_t = find_node_idx(t_node, graph)
            else:
                logging.critical(f"Invalid structure: {matmul_qkt.name} does not have "
                             f"transpose of required dimensions") 
                self.optimize_ok = False
                return 



        # MatMul Q,K

        curr_layer = nodes[self.split_qkv]
        while (curr_layer is not None) and (curr_layer.op != "MatMul") and (curr_layer != nodes[0]):
            curr_layer = find_in_layer(curr_layer, 0)

        if (curr_layer is not None) and (curr_layer != nodes[0]):
            self.b_matmul = find_node_idx(curr_layer, graph)
        else:
            logging.critical(f"Could not find MatMul projection preceeding "\
                             f"Attention Block ::{self.printable_attention(graph)}")
            self.optimize_ok = False
            return

        # Transpose Q, K
        matmul_qkt = nodes[self.matmul_qkt]
        transpose_nodes = find_in_layers(matmul_qkt)
        for t_node in transpose_nodes:
            if t_node.attrs['perm'] == [0,2,1,3]:
                self.q_t = find_node_idx(t_node, graph)
            elif t_node.attrs['perm'] == [0,2,3,1]:
                self.k_t = find_node_idx(t_node, graph)
            else:
                logging.critical(f"Invalid structure: {matmul_qkt.name} does not have "
                             f"transpose of required dimensions") 
                self.optimize_ok = False
                return    
        
        # Transpose V
        matmul_qktv = nodes[self.matmul_qktv]
        input_nodes = find_in_layers(matmul_qktv)
        for node in input_nodes:
            if node.op == "Transpose":
                self.v_t = find_node_idx(node, graph)
        if self.v_t == -1 :
            logging.critical(f"Invalid structure: {matmul_qktv.name} does not have "
                             f"transpose before it") 
            self.optimize_ok = False
            return    
        
        # Reshape Q,K,V
        q_reshape = find_in_layer(nodes[self.q_t], 0)
        k_reshape = find_in_layer(nodes[self.k_t], 0)
        v_reshape = find_in_layer(nodes[self.v_t], 0)
        
        self.q_reshape = find_node_idx(q_reshape, graph)
        self.k_reshape = find_node_idx(k_reshape, graph)
        self.v_reshape = find_node_idx(v_reshape, graph)
        
        if (q_reshape.op != "Reshape") or (v_reshape.op != "Reshape") \
            or (k_reshape.op != "Reshape"):
                logging.critical(f"Invalid structure: {nodes[self.q_t].name} or k or v does not have "
                             f"reshape before it") 
                self.optimize_ok = False
                return  
            
        # Add Q,K,V
        q_add = find_in_layer(q_reshape, 0)
        k_add = find_in_layer(k_reshape, 0)
        v_add = find_in_layer(v_reshape, 0)
        
        self.q_add = find_node_idx(q_add, graph)
        self.k_add = find_node_idx(k_add, graph)
        self.v_add = find_node_idx(v_add, graph)
        
        if (q_add.op != "Add") or (v_add.op != "Add") \
            or (k_add.op != "Add"):
                logging.critical(f"Invalid structure: {q_reshape.name} or k or v  does not have "
                             f"add before it") 
                self.optimize_ok = False
                return
        

        # if reaches this point, ok for optmization
        self.optimize_ok = True
        
    def remove_q_and_v_transpose (self, graph: gs.Graph):
        """
        """
        if not self.optimize_ok:
            return
        nodes = graph.nodes

        q_t = nodes[self.q_t]
        v_t = nodes[self.v_t]
        logging.debug(f"Removing node {q_t.name}")
        remove_node(q_t)
        logging.debug(f"Removing node {v_t.name}")
        remove_node(v_t)
        
        self.remove_output_shape_list.append(self.q_t)
        self.remove_output_shape_list.append(self.v_t)

        # optimization ok is reached point
        self.optimize_ok = True
        
    def adjust_k_transpose (self, graph: gs.Graph):
        """
        """
        if not self.optimize_ok:
            return
        nodes = graph.nodes
        
        transpose_node = nodes[self.k_t]
        if transpose_node.op == "Transpose":
            transpose_node.attrs['perm'] = [0,1,3,2]
            logging.debug(f"{transpose_node.name} perm changed to {transpose_node.attrs['perm']}")

            # need to remove old output shapes
            self.remove_output_shape_list.append(self.k_t)

        else:
            logging.critical(f"Unsupported operation {transpose_node.op} at transpose"
                             f"location in node {transpose_node.name}")
            self.optimize_ok = False
            return
        #endif
        
        # optimization ok is reached point
        self.optimize_ok = True
        
    def remove_q_and_k_and_v_reshape (self, graph: gs.Graph):
        """
        """
        if not self.optimize_ok:
            return
        nodes = graph.nodes

        q_reshape = nodes[self.q_reshape]
        k_reshape = nodes[self.k_reshape]
        v_reshape = nodes[self.v_reshape]
        
        logging.debug(f"Removing node {q_reshape.name}")
        remove_node(q_reshape)
        logging.debug(f"Removing node {k_reshape.name}")
        remove_node(k_reshape)
        logging.debug(f"Removing node {v_reshape.name}")
        remove_node(v_reshape)
        
        self.remove_output_shape_list.append(self.q_reshape)
        self.remove_output_shape_list.append(self.k_reshape)
        self.remove_output_shape_list.append(self.v_reshape)

        # optimization ok is reached point
        self.optimize_ok = True   
    
    def remove_outdated_output_shapes (self, graph: gs.Graph):
        """
        After updating, many output tensors have different shapes
        Remove existing shapes to help running shape inference
        """
        nodes = graph.nodes

        for idx in self.remove_output_shape_list:
            node = nodes[idx]
            logging.debug(f"Removing old shape of {node.name} output")
            for outp in node.outputs:
                outp.shape = None
                
                
    def add_transpose_and_reshape_before_split (self, graph: gs.Graph):

        """
        Add transpose and reshape layers before spliting layer to maintain consistency

        """
        if not self.optimize_ok:
            return
        nodes = graph.nodes

        split_node = nodes[self.split_qkv]
        b_matmul_node = nodes[self.b_matmul]
        
        tr_perm = {"perm": np.array([0,2,1,3], dtype=np.int64)}
        
        tr_out = gs.Variable(f"tr_{b_matmul_node.inputs[0].name}", dtype= np.float32)
        tr = gs.Node(name= f"Transpose_{b_matmul_node.inputs[0].name}" ,
            op= "Transpose", attrs= tr_perm,
            inputs= [b_matmul_node.outputs[0]], outputs= [tr_out])
        logging.debug(f"Adding new node to graph {tr}")
        graph.nodes.append(tr)
        
        split_node.inputs[0] = tr_out
        
        dims = copy.deepcopy(nodes[self.q_reshape].inputs[1].values)
        dims[-2] = dims[-2]*3
        reshape_dims = gs.Constant(name= f"{b_matmul_node.inputs[0].name}_reshape_dims", values= dims)
              
        reshape_out = gs.Variable(f"reshape_val_{b_matmul_node.inputs[0].name}", dtype= np.float32)
                
        reshape = gs.Node(name= f"Reshape_{b_matmul_node.inputs[0].name}" ,
            op= "Reshape", inputs= [b_matmul_node.outputs[0], reshape_dims],
            outputs= [reshape_out])
        logging.debug(f"Adding new node to graph {reshape}")
        graph.nodes.append(reshape)
        
        tr.inputs[0] = reshape_out

        # optimization ok is reached point
        self.optimize_ok = True 
        
    def adjust_split(self, graph: gs.Graph):
        """
        """
        if not self.optimize_ok:
            return
        nodes = graph.nodes
        
        split_qkv = nodes[self.split_qkv]
        if split_qkv.op == "Split":
            split_qkv.attrs['axis'] = -3
            # split_qkv.attrs['num_outputs'] = 3 # can be used with opset=18
            split_qkv.inputs[1] = gs.Constant(name= f"{nodes[self.b_matmul].inputs[0].name}_new", \
                values = np.array([self.num_heads, self.num_heads, self.num_heads], dtype=np.int64))
            logging.debug(f"{split_qkv.name} axis and split changed to \
                {split_qkv.attrs['axis']} and {split_qkv.inputs[1].values} ")

            # need to remove old output shapes
            self.remove_output_shape_list.append(self.split_qkv)

        else:
            logging.critical(f"Unsupported operation {split_qkv.op} at split"
                             f"location in node {split_qkv.name}")
            self.optimize_ok = False
            return
        #endif
        
        # optimization ok is reached point
        self.optimize_ok = True
        
    def adjust_q_and_k_and_v_add (self, graph: gs.Graph):
        """
        """
        if not self.optimize_ok:
            return
        nodes = graph.nodes

        q_add = nodes[self.q_add]
        k_add = nodes[self.k_add]
        v_add = nodes[self.v_add]
        
        bias_value = np.concatenate((q_add.inputs[0].values, k_add.inputs[0].values, v_add.inputs[0].values))
        
        logging.debug(f"Removing node {q_add.name}")
        remove_node(q_add)
        logging.debug(f"Removing node {k_add.name}")
        remove_node(k_add)
        logging.debug(f"Removing node {v_add.name}")
        remove_node(v_add)
        
        self.remove_output_shape_list.append(self.q_add)
        self.remove_output_shape_list.append(self.k_add)
        self.remove_output_shape_list.append(self.v_add)
        
        b_matmul = nodes[self.b_matmul]
        
        add_out = gs.Variable(f"add_out_{b_matmul.inputs[0].name}", dtype= np.float32)
        add_wts = gs.Constant(name= f"{b_matmul.inputs[0].name}_Bias_Add_constant", values= bias_value)
        add = gs.Node(name= f"{b_matmul.inputs[0].name}_Bias_Add", op= "Add",
                        inputs= [b_matmul.outputs[0], add_wts], outputs= [add_out])
        logging.debug(f"Adding Add node {add.name} with bias from {b_matmul.inputs[0].name}")
        graph.nodes.append(add)
        
        b_matmul.outputs[0].outputs[0].inputs[0] = add_out

        # optimization ok is reached point
        self.optimize_ok = True
 

def tidl_find_attention_block (graph: gs.Graph, onnx_graph: onnx.GraphProto) -> List[Attention]:
    """
    Return a list of Attention objects
    """

    attention_blocks = list()
    nodes = graph.nodes
    num_attentions = 0

    for idx, node in enumerate(nodes):
        # identify softmax as the mark of attention block
        if node.op == "Softmax":
            # create new attention object
            softmax = idx
            att = None
            matmul_qkt, matmul_qktv = -1, -1
            h, dh, w = -1, -1, -1
            logging.debug("-"*50)
            logging.debug(f"Softmax :: {node.name}")
            logging.debug("Searching for Attention...")

            ### The MatMul before and after the softmax are guranteed to be there ###
            ### Match MatMul(Q, K^t) ###
            # find MatMul tracing back softmax's input
            curr_layer = find_in_layer(node, 0)
            while (curr_layer is not None) and (curr_layer.op != "MatMul") and (curr_layer != nodes[0]):      # input is matmul, considering only one input
                curr_layer = find_in_layer(curr_layer, 0)
            # must have 2 variable inputs
            if (curr_layer is not None) and (len(curr_layer.inputs) == 2) and \
                isinstance(curr_layer.inputs[0], gs.Variable) and \
                isinstance(curr_layer.inputs[1], gs.Variable):
                matmul_qkt = find_node_idx(curr_layer, graph)
                logging.debug(f"MatMul (Q,Kt) :: {curr_layer.name}")
            else:
                # do not consider this as attention
                logging.debug("MatMul (Q,Kt) :: Not found")
                continue


            ### Match MatMul(QK^t, V) ###
            # find MatMul in tracing forward softmax's output
            curr_layer = find_out_layer(node, 0)
            while (curr_layer is not None) and (curr_layer.op != "MatMul"):    # output is matmul, considering only one output
                curr_layer = find_out_layer(curr_layer, 0)
            # must have 2 variable inputs
            if (curr_layer is not None) and (len(curr_layer.inputs) == 2) and \
                isinstance(curr_layer.inputs[0], gs.Variable) and \
                isinstance(curr_layer.inputs[1], gs.Variable):
                matmul_qktv = find_node_idx(curr_layer, graph)
                logging.debug(f"MatMul (QKt, V) :: {curr_layer.name}")
            else:
                # do not consider this as attention
                logging.debug("MatMul (QKt, V) :: Not found")
                continue


            ### extract number of heads and dimension of head
            k = nodes[matmul_qkt].outputs[0].shape[-1]  # last dim of MatMul(Q, K^t) output
            for inp in nodes[matmul_qkt].inputs:
                if inp.shape[-1] == k:
                    if len(inp.shape) < 3:
                        logging.info(f"Invalid dimension for input in K side, unable to \
                        resolve number of heads and head dimension for {inp.name}, skipping")
                    # shape generalized as W x h x dh x K
                    dh = inp.shape[-2]
                    h = inp.shape[-3]
                    logging.debug(f"Resolved number of heads = {h}, head dimension = {dh}")

                    if len(inp.shape) > 3:
                        # W is window size when non-zero and validated to be not batch
                        # check if w is batch dimension'
                        # compare with batch size of input to the network i.e., first node
                        if inp.shape[-4] != nodes[0].inputs[0].shape[0]:
                            w = inp.shape[-4]
                            logging.debug(f"Window like dimension found in attention "
                                          f"block:: W = {w}")
                        #endif
                    #endif
                #endif
            #endfor


            # trace back the two inputs to MatMul for (QK^T)V
            logging.debug("Searching for Q, K, V split origin node...")
            for inp_node in find_in_layers(nodes[matmul_qktv]):
                # check if this is the softmax side input i.e., QK^T
                # it is softmax or softmax node is ancestor of this node
                if is_ancestor(nodes[softmax], inp_node, graph):
                    logging.debug(f"Softmax side input from {inp_node.name}")
                    continue
                # this is V side input
                else:
                    logging.debug(f"V-side input from {inp_node.name}")
                    # go upwards tracing back till you find some node which is ancestor to both
                    # input to matmul_qkt
                    curr_layer = inp_node
                    qkt_in_layers = find_in_layers(nodes[matmul_qkt])
                    q_side_in_node, kt_side_in_node = qkt_in_layers[0], qkt_in_layers[1]

                    # find ancestor of q_side_in_node
                    while (curr_layer is not None) and                              \
                        (                                                           \
                            (not is_ancestor(curr_layer, q_side_in_node, graph)) or \
                            (not is_ancestor(curr_layer, kt_side_in_node, graph))   \
                        ):
                        # stop the search when traced back to first node
                        if curr_layer == nodes[0]:
                            break
                        curr_layer = find_in_layer(curr_layer, 0)

                    # searches up from the v branch till a node is found which is an ancestor 
                    # of q or k, assumes self attention - q and k from same branch

                    if (curr_layer is not None) and (curr_layer != nodes[0]) and \
                        (curr_layer.op =="Transpose") :
                        # detr has transpose, making for that rn
                        split_qkv = find_node_idx(curr_layer, graph)
                        # att = DeitLikeAttention()
                        att = DeTRLikeAttention()
                        att.split_qkv = split_qkv
                        
                        logging.debug(f"Found common ancestor of {inp_node.name}, "\
                            f"{q_side_in_node.name} and {kt_side_in_node.name} as "\
                            f"{curr_layer.name}")
                    #endif
                #endif
            #endfor


            ### Finished parsing compulsory elements for this attention ###
            if att is not None:
                att.matmul_qkt = matmul_qkt
                att.matmul_qktv = matmul_qktv
                att.softmax = softmax
                att.num_heads = h
                att.head_dim = dh
                att.window = w

                logging.debug(f"Attention Block {num_attentions} :: {type(att)} :: "\
                    f"{att.printable_attention(graph)}")
                num_attentions += 1
                attention_blocks.append(att)

            logging.debug("-"*50)
        # endif
    # endfor

    return attention_blocks
