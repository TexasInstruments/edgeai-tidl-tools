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



def tidl_optimize_attention (graph: gs.Graph, onnx_graph: onnx.GraphProto):
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


class TorchLikeAttention (Attention):
    """
    Torch-like as well as HF-like attention structure needs to be optimized.
    #TODO Improve on the Swin Add optimization and adjust of levit type arch (no reshape)
    #TODO Add support for DETR and Segformer Transformation as well
    """
    def __init__(self):
        super().__init__()
        
        self.split_qkv = -1
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

        # MatMul
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
 

# class DeitLikeAttention (Attention):
#     """
#     Deit-like attention structure - classic attention for vision transformer (based on old timm)
#     """
#     def __init__(self):
#         super().__init__()
#         self.split_qkv = -1     # layer where split in Q, K, V happens
#                                 # actual split might happen later
#                                 # this marks the start of one attention block
#         # layers before going inside Q,K,V split
#         self.b_matmul = -1
#         self.b_reshape = -1
#         self.b_transpose = -1
#         self.b_add = -1

#         # nodes to remove output shape after optimization is done
#         self.remove_output_shape_list = list()


#     def optimize(self, graph: gs.Graph):
#         """
#         Deit-like attention specific optmization
#         """
#         logging.debug(f"Optimizing attention block {self.printable_attention(graph)}")
#         # run various optimizations
#         # set optmize to true to start processing
#         self.optimize_ok = True
#         # check if sructural changes are possible
#         self.refactor_split_qkv(graph)
#         # populate layers for optimization
#         self.populate_structure_specific_layers(graph)
#         # process optmizing changes
#         self.matmul_layout_optimization(graph)
#         self.eltwise_layout_optimization(graph)

#         self.change_split_axis(graph)
#         self.change_squeeze_axis(graph)
#         self.add_transpose_after_split(graph)

#         self.remove_transpose_before_split(graph)
#         self.change_reshape(graph)

#         # remove stale data
#         self.remove_outdated_output_shapes(graph)


#     def populate_structure_specific_layers (self, graph: gs.Graph):
#         """
#         After attention object is created with basic necessary layers
#         and the layer where Q,K,V split happens is consolidated,
#         this function identifies layers which are specific to deit-like
#         attention structures before the layer that splits Q,K,V
#         """
#         if not self.optimize_ok:
#             return

#         nodes = graph.nodes
#         if self.split_qkv == -1:
#             return
#         # fill up layers before Q, K, V split
#         # traceback input considering only single input

#         # MatMul
#         curr_layer = nodes[self.split_qkv]
#         while (curr_layer is not None) and (curr_layer.op != "MatMul") and (curr_layer != nodes[0]):
#             curr_layer = find_in_layer(curr_layer, 0)

#         if (curr_layer is not None) and (curr_layer != nodes[0]):
#             self.b_matmul = find_node_idx(curr_layer, graph)
#         else:
#             logging.critical(f"Could not find MatMul projection preceeding "\
#                              f"Attention Block ::{self.printable_attention(graph)}")
#             self.optimize_ok = False
#             return

#         # Transpose
#         split_qkv = nodes[self.split_qkv]
#         tr = find_in_layer(split_qkv, 0)
#         if (curr_layer is None) or (tr.op != "Transpose"):
#             logging.critical(f"Invalid structure: {split_qkv.name} is not consumer of "
#                              f"Tranpose layer output")
#             self.optimize_ok = False
#             return
#         else:
#             self.b_transpose = find_node_idx(tr, graph)

#         # Reshape
#         reshp = find_in_layer(tr, 0)
#         if (curr_layer is None) or (reshp.op != "Reshape"):
#             logging.critical(f"Invalid structure: {tr.name} is not consumer of "
#                              f"Reshape layer output")
#             self.optimize_ok = False
#             return
#         else:
#             self.b_reshape = find_node_idx(reshp, graph)

#         # Add
#         curr_layer = nodes[self.split_qkv]
#         while (curr_layer is not None) and (curr_layer.op != "Add") and (curr_layer != nodes[0]):
#             curr_layer = find_in_layer(curr_layer, 0)

#         if (curr_layer is not None) and (curr_layer != nodes[0]) and \
#             (find_node_idx(curr_layer, graph) > self.b_matmul):
#             self.b_add = find_node_idx(curr_layer, graph)
#         else:
#             logging.warning(f"Could not find Add preceeding "
#                 f"Attention Block :: {self.printable_attention(graph)}")


#         # if reaches this point, ok for optmization
#         self.optimize_ok = True


#     def refactor_split_qkv (self, graph: gs.Graph):
#         """
#         Refactor the layer where Q, K, V is split
#         from single tensor
#         ------------------------------------------------
#         If the layer is direct split - just check for
#         valid parameters in output
#         If the layer is something else - and have single
#         output going to 3 consumer layers, check if
#         consumer layers can be refactored as a single split,
#         possible cases are Gather, Slice etc.
#         """
#         if not self.optimize_ok:
#             return

#         nodes = graph.nodes
#         tensors = graph.tensors()

#         split_node = nodes[self.split_qkv]
#         if split_node.op == "Split":
#             self.optimize_ok = True
#         # convert gathers followed by transpose to split and squeeze
#         elif split_node.op == "Transpose":
#             # check all output layers
#             # must be gathers on axis 0 with indices 0,1,2
#             out_layers = find_out_layers(split_node)
#             gather_indices_list = [0, 1, 2]
#             for out_node in out_layers:
#                 if out_node.op == "Gather":
#                     _, indices = out_node.inputs[0], out_node.inputs[1]
#                     gather_indices = np.array(tensors[indices.name].values,
#                                               dtype=np.float32)
#                     if (out_node.attrs['axis'] == 0) and (gather_indices in gather_indices_list):
#                         gather_indices_list.remove(gather_indices)
#                     else:
#                         logging.critical(f"Gather layer {out_node.name} has axis "
#                                          f"{out_node.attrs['axis']} and indices "
#                                          f"{gather_indices} => Unable to refactor to Split")
#                         self.optimize_ok = False
#                         return
#                 else:
#                     logging.critical(f"Output layer of {split_node.name} is expected to be Gather"
#                                      f", found {out_node.name} => Unable to refactor to Split")
#                     self.optimize_ok = False
#                     return
#                 #endif
#             #endfor
#             if len(gather_indices_list) == 0:
#                 logging.debug(f"Refactoring {split_node.name} and followed by Gather layers =>"
#                               "Transpose-Split-Squeeze")
#                 gathers = find_out_layers(split_node)
#                 split_axis = gathers[0].attrs['axis']

#                 new_split_attrs = {
#                         'axis':split_axis,
#                         'split': np.array([1, 1, 1], dtype=np.int64)
#                     }
#                 new_split_outputs = list()
#                 for idx, g in enumerate(gathers):
#                     # create single output for input to the split
#                     op = gs.Variable(name=f"{split_node.name}_Split_Output_{idx}",
#                                      dtype=np.float32)
#                     new_split_outputs.append(op)
#                     # create squeeze to replace gather and remove gather
#                     sq = gs.Node(name= f"{g.name}_replaced_by_squeeze", op="Squeeze",
#                                  attrs={'axes':[split_axis]},inputs=[op],
#                                  outputs=g.outputs)
#                     g.inputs = []
#                     g.outputs = []
#                     self.remove_output_shape_list.append(find_node_idx(g, graph))
#                     nodes[find_node_idx(g, graph)] = sq
#                     logging.debug(f"Replacing {g.name} by new node {sq}")
#                 #endfor
#                 new_split_node = gs.Node(name=f"{split_node.name}_Split", op="Split",
#                                     attrs=new_split_attrs, inputs=split_node.outputs,
#                                     outputs=new_split_outputs)
#                 graph.nodes.append(new_split_node)
#                 self.split_qkv = find_node_idx(new_split_node, graph)
#                 logging.debug(f"Adding new node {new_split_node} as the split origin of K,Q,V")
#             # all gathers with indices 0, 1 & 2 not found
#             else:
#                 logging.critical("Transpose-Gather structure not valid, unable to refactor")
#                 self.optimize_ok = False
#                 return
#             #endif
#         #endelif
#         else:
#             logging.critical(f"Currently operator {split_node.op} for layer as "
#                           f"Origin of Q, K, V is not supported")
#             self.optimize_ok = False
#             return
#         #endif

#         # if reached this point, ok to optimize
#         self.optimize_ok = True


#     def matmul_layout_optimization (self, graph:gs.Graph):
#         """
#         Change the layout of projection MatMul
#         change B-side tensor shape from d x (3*h*dh)
#         to d x (h*3*dh)
#         """
#         if not self.optimize_ok:
#             return

#         nodes = graph.nodes
#         tensors = graph.tensors()
#         # the change of layout of const data start from MatMul
#         node = nodes[self.b_matmul]
#         node_inputs = node.inputs
#         var_inp, const_inp = node_inputs[0], node_inputs[1]
#         b_in = np.array(tensors[const_inp.name].values, dtype=np.float32)

#         if len(b_in.shape) != 2:
#             logging.critical(f"Invalid input "
#             f"shape of const data of projection MatMul :: {b_in.shape} => "
#             f"more than 2 dimensions")
#             self.optimize_ok = False
#             return
#         #endif

#         if b_in.shape[-1] != (self.num_heads*self.head_dim*3):
#             logging.critical(f"Invalid input "
#             f"shape of const data of projection MatMul :: {b_in.shape} != "
#             f"{(self.num_heads*self.head_dim*3)}")
#             self.optimize_ok = False
#             return
#         #endif

#         # change the layout of MatMul's const data tensor
#         k, t = b_in.shape[0], b_in.shape[1]
#         # split t in 3 x h x dh
#         b_in_reshaped = b_in.reshape((k, 3, self.num_heads, self.head_dim))
#         # reshape as h x 3 x dh
#         b_in_reshaped = np.transpose(b_in_reshaped, axes=(0, 2, 1, 3))  # [k, h, 3, dh]
#         # change back to t
#         b_in_reshaped = b_in_reshaped.reshape((k, t))
#         # change const input
#         const_inp_updated = gs.Constant(f"{node.name}_{const_inp.name}", values=b_in_reshaped)
#         # put modified array as input
#         node.inputs = [var_inp, const_inp_updated]
#         logging.debug(f"Updated layout of const data in Node:: {node.name}")

#         # optimization ok is reached point
#         self.optimize_ok = True


#     def eltwise_layout_optimization (self, graph: gs.Graph):
#         """
#         Any eltwise layer between Mul and Reshape must have layout change to
#         maintain consistency
#         Currenly only supports for b_add layer and sinle const data dim
#         """
#         if not self.optimize_ok:
#             return

#         nodes = graph.nodes
#         tensors = graph.tensors()

#         # Similar processing as matmul
#         node = nodes[self.b_add]
#         node_inputs = node.inputs

#         var_inp, const_inp = None, None
#         for inp in node_inputs:
#             if isinstance(inp, gs.Constant):
#                 const_inp = inp
#             else:
#                 var_inp = inp

#         b_in = np.array(tensors[const_inp.name].values, dtype=np.float32)

#         # needs only one dimension
#         if len(b_in.shape) != 1:
#             logging.critical(f"Invalid input "
#             f"shape of const data of {node.name} :: {b_in.shape} => "
#             f"not  only 1 dimension")
#             self.optimize_ok = False
#             return
#         #endif

#         if b_in.shape[-1] != (self.num_heads*self.head_dim*3):
#             logging.critical(f"Invalid input "
#             f"shape of const data of {node.name} :: {b_in.shape} != "
#             f"{(self.num_heads*self.head_dim*3)}")
#             self.optimize_ok = False
#             return
#         #endif

#         # change the layout of MatMul's const data tensor
#         t = b_in.shape[-1]
#         # split t in 3 x h x dh
#         b_in_reshaped = b_in.reshape((3, self.num_heads, self.head_dim))
#         # reshape as h x 3 x dh
#         b_in_reshaped = np.transpose(b_in_reshaped, axes=(1, 0, 2))  # [k, h, 3, dh]
#         # change back to t
#         b_in_reshaped = b_in_reshaped.reshape((t,))
#         # change const input
#         const_inp_updated = gs.Constant(f"{node.name}_{const_inp.name}", values=b_in_reshaped)
#         # put modified array as input
#         node.inputs = [var_inp, const_inp_updated]
#         logging.debug(f"Updated layout of const data in Node:: {node.name}")

#         # optimization ok is reached point
#         self.optimize_ok = True


#     def change_split_axis (self, graph: gs.Graph):
#         """
#         Change the axis of the spliting layer
#         """
#         if not self.optimize_ok:
#             return

#         nodes = graph.nodes

#         split_node = nodes[self.split_qkv]
#         # currently only "Split" operation is supported
#         if split_node.op == "Split":
#             # update split axis
#             num_dims = len(split_node.inputs[0].shape)
#             # as now input is shape [.., k, h, 3, dh]
#             # split in second last axis
#             split_node.attrs['axis'] = num_dims - 2
#             logging.debug(f"{split_node.name} axis changed to {split_node.attrs['axis']}")

#             # need to remove old output shapes
#             self.remove_output_shape_list.append(self.split_qkv)

#         else:
#             logging.critical(f"Unsupported operation {split_node.op} at split"
#                              f"location in node {split_node.name}")
#             self.optimize_ok = False
#             return
#         #endif

#         # optimization ok is reached point
#         self.optimize_ok = True


#     def remove_transpose_before_split (self, graph: gs.Graph):
#         """
#         After optimization we do not need the transpose before the splitting layer
#         """
#         # remove transpose
#         # this only detaches the node, but the node is still
#         # present in node list. This preserves the indices,
#         # which is very vital for stored indices for attention blocks
#         if not self.optimize_ok:
#             return
#         nodes = graph.nodes

#         tr_node = nodes[self.b_transpose]
#         logging.debug(f"Removing node {tr_node.name}")
#         remove_node(tr_node)

#         # optimization ok is reached point
#         self.optimize_ok = True


#     def change_reshape (self, graph: gs.Graph):
#         """
#         The reshape needs to change as the layout of the variable input data has
#         been changed
#         """
#         if not self.optimize_ok:
#             return
#         nodes = graph.nodes
#         tensors = graph.tensors()

#         # change reshape shape
#         reshp_node = nodes[self.b_reshape]
#         data, shape = reshp_node.inputs[0], reshp_node.inputs[1]
#         shape_tensor = np.array(tensors[shape.name].values, dtype= np.int64)

#         if (shape_tensor[-1] == self.head_dim) and \
#             (shape_tensor[-2] == self.num_heads) and \
#             (shape_tensor[-3] == 3):
#             shape_tensor_updated = np.array(list(shape_tensor[:-3]) +
#                                             [self.num_heads, 3, self.head_dim])
#             shape_updated = gs.Constant(name= f"{reshp_node.name}_{shape.name}",
#                                         values=shape_tensor_updated)
#             reshp_node.inputs = [data, shape_updated]
#             logging.debug(f"{reshp_node.name} shape input changed from {shape_tensor}"
#                             f" to {shape_tensor_updated}")

#             # need to remove old output shape
#             self.remove_output_shape_list.append(self.b_reshape)

#         else:
#             logging.critical(f"Invalid shape input for Reshape optmization "
#                                 f":: {shape_tensor}")
#             self.optimize_ok = False
#             return

#         # optimization ok is reached point
#         self.optimize_ok = True


#     def change_squeeze_axis (self, graph: gs.Graph):
#         """
#         As we change the axis of split, we need to change the axis of any
#         subsequent squeeze as well
#         """
#         if not self.optimize_ok:
#             return
#         nodes = graph.nodes
#         tensors = graph.tensors()

#         split_node = nodes[self.split_qkv]
#         out_layers = find_out_layers(split_node)
#         for node in out_layers:
#             # if squeeze found in immediate output
#             if node.op == "Squeeze":
#                 # find axes
#                 attrs = node.attrs
#                 # in attributes
#                 if 'axes' in attrs.keys():
#                     if len(attrs['axes']) == 1:
#                         # update to match split axis
#                         node.attrs['axes'] = np.array([split_node.attrs['axis']], dtype=np.int64)
#                         logging.debug(f"Changed axes of {node.name} to {node.attrs['axes']}")
#                         # need to remove old output shape as well
#                         self.remove_output_shape_list.append(find_node_idx(node, graph))
#                     else:
#                         logging.critical(f"{node.name} has squeeze axes {attrs['axes']} =>"
#                                          f"more than one not supported")
#                         self.optimize_ok = False
#                         return
#                 else:
#                     # search in inputs
#                     logging.debug(f"No attribute `axes` for {node.name}, searching in inputs")
#                     axes_inp = None
#                     if len(node.inputs) == 2 and isinstance(node.inputs[1], gs.Constant):
#                         axes_inp = node.inputs[1]

#                     # found in inputs
#                     if axes_inp is not None:
#                         axes_tensor = np.array(tensors[axes_inp.name].values, dtype=np.float32)
#                         if len(axes_tensor) == 1:
#                             # update to match split axis
#                             axes_tensor_updated = np.array([split_node.attrs['axis']]
#                                                            , dtype=np.int64)
#                             axes_updated_inp = gs.Constant(name= f"{node.name}_{axes_inp.name}",
#                                                            values=axes_tensor_updated)
#                             node.inputs = [node.inputs[0], axes_updated_inp]
#                             logging.debug(f"Changed axes input of {node.name} "
#                                           f"to {axes_tensor_updated}")
#                             # need to remove old output shape as well
#                             self.remove_output_shape_list.append(find_node_idx(node, graph))
#                         else:
#                             logging.critical(f"{node.name} has squeeze axes {axes_tensor} =>"
#                                          f"more than one not supported")
#                             self.optimize_ok = False
#                             return
#                     # not even in inputs => no axes => assume default axes
#                     else:
#                         logging.warning(f"No axes found in {node.name}, adding to attributes")
#                         node.attrs['axes'] = np.array([split_node.attrs['axis']], dtype=np.int64)
#                     #endif
#                 #endif
#             # not supported anything else
#             else:
#                 logging.critical(f"{node.name} after Split {split_node.name} is not supported"
#                                  f"=> operator {node.op} instead of Squeeze")
#                 self.optimize_ok = False
#                 return
#             #endif
#         #endfor

#         # optimization ok is reached point
#         self.optimize_ok = True


#     def add_transpose_after_split (self, graph: gs.Graph):

#         """
#         Add transpose layers after spliting layer to maintain consistency
#         ----------------------------------------------------------------
#         Currenty only split layer is supported and we assume squeeze layer
#         is present afte each split. Transpose layers are added after this
#         squeeze
#         """
#         if not self.optimize_ok:
#             return
#         nodes = graph.nodes

#         split_node = nodes[self.split_qkv]
#         # add transpose for each output after split
#         out_layers = find_out_layers(split_node)
#         for out_node in out_layers:
#             num_dims = len(out_node.outputs[0].shape)
#             dim_k, dim_h, dim_dh = (num_dims-3), (num_dims-2), (num_dims-1)
#             # construct perm array of new transpose layers
#             perm_array = list(range(dim_k)) + [dim_h, dim_k, dim_dh]
#             perm_array = np.array(perm_array, dtype=np.int64)
#             tr_perm = {"perm": perm_array}

#             # output of this node is input to transpose
#             skip_out = out_node.outputs[0]
#             tr_out = gs.Variable(f"tr_{skip_out.name}", dtype= np.float32)
#             tr = gs.Node(name= f"Transpose_{out_node.name}" ,
#                             op= "Transpose", attrs= tr_perm,
#                             inputs= [skip_out], outputs= [tr_out])
#             logging.debug(f"Adding new node to graph {tr}")
#             graph.nodes.append(tr)

#             # fit the node in position
#             # for all consumer of skip_out
#             for in_node in skip_out.outputs:
#                 # get which input was pointing to skip_out
#                 for idx, inp in enumerate(in_node.inputs):
#                     if inp == skip_out:
#                         # change pointing to transpose output
#                         in_node.inputs[idx] = tr_out
#                 #endfor
#             #endfor
#         #endfor

#         # optimization ok is reached point
#         self.optimize_ok = True


#     def remove_outdated_output_shapes (self, graph: gs.Graph):
#         """
#         After updating, many output tensors have different shapes
#         Remove existing shapes to help running shape inference
#         """
#         nodes = graph.nodes

#         for idx in self.remove_output_shape_list:
#             node = nodes[idx]
#             logging.debug(f"Removing old shape of {node.name} output")
#             for outp in node.outputs:
#                 outp.shape = None



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


                    if (curr_layer is not None) and (curr_layer != nodes[0]) and \
                        (curr_layer.op == "Split" or curr_layer.op =="Transpose") :
                        # common ancestor found and this is some split layer
                        # currently assumes that q,k and v are coming from same split
                        split_qkv = find_node_idx(curr_layer, graph)
                        # att = DeitLikeAttention()
                        att = TorchLikeAttention()
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
