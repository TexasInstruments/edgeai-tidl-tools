import logging
from typing import List
from abc import ABC, abstractmethod
import numpy as np
import onnx_graphsurgeon as gs
import onnx
from .common import insert_subgraph_between_tensors  
from .common import insert_subgraph_with_mappings

# Import helper functions from your common utilities
from .common import (
    find_in_layers, find_node_idx,
    find_in_layer, find_out_layer,
)

class Attention(ABC):
    """
    Abstract base class for attention block representation.
    """
    def __init__(self):
        self.matmul_qkt = -1      # QK^T MatMul node index
        self.softmax = -1         # Softmax node index
        self.matmul_qktv = -1     # (QK^T)V MatMul node index
        self.num_heads = -1
        self.head_dim = -1
        self.window = -1
        self.att_idx = -1
        self.before_tensor = -1
        self.after_tensor = -1
        self.subgraph_input = None
        self.subgraph_output = None
        self.new_subgraph = None
        self.optimize_ok = False

    def printable_attention(self, graph: gs.Graph) -> str:
        """
        Return a string of identified nodes for each specific attention block
        """
        nodes = graph.nodes
        return (f"[heads: {self.num_heads}, headDim: {self.head_dim}]::"
                f"[{nodes[self.matmul_qkt].name if self.matmul_qkt != -1 else 'N/A'}, "
                f"{nodes[self.softmax].name if self.softmax != -1 else 'N/A'}, "
                f"{nodes[self.matmul_qktv].name if self.matmul_qktv != -1 else 'N/A'}]")

    @abstractmethod
    def optimize(self, graph: gs.Graph):
        """
        Run optimization for a single attention block
        """
        logging.debug("Abstract Attention class has nothing to optimize")

class DeTRLikeAttention(Attention):
    """
    Class for DETR-style attention patterns where Q and K branches are traced up,
    and both must merge at an Add node (with matching operator order and details).
    """
    def __init__(self):
        super().__init__()
        self.q_branch = []        
        self.k_branch = []        
        self.add_node = -1
        
        self.q_proj_matmul = -1         # can fuse
        self.k_proj_matmul = -1
        
        self.q_proj_add = -1            # can fuse
        self.k_proj_add = -1  
        
        self.q_reshape = -1             # can fuse
        self.k_reshape = -1
              
        self.q_transpose = -1           # can fuse
        self.k_transpose = -1 
        
        self.q_sec_reshape = -1         # can fuse    
        self.k_sec_reshape = -1
        
        self.k_last_transpose = -1
        self.scalar_op = -1
            
        self.branch_ops_match = False  

    def printable_attention(self, graph: gs.Graph) -> str:
        nodes = graph.nodes
        q_ops = [nodes[idx].op for idx in self.q_branch]
        k_ops = [nodes[idx].op for idx in self.k_branch]
        return (f"[heads: {self.num_heads}, headDim: {self.head_dim}]::"
                f"Softmax: {nodes[self.softmax].name if self.softmax != -1 else 'N/A'}, "
                f"MatMul_before: {nodes[self.matmul_qkt].name if self.matmul_qkt != -1 else 'N/A'}, "
                f"MatMul_after: {nodes[self.matmul_qktv].name if self.matmul_qktv != -1 else 'N/A'}, "
                f"Add_merge: {nodes[self.add_node].name if self.add_node != -1 else 'N/A'}, "
                f"Q_branch_ops: {q_ops}, "
                f"K_branch_ops: {k_ops}, "
                f"Branches_match: {self.branch_ops_match}")
        
    def optimize(self, graph: gs.Graph):
        """
        Fuses Q and K branches into a single subgraph and stores it in self.new_subgraph.
        """
        suffix = f"_att_idd_{self.att_idx}"

        nodes = graph.nodes

        # 1. Get Q and K MatMul weights and biases
        q_matmul = nodes[self.q_proj_matmul]
        k_matmul = nodes[self.k_proj_matmul]
        q_weight = next(inp.values for inp in q_matmul.inputs if isinstance(inp, gs.Constant))
        k_weight = next(inp.values for inp in k_matmul.inputs if isinstance(inp, gs.Constant))

        q_add = nodes[self.q_proj_add]
        k_add = nodes[self.k_proj_add]
        q_bias = next(inp.values for inp in q_add.inputs if isinstance(inp, gs.Constant))
        k_bias = next(inp.values for inp in k_add.inputs if isinstance(inp, gs.Constant))

        # 2. Fuse weights and biases
        fused_weight = np.concatenate([q_weight, k_weight], axis=-1)
        fused_bias = np.concatenate([q_bias, k_bias], axis=0)

        # 3. Build fused MatMul and Add
        input_shape = q_matmul.inputs[0].shape  # Assuming both Q and K use the same input shape
        input_tensor = gs.Variable(f"input_tensor{suffix}", dtype=np.float32, shape=input_shape)  # Both Q and K use the same input
        self.subgraph_input = input_tensor.name
        fused_weight_const = gs.Constant(f"fused_qk_weight{suffix}", values=fused_weight)
        fused_bias_const = gs.Constant(f"fused_qk_bias{suffix}", values=fused_bias)
        fused_matmul_out = gs.Variable(f"fused_matmul_out{suffix}", dtype=np.float32)
        fused_matmul = gs.Node(op="MatMul", name=f"fused_qk_matmul{suffix}", inputs=[input_tensor, fused_weight_const], outputs=[fused_matmul_out])
        fused_add_out = gs.Variable(f"fused_add_out{suffix}", dtype=np.float32)
        fused_add = gs.Node(op="Add", name=f"fused_qk_add{suffix}", inputs=[fused_matmul_out, fused_bias_const], outputs=[fused_add_out])

        # 4. Fused Reshape
        orig_reshape = nodes[self.k_reshape]
        orig_shape = next(inp.values for inp in orig_reshape.inputs if isinstance(inp, gs.Constant))
        fused_shape = orig_shape.copy()
        fused_shape[-2] = fused_shape[-2] * 2  # Double the num_heads axis
        fused_reshape_shape_const = gs.Constant(f"fused_reshape_shape{suffix}", values=fused_shape)
        fused_reshape_out = gs.Variable(f"fused_reshape_out{suffix}", dtype=np.float32)
        fused_reshape = gs.Node(
            op="Reshape",
            name=f"fused_qk_reshape{suffix}",
            inputs=[fused_add_out, fused_reshape_shape_const],
            outputs=[fused_reshape_out]
        )

        # 5. Fused Transpose (use K side's perm)
        q_transpose = nodes[self.k_transpose]
        perm = q_transpose.attrs["perm"]
        fused_transpose_out = gs.Variable(f"fused_transpose_out{suffix}", dtype=np.float32)
        fused_transpose = gs.Node(
            op="Transpose",
            name=f"fused_qk_transpose{suffix}",
            inputs=[fused_reshape_out],
            outputs=[fused_transpose_out],
            attrs={"perm": perm}
        )

        # 6. Fused second Reshape (if present)
        orig_sec_reshape = nodes[self.k_sec_reshape]
        sec_shape = next(inp.values for inp in orig_sec_reshape.inputs if isinstance(inp, gs.Constant))
        fused_sec_shape = sec_shape.copy()
        fused_sec_shape[-3] = fused_sec_shape[-3] * 2
        fused_sec_reshape_shape_const = gs.Constant(f"fused_sec_reshape_shape{suffix}", values=fused_sec_shape)
        fused_sec_reshape_out = gs.Variable(f"fused_sec_reshape_out{suffix}", dtype=np.float32)
        fused_sec_reshape = gs.Node(
            op="Reshape",
            name=f"fused_qk_sec_reshape{suffix}",
            inputs=[fused_transpose_out, fused_sec_reshape_shape_const],
            outputs=[fused_sec_reshape_out]
        )


        # 7. Split Q and K
        split_size = self.num_heads
        split_sizes = np.array([split_size, split_size], dtype=np.int64)
        split_sizes_const = gs.Constant(f"split_sizes{suffix}", values=split_sizes)
        split_q = gs.Variable(f"split_q{suffix}", dtype=np.float32)
        split_k = gs.Variable(f"split_k{suffix}", dtype=np.float32)
        split_node = gs.Node(
            op="Split",
            name=f"split_qk{suffix}",
            inputs=[fused_sec_reshape_out, split_sizes_const],
            outputs=[split_q, split_k],
            attrs={"axis": -3}
        )
        
        # 8. k side split to transpose
        k_last_transpose = nodes[self.k_last_transpose]
        k_perm = k_last_transpose.attrs["perm"]
        last_k_transpose_out = gs.Variable(f"last_k_transpose_out{suffix}", dtype=np.float32)
        last_k_transpose_node = gs.Node(
            op="Transpose",
            name=f"split_k_transpose{suffix}",
            inputs=[split_k],
            outputs=[last_k_transpose_out],
            attrs={"perm": k_perm}
        )
        
        # 9. QK^T MatMul
        matmul_out = gs.Variable(f"matmul_qkt_out{suffix}", dtype=np.float32)
        matmul_qkt = gs.Node(
            op="MatMul",
            name=f"matmul_qkt{suffix}",
            inputs=[split_q, last_k_transpose_out],
            outputs=[matmul_out]
        )

        # 10. Fused Q scaling (if present)
        scalar_node = nodes[self.scalar_op]
        output_shape = nodes[self.matmul_qkt].outputs[0].shape 
        # Find the constant input (the scalar value)
        scalar_const = next(inp for inp in scalar_node.inputs if isinstance(inp, gs.Constant))
        scalar_out = gs.Variable(f"q_scaled{suffix}", dtype=np.float32, shape=output_shape)
        q_scaled = gs.Node(
            op=scalar_node.op,
            name=f"q_scaled{suffix}",
            inputs=[matmul_out, scalar_const],  
            outputs=[scalar_out]
        )

        self.subgraph_output = scalar_out.name
        self.new_subgraph = gs.Graph(
            nodes=[fused_matmul, fused_add, fused_reshape, fused_transpose, fused_sec_reshape, split_node, last_k_transpose_node, matmul_qkt, q_scaled],
            inputs=[input_tensor],
            outputs=[scalar_out] 
        )
        
        # onnx.save(gs.export_onnx(self.new_subgraph), "/home/a1244837/Pulkit/TIDL_Optim/new_new_graph.onnx")
        self.optimize_ok = True


        
    
def tidl_detr_attention(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Find QK-Add attention patterns in the graph.
    """
    attention_blocks = []
    nodes = graph.nodes
    num_attentions = 0

    for idx, node in enumerate(nodes):
        if node.op == "Softmax":
            att = DeTRLikeAttention()
            att.softmax = idx
            att.att_idx = num_attentions

            # Find MatMul before Softmax
            curr_layer = find_in_layer(node, 0)
            while (curr_layer is not None) and (curr_layer.op != "MatMul") and (curr_layer != nodes[0]):
                curr_layer = find_in_layer(curr_layer, 0)
            if (curr_layer is not None) and (curr_layer.op == "MatMul") and (len(curr_layer.inputs) == 2) and isinstance(curr_layer.inputs[0], gs.Variable) and isinstance(curr_layer.inputs[1], gs.Variable) :
                att.matmul_qkt = find_node_idx(curr_layer, graph)
                logging.debug(f"{att.att_idx} MatMul before Softmax : {curr_layer.name}")
            else:
                logging.debug("MatMul before Softmax : Not found")
                continue

            # Find MatMul after Softmax
            curr_layer = find_out_layer(node, 0)
            while (curr_layer is not None) and (curr_layer.op != "MatMul"):
                curr_layer = find_out_layer(curr_layer, 0)
            if (curr_layer is not None) and (curr_layer.op == "MatMul") and (len(curr_layer.inputs) == 2) and isinstance(curr_layer.inputs[0], gs.Variable) and isinstance(curr_layer.inputs[1], gs.Variable):
                att.matmul_qktv = find_node_idx(curr_layer, graph)
                logging.debug(f"{att.att_idx} MatMul after Softmax :: {curr_layer.name}")
            else:
                logging.debug("MatMul after Softmax :: Not found")
                continue

            ### extract number of heads and dimension of head
            k = nodes[att.matmul_qkt].outputs[0].shape[-1]  # last dim of MatMul(Q, K^t) output
            for inp in nodes[att.matmul_qkt].inputs:
                if inp.shape[-1] == k:
                    if len(inp.shape) < 3:
                        logging.info(f"{att.att_idx} Invalid dimension for input in K side, unable to \
                        resolve number of heads and head dimension for {inp.name}, skipping")
                    # shape generalized as W x h x dh x K
                    att.head_dim = inp.shape[-2]
                    att.num_heads = inp.shape[-3]
                    logging.debug(f"{att.att_idx} Resolved number of heads = {att.num_heads}, head dimension = {att.head_dim}")

                    if len(inp.shape) > 3:
                        # W is window size when non-zero and validated to be not batch
                        # check if w is batch dimension'
                        # compare with batch size of input to the network i.e., first node
                        if inp.shape[-4] != nodes[0].inputs[0].shape[0]:
                            att.window = inp.shape[-4]
                            logging.debug(f"{att.att_idx} Window like dimension found in attention "
                                          f"{att.att_idx} block:: W = {att.window}")

            logging.debug("Searching for common node(mostly Add node)")
            
            qkt_in_layers = find_in_layers(nodes[att.matmul_qkt])
            q_side_in_node, kt_side_in_node = qkt_in_layers[0], qkt_in_layers[1]
            
            def is_scalar_mul_or_div(node):
                if node.op not in ["Mul", "Div"]:
                    return False
                if len(node.inputs) != 2:
                    return False
                shapes = [inp.shape for inp in node.inputs]
                # Check for scalar: shape == [] or shape == [1]
                is_scalar = [s is not None and (len(s) == 0 or (len(s) == 1 and s[0] == 1)) for s in shapes]
                return any(is_scalar)
            
            # Finding the attention pattern
            q_branch = []
            k_branch = []
            q_branch.append(att.matmul_qkt)
            k_branch.append(att.matmul_qkt)
            while q_side_in_node and q_side_in_node != nodes[0] and kt_side_in_node and kt_side_in_node != nodes[0]:
                if(q_side_in_node == kt_side_in_node ):
                    # if(q_side_in_node.op == "Add"):
                    q_branch.append(find_node_idx(q_side_in_node, graph))
                    k_branch.append(find_node_idx(kt_side_in_node, graph))
                    att.add_node = find_node_idx(q_side_in_node, graph)
                    att.branch_ops_match = True
                    break
                    # else:
                    #     logging.debug("Not an attention pattern as q and kT branch don't meet")
                    #     break
                
                #first transpose in the kT branch needs to be skipped
                if(kt_side_in_node.op == "Transpose" and q_side_in_node.op != "Transpose"):
                    k_branch.append(find_node_idx(kt_side_in_node, graph))
                    kt_side_in_node = find_in_layer(kt_side_in_node, 0)
                    continue
                if(kt_side_in_node.op != q_side_in_node.op):
                    if(q_side_in_node.op == "Mul" or q_side_in_node.op == "Div"):
                        # check if its a scalar mul or div
                        if(is_scalar_mul_or_div(q_side_in_node)): #FIXME
                            q_branch.append(find_node_idx(q_side_in_node, graph))
                            q_side_in_node = find_in_layer(q_side_in_node, 0)
                        else:
                            logging.debug("Not an attention pattern as operators don't match")
                            break
                        
                q_branch.append(find_node_idx(q_side_in_node, graph))
                k_branch.append(find_node_idx(kt_side_in_node, graph))
                q_side_in_node = find_in_layer(q_side_in_node, 0)
                kt_side_in_node = find_in_layer(kt_side_in_node, 0)
                
                
            if(att.branch_ops_match == False):
                logging.debug("Not an attention pattern as q and kT branch don't meet")
                continue

            att.q_branch = list(reversed(q_branch))
            att.k_branch = list(reversed(k_branch))
            
            
            # specific to DeTR like attention pattern
            if len(att.q_branch) < 8 or len(att.k_branch) < 8:
                logging.debug("Not an attention pattern as q or kT branch is too short")
                continue
            if att.add_node == -1:
                logging.debug("Not an attention pattern as q and kT branch don't meet at Add node")
                continue
            if att.q_branch[0] != att.k_branch[0]:
                logging.debug("Not an attention pattern as q and kT branch don't start at same node")
                continue
            if len(att.q_branch) != len(att.k_branch):
                logging.debug("Not an attention pattern as q and kT branch don't have same length")
                continue
            if(att.q_branch[0] != att.add_node or att.k_branch[0] != att.add_node):
                logging.debug("Not an attention pattern as q and kT branch don't start at Add node")
                continue
            if(nodes[att.q_branch[1]].op != "MatMul" or nodes[att.k_branch[1]].op != "MatMul"):
                logging.debug("Not an attention pattern as q and kT branch don't start with MatMul")
                continue
            if(nodes[att.q_branch[2]].op != "Add" or nodes[att.k_branch[2]].op != "Add"):
                logging.debug("Not an attention pattern as q and kT branch don't end with Transpose")
                continue
            if(nodes[att.q_branch[3]].op != "Div" and nodes[att.q_branch[3]].op != "Mul" ):
                logging.debug("Not an attention pattern as q and kT branch don't have scalar op")
                continue
            if(nodes[att.q_branch[4]].op != "Reshape" or nodes[att.k_branch[3]].op != "Reshape"):
                logging.debug("Not an attention pattern as q and kT branch don't have Reshape")
                continue
            if(nodes[att.q_branch[5]].op != "Transpose" or nodes[att.k_branch[4]].op != "Transpose"):   
                logging.debug("Not an attention pattern as q and kT branch don't have second Reshape")
                continue
            if(nodes[att.q_branch[6]].op != "Reshape" or nodes[att.k_branch[5]].op != "Reshape"):
                logging.debug("Not an attention pattern as q and kT branch don't have second Reshape")
                continue
            if(nodes[att.k_branch[6]].op != "Transpose"):
                logging.debug("Not an attention pattern as kT branch doesn't have Transpose")
                continue
            if(find_in_layer(nodes[att.softmax],0) != nodes[att.matmul_qkt]):
                logging.debug("Not an attention pattern as Softmax is not after MatMul(Q, K^T)")
                continue
            
            
            
            att.before_tensor = nodes[att.q_branch[0]].outputs[0].name
            att.after_tensor = nodes[att.k_branch[-1]].outputs[0].name
            
            # populate the vars for q and k branches
            for idx in att.q_branch[1:]:  # Start traversing from index 1 after reversing
                node = graph.nodes[idx]
                if node.op == "MatMul" and att.q_proj_matmul == -1:
                    att.q_proj_matmul = idx
                elif node.op == "Add" and att.q_proj_add == -1:
                    att.q_proj_add = idx
                elif node.op == "Reshape" and att.q_reshape == -1:
                    att.q_reshape = idx
                elif node.op == "Reshape" and att.q_sec_reshape == -1:
                    att.q_sec_reshape = idx
                elif node.op == "Transpose" and att.q_transpose == -1:
                    att.q_transpose = idx
                elif node.op == "Mul" or node.op == "Div":
                    att.scalar_op = idx

            # Populate class variables for K branch
            for idx in att.k_branch[1:]:
                node = graph.nodes[idx]
                if node.op == "MatMul" and att.k_proj_matmul == -1:
                    att.k_proj_matmul = idx
                elif node.op == "Add" and att.k_proj_add == -1:
                    att.k_proj_add = idx
                elif node.op == "Reshape" and att.k_reshape == -1:
                    att.k_reshape = idx
                elif node.op == "Reshape" and att.k_sec_reshape == -1:
                    att.k_sec_reshape = idx
                elif node.op == "Transpose" and att.k_transpose == -1:
                    att.k_transpose = idx
                elif node.op == "Transpose" and att.k_last_transpose == -1:
                    att.k_last_transpose = idx
            att.optimize(graph)
            
            attention_blocks.append(att)
            num_attentions += 1


    for att in attention_blocks:
        success = insert_subgraph_with_mappings(
            graph,
            {att.subgraph_input : att.before_tensor},
            {att.subgraph_output : att.after_tensor},
            att.new_subgraph
        )
        if success:
            # onnx.save(gs.export_onnx(graph), "/home/a1244837/Pulkit/TIDL_Optim/super_new_graph.onnx")
            logging.info(f"Optimized attention block inserted for {att.att_idx}")
        else:
            logging.info(f"Failed to insert optimized attention block for {att.att_idx}")
            
    # return attention_blocks
