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
"""
Module tracking all possible opts and their dependencies.
Can give a topologically sorted order of execution to handle
dependencies between optimizations.
Only module to grow with time
"""
import logging
from typing import List, Dict


# importing all opt functions
from .src.argmax import tidl_change_argmax_keepdims_to_1
from .src.resize import tidl_convert_resize_params_size_to_scale, tidl_convert_resize_params_size_to_scale_dynamic_batch, tidl_remove_unity_resize
from .src.attention import tidl_optimize_attention
from .src.attention_hf import tidl_optimize_hf_attention
from .src.batch import tidl_modify_batch_dim
from .src.concat import tidl_convert_concat_axis_width_to_channel, tidl_convert_single_concat_to_consecutive_concats
from .src.maxpool import tidl_convert_maxpool_to_cascaded_maxpool
from .src.reducemean import tidl_convert_reducemean_to_matmul
from .src.gemm import tidl_convert_gemm_to_matmul_and_add
from .src.matmul import tidl_convert_matmul_to_conv_1x1s1, tidl_push_matmul_channel_in_height
from .src.global_avg_pool import tidl_convert_large_global_avg_pooling_to_matmul
from .src.gather import tidl_convert_gather_with_single_index_to_slice
from .src.batchnorm import tidl_convert_batchnorm_input_to_4D
from .src.softmax import tidl_convert_softmax_axis_channel_to_width, tidl_convert_softmax_axis_height_to_width
from .src.softmax import tidl_push_large_channel_dim_to_height_for_width_wise_softmax
from .src.conv import tidl_convert_conv_large_pad_to_smaller_kernel, tidl_convert_conv_7x7_stride4_to_stride1, tidl_convert_conv_even_filter_to_odd, \
    tidl_convert_tr_conv_stride_n_tr_to_matmul
from .src.layernorm import tidl_expand_layernorm_to_component_ops
from .src.slice import tidl_expand_slice_across_multiple_axis, tidl_convert_2_dimension_slice_to_maxpool, tidl_eliminate_noop_slice
from .src.instancenorm import tidl_convert_instancenorm_to_layernorm
from .src.unsqueeze import tidl_convert_unsqueeze_to_reshape, tidl_eliminate_unsqueeze
from .src.qdq import tidl_add_bias_qdq, tidl_remove_quantize_initializer, tidl_remove_duplicate_quantize_dequantize
from .src.neg import tidl_convert_neg_to_mul
from .src.expand import tidl_convert_expand_to_reshape_and_concat
from .src.reducesum import tidl_convert_reducesum_to_matmul
from .src.eltwise import tidl_replace_mean_with_eltwise, tidl_replace_sub_with_neg_add, tidl_support_broadcast_ops_constant_input
from .src.depthtospace import tidl_insert_1x1_conv_before_depthtospace, tidl_convert_depth2space_to_reshp_tr_reshp
from .src.spacetodepth import tidl_convert_space2depth_to_reshp_tr_reshp
from .src.common import tidl_remove_duplicates
from .src.gelu import tidl_convert_tanhgelu_to_erfgelu, tidl_break_gelu_to_components
from .src.where import tidl_remove_where_layer
from .src.reshp_tr_reshp import tidl_optimize_reshp_tr_reshp
from .src.attention_detr import tidl_detr_attention
from .src.input_optimization import tidl_add_input_normalization

### function dict to execute
opt_ops = {
        "add_input_normalization"                   : tidl_add_input_normalization,
        'convert_resize_params_size_to_scale'       : tidl_convert_resize_params_size_to_scale,
        'attention_block_optimization'              : tidl_optimize_attention,
        'hf_attention_block_optimization'           : tidl_optimize_hf_attention,
        'hf_detr_attention_block_optimization'      : tidl_detr_attention,
        'convert_concat_axis_width_to_channel'      : tidl_convert_concat_axis_width_to_channel,
        'split_batch_dim_to_parallel_input_branches': tidl_modify_batch_dim,
        'convert_maxpool_to_cascaded_maxpool'       : tidl_convert_maxpool_to_cascaded_maxpool,
        'convert_reducemean_to_matmul'              : tidl_convert_reducemean_to_matmul,
        'convert_gemm_to_matmul_and_add'            : tidl_convert_gemm_to_matmul_and_add,
        'convert_matmul_to_conv_1x1s1'              : tidl_convert_matmul_to_conv_1x1s1,
        'convert_large_global_avg_pooling_to_matmul': tidl_convert_large_global_avg_pooling_to_matmul,
        'convert_gather_with_single_index_to_slice' : tidl_convert_gather_with_single_index_to_slice,
        'convert_batchnorm_input_to_4D'             : tidl_convert_batchnorm_input_to_4D,
        'convert_softmax_axis_channel_to_width'     : tidl_convert_softmax_axis_channel_to_width,
        'convert_softmax_axis_height_to_width'      : tidl_convert_softmax_axis_height_to_width,
        'push_large_channel_dim_to_height_for_width_wise_softmax': tidl_push_large_channel_dim_to_height_for_width_wise_softmax,
        'convert_conv_large_pad_to_smaller_kernel'  : tidl_convert_conv_large_pad_to_smaller_kernel,
        'convert_conv_7x7_stride4_to_stride1'       : tidl_convert_conv_7x7_stride4_to_stride1,
        'expand_layernorm_to_component_ops'         : tidl_expand_layernorm_to_component_ops,
        'push_matmul_channel_in_height'             : tidl_push_matmul_channel_in_height,
        'expand_slice_across_multiple_axis'         : tidl_expand_slice_across_multiple_axis,
        'convert_instancenorm_to_layernorm'         : tidl_convert_instancenorm_to_layernorm,
        'convert_unsqueeze_to_reshape'              : tidl_convert_unsqueeze_to_reshape,
        'add_bias_qdq'                              : tidl_add_bias_qdq,
        'remove_quantize_initializer'               : tidl_remove_quantize_initializer,
        'remove_duplicate_quantize_dequantize'      : tidl_remove_duplicate_quantize_dequantize,
        "convert_neg_to_mul"                        : tidl_convert_neg_to_mul,
        "convert_expand_to_reshape_and_concat"      : tidl_convert_expand_to_reshape_and_concat,
        "convert_single_concat_to_consecutive_concats" : tidl_convert_single_concat_to_consecutive_concats, 
        "change_argmax_keepdims_to_1"               : tidl_change_argmax_keepdims_to_1,
        "convert_2_dimension_slice_to_maxpool"      : tidl_convert_2_dimension_slice_to_maxpool,
        "convert_reducesum_to_matmul"               : tidl_convert_reducesum_to_matmul,
        'convert_resize_params_size_to_scale_dynamic_batch' : tidl_convert_resize_params_size_to_scale_dynamic_batch,
        'replace_mean_with_eltwise'                 : tidl_replace_mean_with_eltwise,
        'replace_sub_with_neg_add'                  : tidl_replace_sub_with_neg_add,
        'convert_conv_even_filter_to_odd'           : tidl_convert_conv_even_filter_to_odd,
        'remove_duplicates'                         : tidl_remove_duplicates,
        'remove_unity_resize'                       : tidl_remove_unity_resize,
        'insert_1x1_conv_before_depthtospace'       : tidl_insert_1x1_conv_before_depthtospace,
        'convert_depth2space_to_reshp_tr_reshp'     : tidl_convert_depth2space_to_reshp_tr_reshp,
        'convert_space2depth_to_reshp_tr_reshp'     : tidl_convert_space2depth_to_reshp_tr_reshp,
        'convert_tanhgelu_to_erfgelu'               : tidl_convert_tanhgelu_to_erfgelu,
        'support_broadcast_ops_constant_input'      : tidl_support_broadcast_ops_constant_input,
        "remove_where_layer"                        : tidl_remove_where_layer,
        "convert_tr_conv_stride_n_tr_to_matmul"     : tidl_convert_tr_conv_stride_n_tr_to_matmul,
        "optimize_reshp_tr_reshp"                   : tidl_optimize_reshp_tr_reshp,
        "eliminate_noop_slice"                      : tidl_eliminate_noop_slice,
        "eliminate_unsqueeze"                       : tidl_eliminate_unsqueeze,
        "break_gelu_to_components"                  : tidl_break_gelu_to_components,   
}

# Bucket definitions
BUCKETS = {
    "BASIC_ALL": [
        'eliminate_noop_slice',
        'remove_duplicates',
        'convert_resize_params_size_to_scale',
    ],
    "EXTENDED_ALL": [
        'expand_slice_across_multiple_axis',
        'convert_maxpool_to_cascaded_maxpool',
    ],
    "LAYOUT_ALL": [
        'convert_depth2space_to_reshp_tr_reshp',
    ]
}

BUCKET_ADJ_LIST = {
    "BASIC_ALL": [],
    "EXTENDED_ALL": ["BASIC_ALL"],
    "LAYOUT_ALL": ["EXTENDED_ALL"],
}


def expand_bucket_flags(args):
    """
    Only log which buckets are enabled. Do NOT set all optimizations in the bucket to True.
    """
    for bucket in BUCKETS.keys():
        if args.get(bucket, False):
            logging.info(f"[BUCKET] {bucket} optimizations enabled via {bucket}")
                

qdq_supported_ops = ['add_bias_qdq', 'remove_quantize_initializer', 'remove_duplicate_quantize_dequantize']

# adjancency list
adj_list = {
        'add_input_normalization'                   : [],
        'convert_resize_params_size_to_scale'       : [],
        'attention_block_optimization'              : [],
        'hf_attention_block_optimization'           : ['hf_detr_attention_block_optimization'],
        'hf_detr_attention_block_optimization'        : [],
        'convert_concat_axis_width_to_channel'      : [],
        'split_batch_dim_to_parallel_input_branches': [],
        'convert_maxpool_to_cascaded_maxpool'       : [],
        'convert_reducemean_to_matmul'              : ['expand_layernorm_to_component_ops'],
        'convert_gemm_to_matmul_and_add'            : ['convert_large_global_avg_pooling_to_matmul'],
        'convert_matmul_to_conv_1x1s1'              : ['convert_gemm_to_matmul_and_add'],     # don't want the matmul from gemm to change                                          
        'convert_large_global_avg_pooling_to_matmul': ['push_matmul_channel_in_height'],
        'convert_gather_with_single_index_to_slice' : [],
        'convert_batchnorm_input_to_4D'             : [],
        'convert_softmax_axis_channel_to_width'     : [],
        'convert_softmax_axis_height_to_width'      : [],
        'push_large_channel_dim_to_height_for_width_wise_softmax': [],
        'convert_conv_large_pad_to_smaller_kernel'  : [],
        'convert_conv_7x7_stride4_to_stride1'       : [],
        'expand_layernorm_to_component_ops'         : ['attention_block_optimization', 'hf_attention_block_optimization'],
        'push_matmul_channel_in_height'             : [],
        'expand_slice_across_multiple_axis'         : ['expand_slice_across_multiple_axis'],
        'convert_instancenorm_to_layernorm'         : ['expand_layernorm_to_component_ops'],
        'convert_unsqueeze_to_reshape'              : [],
        'add_bias_qdq'                              : [],
        'remove_quantize_initializer'               : [],
        'remove_duplicate_quantize_dequantize'      : [],
        "convert_neg_to_mul"                        : [],
        "convert_expand_to_reshape_and_concat"      : ['convert_single_concat_to_consecutive_concats'],
        "convert_single_concat_to_consecutive_concats" : [],
        "change_argmax_keepdims_to_1"               : [],
        "convert_2_dimension_slice_to_maxpool"      : ['expand_slice_across_multiple_axis', 'convert_maxpool_to_cascaded_maxpool'],
        "convert_reducesum_to_matmul"               : [],
        'convert_resize_params_size_to_scale_dynamic_batch' : ['convert_resize_params_size_to_scale'],
        'replace_mean_with_eltwise'                 : [],
        'replace_sub_with_neg_add'                  : ['support_broadcast_ops_constant_input'],
        'convert_conv_even_filter_to_odd'           : [],
        'remove_duplicates'                         : [],
        'remove_unity_resize'                       : ['convert_resize_params_size_to_scale'],
        'insert_1x1_conv_before_depthtospace'       : [],
        'convert_depth2space_to_reshp_tr_reshp'     : ['optimize_reshp_tr_reshp'],
        'convert_space2depth_to_reshp_tr_reshp'     : ['optimize_reshp_tr_reshp'],
        'convert_tanhgelu_to_erfgelu'               : ['break_gelu_to_components'],
        'support_broadcast_ops_constant_input'      : [],
        'remove_where_layer'                        : [],
        'convert_tr_conv_stride_n_tr_to_matmul'     : [],
        'optimize_reshp_tr_reshp'                   : [], 
        'eliminate_noop_slice'                      : [],
        'eliminate_unsqueeze'                       : [],
        'break_gelu_to_components'                  : [],
}

def get_optimizers(bucket_flags=None):
    """
    Default optimizers option list
    """
    opts = {
        # operation specific
        'add_input_normalization'                   : False,
        'convert_resize_params_size_to_scale'       : False,
        'convert_concat_axis_width_to_channel'      : False,
        'convert_maxpool_to_cascaded_maxpool'       : True,
        'convert_reducemean_to_matmul'              : True,
        'convert_gemm_to_matmul_and_add'            : True,
        'convert_matmul_to_conv_1x1s1'              : False,
        'convert_large_global_avg_pooling_to_matmul': True,
        'convert_gather_with_single_index_to_slice' : True,
        'convert_batchnorm_input_to_4D'             : True,
        'attention_block_optimization'              : False,
        'split_batch_dim_to_parallel_input_branches': False,
        'convert_softmax_axis_channel_to_width'     : True,
        'convert_softmax_axis_height_to_width'      : True,
        'push_large_channel_dim_to_height_for_width_wise_softmax': True,
        'convert_conv_large_pad_to_smaller_kernel'  : True,
        'expand_layernorm_to_component_ops'         : False, # Added support in import, no longer needed
        'push_matmul_channel_in_height'             : False,
        'expand_slice_across_multiple_axis'         : True,
        'convert_instancenorm_to_layernorm'         : False,
        'convert_unsqueeze_to_reshape'              : True,
        'add_bias_qdq'                              : False,
        'remove_quantize_initializer'               : True, 
        'remove_duplicate_quantize_dequantize'      : False, # not yet implemented 
        "convert_neg_to_mul"                        : True,
        "convert_expand_to_reshape_and_concat"      : True,
        "convert_single_concat_to_consecutive_concats" : True,
        'convert_conv_7x7_stride4_to_stride1'       : True,
        "convert_2_dimension_slice_to_maxpool"      : False,  # theoritically better than splitting in 2 axis
        "change_argmax_keepdims_to_1"               : True,
        'hf_attention_block_optimization'           : True,
        "convert_reducesum_to_matmul"               : True,
        'convert_resize_params_size_to_scale_dynamic_batch' : False, 
        'replace_mean_with_eltwise'                 : False, 
        'replace_sub_with_neg_add'                  : False, 
        'convert_conv_even_filter_to_odd'           : False, 
        'remove_duplicates'                         : False, 
        'remove_unity_resize'                       : False, 
        'insert_1x1_conv_before_depthtospace'       : False,
        'convert_depth2space_to_reshp_tr_reshp'     : True,
        'convert_space2depth_to_reshp_tr_reshp'     : True,
        'convert_tanhgelu_to_erfgelu'               : True,
        'support_broadcast_ops_constant_input'      : False, 
        'remove_where_layer'                        : True,
        'eliminate_noop_slice'                      : True,
        'eliminate_unsqueeze'                       : True,
        'break_gelu_to_components'                  : True, 
        'convert_tr_conv_stride_n_tr_to_matmul'     : True,
        'optimize_reshp_tr_reshp'                   : True,
        'hf_detr_attention_block_optimization'      : True,
        
        # utilities specific
        'shape_inference_mode'      : 'all',
        'simplify_mode'             : 'pre',
        'simplify_kwargs'           : {'skipped_optimizers': ['fuse_consecutive_concats']},
    }
    
    for bucket in BUCKETS:
        opts[bucket] = False
    
    if bucket_flags:
        # Set all individual optimizers to False
        for k in opt_ops:
            opts[k] = False
        # Set all bucket flags to False, then enable only those requested
        for b in BUCKETS:
            opts[b] = b in bucket_flags
    return opts


    
def test_optimizers():
    """
    Specify the individual optmizers that need to be tested
    """
    return {
        # operation specific to be specified here
        'hf_detr_attention_block_optimization' : True,

        # utilities specific
        'shape_inference_mode'      : 'all',
        'simplify_mode'             : 'pre',
        'simplify_kwargs'           : None
    }


class DependencyGraph:
    """
    Graph to represent the dependencied between different dependency functions

    * Each node is a function key string corresponding to the opt_ops dict
    * Edge A -> B suggests B optimization needs A optimization to run first i.e.
      There is dependency of A for B
    """
    def __init__(self, vertices: int, adj: Dict[str, List[str]]):
        self.num_vertices = vertices        # No. of vertices
        self.graph = adj                    # dictionary containing adjacency List


    def topological_sort_util(self , v: str, visited: Dict[str, bool], stack: List[str]):
        """
        A recursive function used by topologicalSort
        """

        # Mark the current node as visited.
        visited[v] = True

        # Recur for all the vertices adjacent to this vertex
        for u in self.graph[v]:
            if not visited[u]:
                self.topological_sort_util(u, visited, stack)

        # Push current vertex to stack which stores result
        stack.insert(0, v)

    def topological_sort (self) -> List[str]:
        """
        The function to do Topological Sort. It uses recursive
        topological_sort_util
        It returns the list of nodes sorted in topological
        sorted order
        """
        # Mark all the vertices as not visited
        # keys same as adj list and False value
        visited = dict(zip(self.graph.keys(), [False] * self.num_vertices))
        stack = list()

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for v in self.graph.keys():
            if not visited[v]:
                self.topological_sort_util(v, visited, stack)

        return stack


def get_topological_sorted_key_order ():
    """
    Construct a graph from the adjacency list
    return the key list in a topological sorted
    order
    """
    # construct the graph
    g = DependencyGraph(vertices= len(opt_ops), adj= adj_list)
    # return topo sorted order of keys
    return g.topological_sort()

class BucketDependencyGraph(DependencyGraph):
    def __init__(self, adj):
        self.graph = adj
        self.buckets = list(adj.keys())

    def topological_sort_util(self, v, visited, stack):
        visited[v] = True
        for u in self.graph[v]:
            if not visited[u]:
                self.topological_sort_util(u, visited, stack)
        stack.append(v)

    def topological_sort(self):
        visited = {k: False for k in self.buckets}
        stack = []
        for v in self.buckets:
            if not visited[v]:
                self.topological_sort_util(v, visited, stack)
        return stack

def get_topological_sorted_bucket_order():
    g = BucketDependencyGraph(BUCKET_ADJ_LIST)
    return g.topological_sort()