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
from typing import List, Dict


# importing all opt functions
from .src.resize import tidl_convert_resize_params_size_to_scale
from .src.attention import tidl_optimize_attention
from .src.batch import tidl_modify_batch_dim
from .src.concat import tidl_convert_concat_axis_width_to_channel
from .src.maxpool import tidl_convert_maxpool_to_cascaded_maxpool
from .src.reducemean import tidl_convert_reducemean_to_matmul
from .src.gemm import tidl_convert_gemm_to_matmul_and_add
from .src.matmul import tidl_convert_matmul_to_conv_1x1s1
from .src.global_avg_pool import tidl_convert_large_global_avg_pooling_to_matmul
from .src.gather import tidl_convert_gather_with_single_index_to_slice
from .src.batchnorm import tidl_convert_batchnorm_input_to_4D
from .src.softmax import tidl_convert_softmax_axis_channel_to_width, tidl_convert_softmax_axis_height_to_width
from .src.softmax import tidl_push_large_channel_dim_to_height_for_width_wise_softmax
from .src.conv import tidl_convert_conv_large_pad_to_smaller_kernel


### function dict to execute
opt_ops = {
        'convert_resize_params_size_to_scale'       : tidl_convert_resize_params_size_to_scale,
        'attention_block_optimization'              : tidl_optimize_attention,
        'convert_concat_axis_width_to_channel'      : tidl_convert_concat_axis_width_to_channel,
        'split_batch_dim_to_parallel_input_branches': tidl_modify_batch_dim,
        'convert_maxpool_to_cascaded_maxpool'       : tidl_convert_maxpool_to_cascaded_maxpool,
        'convert_reducemean_to_matmul'				: tidl_convert_reducemean_to_matmul,
        'convert_gemm_to_matmul_and_add'            : tidl_convert_gemm_to_matmul_and_add,
        'convert_matmul_to_conv_1x1s1'              : tidl_convert_matmul_to_conv_1x1s1,
        'convert_large_global_avg_pooling_to_matmul': tidl_convert_large_global_avg_pooling_to_matmul,
        'convert_gather_with_single_index_to_slice' : tidl_convert_gather_with_single_index_to_slice,
        'convert_batchnorm_input_to_4D'             : tidl_convert_batchnorm_input_to_4D,
        'convert_softmax_axis_channel_to_width'     : tidl_convert_softmax_axis_channel_to_width,
        'convert_softmax_axis_height_to_width'      : tidl_convert_softmax_axis_height_to_width,
        'push_large_channel_dim_to_height_for_width_wise_softmax': tidl_push_large_channel_dim_to_height_for_width_wise_softmax,
        'convert_conv_large_pad_to_smaller_kernel'  : tidl_convert_conv_large_pad_to_smaller_kernel,
}


# adjancency list
adj_list = {
        'convert_resize_params_size_to_scale'       : [],
        'attention_block_optimization'              : [],
        'convert_concat_axis_width_to_channel'      : [],
        'split_batch_dim_to_parallel_input_branches': [],
        'convert_maxpool_to_cascaded_maxpool'       : [],
        'convert_reducemean_to_matmul'				: [],
        'convert_gemm_to_matmul_and_add'            : [],
        'convert_matmul_to_conv_1x1s1'              : ['convert_gemm_to_matmul_and_add'     # don't want the matmul from gemm to change
                                                       ],
        'convert_large_global_avg_pooling_to_matmul': [],
        'convert_gather_with_single_index_to_slice' : [],
        'convert_batchnorm_input_to_4D'             : [],
        'convert_softmax_axis_channel_to_width'     : [],
        'convert_softmax_axis_height_to_width'      : [],
        'push_large_channel_dim_to_height_for_width_wise_softmax': [],
        'convert_conv_large_pad_to_smaller_kernel'  : [],
}

def get_optimizers():
    """
    Default optimizers option list
    """
    return {
        # operation specific
        'convert_resize_params_size_to_scale'       : True,
        'convert_concat_axis_width_to_channel'      : False,
        'attention_block_optimization'              : False,
        'split_batch_dim_to_parallel_input_branches': False,
        'convert_maxpool_to_cascaded_maxpool'       : False,
        'convert_reducemean_to_matmul'              : False,
        'convert_gemm_to_matmul_and_add'            : True,
        'convert_matmul_to_conv_1x1s1'              : True,
        'convert_large_global_avg_pooling_to_matmul': True,
        'convert_gather_with_single_index_to_slice' : True,
        'convert_batchnorm_input_to_4D'             : True,
        'convert_softmax_axis_channel_to_width'     : False,
        'convert_softmax_axis_height_to_width'      : False,
        'push_large_channel_dim_to_height_for_width_wise_softmax': True,
        'convert_conv_large_pad_to_smaller_kernel'  : False,

        # utilities specific
        'shape_inference_mode'      : 'all',
        'simplify_mode'             : None,
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
