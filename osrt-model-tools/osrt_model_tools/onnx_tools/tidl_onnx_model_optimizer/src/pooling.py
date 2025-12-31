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
ONNX pooling operator transformations: MaxPool/AveragePool -> ReduceMax/ReduceMean for global patterns, and large kernel decomposition.
"""
import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np

def tidl_convert_global_pooling_to_reduce_ops( graph : gs.Graph, onnx_graph:onnx.GraphProto):
    """
    Convert MaxPool/AveragePool with kernel==stride==input_size to ReduceMax/ReduceMean.

    kernel==stride, no padding, kernel matches input spatial dimensions.
    """

    opset = graph.opset  
    pool_nodes = [node for node in graph.nodes if node.op in ["MaxPool", "AveragePool"]]
    
    for pool_node in pool_nodes:
        kernel_shape = pool_node.attrs.get("kernel_shape")
        strides = pool_node.attrs.get("strides")
        pads = pool_node.attrs.get("pads", [0, 0, 0, 0])
        
        # Skip if missing attributes or kernel != stride
        if not kernel_shape or not strides or kernel_shape != strides:
            continue
        
        # Skip if has padding
        if pads != [0, 0, 0, 0]:
            logging.debug(f"Skipping '{pool_node.name}': has padding {pads}")
            continue
        
        # Get input shape
        if not pool_node.inputs or not pool_node.inputs[0].shape:
            continue
        
        input_shape = pool_node.inputs[0].shape
        input_h, input_w = input_shape[-2], input_shape[-1]
        
        # Skip if dynamic dimensions or kernel doesn't match input
        if not isinstance(input_h, int) or not isinstance(input_w, int):
            continue
        
        if kernel_shape[0] != input_h or kernel_shape[1] != input_w:
            continue
        
        # Convert to Reduce operation
        original_op = pool_node.op
        reduce_op = "ReduceMax" if pool_node.op == "MaxPool" else "ReduceMean"
        
        # Modify the node in-place
        pool_node.op = reduce_op
        pool_node.name = f"{pool_node.name}_{reduce_op}"
        pool_node.attrs.clear()
        
        if opset >= 13:
            # Opset 13+: axes as INPUT (Constant)
            axes_constant = gs.Constant(
                name=f"{pool_node.name}_axes",
                values=np.array([-2, -1], dtype=np.int64)
            )
            pool_node.inputs.append(axes_constant)
            pool_node.attrs["keepdims"] = 1
        else:
            # Opset 1-12: axes as ATTRIBUTE
            pool_node.attrs["axes"] = [-2, -1]
            pool_node.attrs["keepdims"] = 1
        
        logging.debug(f"Converted '{pool_node.name}' ({original_op}) to {reduce_op}")