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
Module containing MaxPool layer specific functions and optimizations
"""
import logging
import copy
import onnx_graphsurgeon as gs
import onnx
import numpy as np


def tidl_convert_maxpool_to_cascaded_maxpool(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Convert large MaxPool kernels (>3x3) to cascaded 3x3 and 2x2 layers.

    Supports :
        - Odd kernels (5x5, 7x7, 9x9...) with stride=1 or 2
        - Even kernels (4x4, 6x6, 8x8...) with stride=2 ONLY
    
    Skips:
        - Even kernels with stride=1
        - Stride > 2
        - 2x2 with stride=1
        - Non-default values for optional attributes auto_pad, ceil_mode, dilation
    """
    max_pools = [node for node in graph.nodes if node.op == "MaxPool"]
    
    for maxpool in max_pools:
        kernel_size = maxpool.attrs["kernel_shape"][0]
        orig_stride = maxpool.attrs["strides"][0]
        orig_pads = maxpool.attrs.get("pads", [0, 0, 0, 0])
        orig_padding = orig_pads[0]
        
        logging.debug(f"Checking MaxPool '{maxpool.name}': kernel={kernel_size}x{kernel_size}, stride={orig_stride}, padding={orig_padding}")

        # Check auto_pad (if present, must be "NOTSET")
        if "auto_pad" in maxpool.attrs:
            auto_pad = maxpool.attrs["auto_pad"]
            if auto_pad != "NOTSET":
                logging.warning(f"MaxPool '{maxpool.name}': auto_pad='{auto_pad}' not supported (only 'NOTSET'). Skipping.")
                continue
        
        # Check ceil_mode (if present, must be 0)
        if "ceil_mode" in maxpool.attrs:
            ceil_mode = maxpool.attrs["ceil_mode"]
            if ceil_mode != 0:
                logging.warning(f"MaxPool '{maxpool.name}': ceil_mode={ceil_mode} not supported (only 0). Skipping.")
                continue
        
        # Check dilations (if present, must be [1, 1])
        if "dilations" in maxpool.attrs:
            dilations = maxpool.attrs["dilations"]
            if dilations != [1, 1]:
                logging.warning(f"MaxPool '{maxpool.name}': dilations={dilations} not supported(only [1, 1]). Skipping.")
                continue
        
        # Skip if kernel size is rectangular
        if maxpool.attrs["kernel_shape"][0] != maxpool.attrs["kernel_shape"][1]:
            logging.warning(f"Rectangular kernel({maxpool.attrs['kernel_shape']}) is not supported")
            continue
        
        # Skip if already compatible
        if kernel_size <= 3:
            if kernel_size == 2 and orig_stride == 1:
                logging.warning(f"MaxPool '{maxpool.name}': 2x2 with stride=1 not supported. Skipping.")
                continue
            logging.debug(f"Already compatible (kernel <= 3x3)")
            continue
        
        # Validate stride (ONLY 1 or 2 supported)
        if orig_stride not in [1, 2]:
            logging.warning(f"MaxPool '{maxpool.name}': Stride {orig_stride} not supported.")
            continue
        
        # Check if conversion is possible with EXACT RF
        is_even_kernel = (kernel_size % 2 == 0)
        
        if is_even_kernel and orig_stride == 1:
            logging.warning(f"MaxPool {maxpool.name}: {kernel_size}×{kernel_size} kernel with stride=1. ")
            continue
        
        # Conversion strategy
        if is_even_kernel:
            num_3x3_layers = (kernel_size - 2) // 2
            use_final_2x2 = True
            total_layers = num_3x3_layers + 1
            logging.debug(f"Converting '{maxpool.name}': {num_3x3_layers}x(3x3) + 1x(2x2) → RF={kernel_size}")
        else:
            num_3x3_layers = (kernel_size - 1) // 2
            use_final_2x2 = False
            total_layers = num_3x3_layers
            logging.debug(f"Converting '{maxpool.name}': {num_3x3_layers}x(3x3) → RF={kernel_size}")
        
        # Create stride sequence
        if orig_stride == 1:
            stride_sequence = [1] * total_layers
        else:  # orig_stride == 2
            stride_sequence = [1] * (total_layers - 1) + [2]
        
        logging.debug(f"Stride sequence: {stride_sequence}")

        # Save original outputs
        saved_outputs = copy.copy(maxpool.outputs)
        output_dtype = saved_outputs[0].dtype if saved_outputs[0].dtype is not None else np.float32

        # Only include attributes that were present in the original node
        base_attrs = {}
        
        # Copy optional attributes if they were present
        if "auto_pad" in maxpool.attrs:
            base_attrs["auto_pad"] = maxpool.attrs["auto_pad"]  # Will be "NOTSET"
        
        if "ceil_mode" in maxpool.attrs:
            base_attrs["ceil_mode"] = maxpool.attrs["ceil_mode"]  # Will be 0
        
        if "dilations" in maxpool.attrs:
            base_attrs["dilations"] = maxpool.attrs["dilations"]  # Will be [1, 1]
        
        if "storage_order" in maxpool.attrs:
            base_attrs["storage_order"] = maxpool.attrs["storage_order"]  # Any value is OK
        
        
        # Modify first layer 
        first_layer_attrs = base_attrs.copy()
        first_layer_attrs["kernel_shape"] = [3, 3]
        first_layer_attrs["strides"] = [stride_sequence[0], stride_sequence[0]]
        first_layer_attrs["pads"] = orig_pads

        maxpool.attrs.clear()
        maxpool.attrs.update(first_layer_attrs)
        
        if total_layers == 1:
            logging.debug(f"Converted successfully: 1 layer, RF={kernel_size}")
            continue
        
        # Create intermediate output for first layer
        intermediate_output = gs.Variable(
            name=f"{saved_outputs[0].name}_layer0",
            shape=None,
            dtype=output_dtype
        )
        maxpool.outputs = [intermediate_output]
        current_input = [intermediate_output]
        
        # Add intermediate 3x3 layers (Layer 1 to Layer num_3x3_layers-1)
        for i in range(1, num_3x3_layers):
            # Determine if this is the final layer
            is_final_layer = (i == total_layers - 1)
            
            if is_final_layer and not use_final_2x2:
                layer_output = saved_outputs
            else:
                layer_output = [gs.Variable(
                    name=f"{saved_outputs[0].name}_layer{i}",
                    shape=None,
                    dtype=output_dtype
                )]
            
            # Build attributes for this layer
            layer_attrs = base_attrs.copy()
            layer_attrs["kernel_shape"] = [3, 3]
            layer_attrs["strides"] = [stride_sequence[i], stride_sequence[i]]
            layer_attrs["pads"] = [0, 0, 0, 0] 
            
            new_node = gs.Node(
                op="MaxPool",
                name=f"{maxpool.name}_layer{i}",
                attrs=layer_attrs,
                inputs=current_input,
                outputs=layer_output
            )
            graph.nodes.append(new_node)
            current_input = layer_output
        
        # Add final 2x2 layer (only for even kernels)
        if use_final_2x2:
            final_attrs = base_attrs.copy()
            final_attrs["kernel_shape"] = [2, 2]
            final_attrs["strides"] = [stride_sequence[-1], stride_sequence[-1]]
            final_attrs["pads"] = [0, 0, 0, 0]
            
            final_node = gs.Node(
                op="MaxPool",
                name=f"{maxpool.name}_final",
                attrs=final_attrs,
                inputs=current_input,
                outputs=saved_outputs
            )
            graph.nodes.append(final_node)
        
        logging.debug(f"Converted successfully: {total_layers} layers, RF={kernel_size}")



