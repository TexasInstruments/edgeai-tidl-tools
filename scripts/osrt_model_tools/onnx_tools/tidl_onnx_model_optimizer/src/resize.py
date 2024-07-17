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
Module containing Resize layer specific functions and optimizations
"""

import logging
import onnx_graphsurgeon as gs
import onnx
import numpy as np


def tidl_convert_resize_params_size_to_scale(graph: gs.Graph,
                                             onnx_graph: onnx.GraphProto) :
    """
    If some resize layer has size defined instead of scale,
    change to scale accordingly
    ------------------------------------------------------
    Assumes that the inputs are in following order
    [Variable Input, roi, scales, sizes]
    """
    tensors = graph.tensors()
    for node in graph.nodes:
        # check if satisfy criteria
        if node.op == "Resize":
            # if not 4 inputs, do not consider
            if len(node.inputs) != 4:
                continue
            var, roi, scales, sizes = node.inputs[0], node.inputs[1], node.inputs[2], node.inputs[3]
            # if sizes is not empty and scales are empty and both are constant
            if (not np.any(scales.shape)) and np.any(sizes.shape) \
                and isinstance(sizes, gs.Constant):
                reshaped_sizes = np.array(tensors[sizes.name].values, dtype=np.float32)
                in_sizes = np.array(var.shape, dtype=np.float32)
                scale_params = reshaped_sizes/in_sizes

                # check scale parameter values
                for t in scale_params:
                    if np.log2(t) != int(np.log2(t)):
                        logging.warning(f"{node.name} has scale not as power of "
                                        f"2 which is not supported by TIDL")
                scale_name = f"{node.name}.scales"
                scales_updated = gs.Constant(name=scale_name, values=scale_params)
                node.inputs = [var, roi, scales_updated]
                logging.debug(f"Updating resize node {node.name} inputs from "
                              f"sizes to scale {scale_params}")
            # endif
        # endif
    # endfor
