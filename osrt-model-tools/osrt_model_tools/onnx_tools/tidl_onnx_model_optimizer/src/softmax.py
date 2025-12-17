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


def tidl_convert_softmax_unsupported_axis_to_width(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    Convert Softmax operations to TIDL-supported dimensions (width=-1, height=-2).
    Unsupported axes are transformed to width using Transpose→Softmax→Transpose pattern.
    """
    opset_version= graph.opset

    if opset_version is None:
        logging.warning("Cannot determine the Opset version")
        default_axis= -1
    
    if opset_version >= 13:
        default_axis= -1
        logging.debug(f"Detected ONNX Opset {opset_version}: Softmax default axis = -1")
    else:
        default_axis = 1
        logging.debug(f"Detected ONNX Opset {opset_version}: Softmax default axis = 1")

    SUPPORTED_AXES = [-1,-2]
    TARGET_AXIS = -1

    softmaxes = [node for node in graph.nodes if node.op == "Softmax"]

    converted_count=0
    skipped_count=0
    already_supported_count=0

    for idx, softmax in enumerate(softmaxes):

        softmax_axis = softmax.attrs.get("axis", default_axis)

        if softmax.inputs[0].shape is None:
            logging.warning(f"Softmax '{softmax.name}' input has no shape information. Skipping.")
            skipped_count+=1
            continue

        inp_shape=softmax.inputs[0].shape
        num_dims = len(inp_shape)

        # Check minimum dimensions
        if num_dims < 2 :
            logging.warning(f"Softmax '{softmax.name}' input has only {num_dims} dimension(s).")
            skipped_count+=1
            continue

        # Normalize axis to positive
        if softmax_axis < 0:
            normalized_axis = num_dims + softmax_axis
        else:
            normalized_axis = softmax_axis
        
        # Validate range
        if normalized_axis < 0 or normalized_axis >= num_dims:
            logging.error(
                f"Softmax '{softmax.name}' has invalid axis {softmax_axis} "
                f"for {num_dims}D input. Valid range: [{-num_dims}, {num_dims-1}]. Skipping."
            )
            skipped_count += 1
            continue
        
        normalized_supported =[]
        for ax in SUPPORTED_AXES:
            if ax < 0: 
                norm_ax = num_dims +ax
            else:
                norm_ax = ax
            
            normalized_supported.append(norm_ax)
        
        # Check if already supported
        if normalized_axis in normalized_supported:
            already_supported_count += 1
            continue

        #convert to target axis
        target_axis_normalized = num_dims + TARGET_AXIS

        # swapping dimension
        perm = list(range(num_dims))
        perm[normalized_axis], perm[target_axis_normalized] = perm[target_axis_normalized], perm[normalized_axis] 

        # new shape
        new_shape = list(inp_shape)
        new_shape[normalized_axis], new_shape[target_axis_normalized] = new_shape[target_axis_normalized], new_shape[normalized_axis]

        logging.debug(f"  → Permutation: {perm}")
        logging.debug(f"  → Shape: {inp_shape} → {new_shape} → {inp_shape}")
        
        #---------------------------------------------------------------
        # Create Transpose 1 (Move axis to width)
        #---------------------------------------------------------------
        transpose1_output= gs.Variable( 
                                    name=f"sf_transpose_out.1.{idx}",
                                    dtype=np.float32,
                                    shape=new_shape
        )

        transpose1 = gs.Node(
            op="Transpose",
            name=f"{softmax.name}_transpose_1.{idx}",
            attrs={"perm": perm},
            inputs=softmax.inputs,
            outputs=[transpose1_output]
        )
        graph.nodes.append(transpose1)
        logging.debug(f"  → Added Transpose1 '{transpose1.name}'")
        
        #---------------------------------------------------------------
        # Modify Softmax to use width axis
        #---------------------------------------------------------------
        old_softmax_outputs = copy.copy(softmax.outputs)

        softmax_output = gs.Variable(
            name=f"{softmax.name}_out.{idx}",
            dtype=np.float32,
            shape=new_shape
        )

        softmax.inputs = [transpose1_output]
        softmax.outputs = [softmax_output]
        softmax.attrs['axis'] = TARGET_AXIS

        logging.debug(f"  → Modified Softmax to use axis={TARGET_AXIS}")

        #---------------------------------------------------------------
        # Create Transpose2 (restore original layout)
        #---------------------------------------------------------------
        transpose2 = gs.Node(
            op="Transpose",
            name=f"sf_transpose_2.{idx}",
            attrs={"perm": perm},  # Same perm reverses the operation
            inputs=[softmax_output],
            outputs=old_softmax_outputs
        )
        graph.nodes.append(transpose2)
        logging.debug(f"  → Added Transpose2 '{transpose2.name}'")
        
        converted_count += 1


def tidl_convert_softmax_axis_height_to_width(graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    The SoftMax layer with operation in the height dimension is replaced with
    Transpose -> SoftMax -> Transpose to satisfy constraint of SoftMax layer only
    occuring in width dimension
    """
    softmaxes = [node for node in graph.nodes if node.op == "Softmax"]
    logging.warning("This optimization is deprecated and will be removed in a future.")

    for idx, softmax in enumerate(softmaxes):
        # Dimension in which Softmax should occur
        softmax_dimension = softmax.attrs["axis"] if 'axis' in softmax.attrs else -1
        if softmax.inputs[0].shape is not None:
            inp_shape = softmax.inputs[0].shape
            #TODO modify code to use this instead and check
        else:
            logging.info(f"{softmax.inputs[0].name} softmax layer does not have input shape, disabling the onnx optimization.")
            continue

        # Assumes tensor height axis, with NxCxHxW, CxHxW, HxW order
        if len(softmax.inputs[0].shape) >= 2:

            # If softmax op occurs across height
            if softmax_dimension == (len(softmax.inputs[0].shape) - 2):
                logging.debug(f"Converting axis for layer {softmax.name} from dimension {softmax_dimension} to {len(softmax.inputs[0].shape) - 1}")

                # Permutation array
                perm = list(range(len(softmax.inputs[0].shape)))
                temp = perm[-1]
                perm[-1] = perm[-2]
                perm[-2] = temp

                # New output shape from transpose1
                new_shape = copy.copy(softmax.inputs[0].shape)
                temp = new_shape[-1]
                new_shape[-1] = new_shape[-2]
                new_shape[-2] = temp

                var_outshape   = [gs.Variable(f"sf_transpose_out.1.{idx}",
                                                dtype=np.float32, shape=new_shape)]

                # Create transpose node to swap height to width
                transpose1 = gs.Node(op="Transpose", name=f"sf_transpose_1.{idx}",
                                        attrs={"perm": perm}, inputs=softmax.inputs,
                                        outputs=var_outshape)
                graph.nodes.append(transpose1)
                logging.debug(f"Adding transpose layer {transpose1.name} with perm {perm}")

                # Modify softmax layer
                softmax.inputs = transpose1.outputs
                old_softmax_outputs = copy.copy(softmax.outputs)
                softmax.outputs = [gs.Variable(f"sf_softmax_out.{idx}",
                                                dtype=np.float32, shape=new_shape)]

                # Create transpose node to swap width to height
                transpose2 = gs.Node(op="Transpose", name=f"sf_transpose_2.{idx}",
                                        attrs={"perm": perm}, inputs=softmax.outputs,
                                        outputs=old_softmax_outputs)
                graph.nodes.append(transpose2)
                logging.debug(f"Adding transpose layer {transpose1.name} with perm {perm}")
                softmax.attrs['axis']=-1
        else:
            logging.critical(f"{softmax.inputs[0].name} input to {softmax.name} has no height dim"
                                         "Unable to convert axis to height")



TIDL_SOFTMAX_LARGE_DIM_THRESHOLD        = 1024
TIDL_GENERIC_SCRATCH_VOLUME             = 32 * 1024
TIDL_SOFTMAX_OPTIMIAL_HEIGHT_DIM        = 64

def tidl_push_large_channel_dim_to_height_for_width_wise_softmax (graph: gs.Graph, onnx_graph: onnx.GraphProto):
    """
    When a softmax has high value of dimensions channel and upper
    it performs unoptimal. But reshaping the shape to have a larger
    height can make it more efficient
    """
    softmax_nodes = [node for node in graph.nodes if node.op == "Softmax"]

    for node in softmax_nodes:
        # must be widthwise
        inp = node.inputs[0]
        axis = node.attrs['axis'] if 'axis' in node.attrs else -1
        if inp.shape is not None:
            inp_shape = inp.shape
            #TODO modify code to use this instead and check
        else:
            logging.info(f"{node.name} softmax layer does not have input shape, disabling the onnx optimization.")
            continue

        if axis in ((len(inp.shape) - 1), -1):
            # must have large channel dim value
            if (len(inp.shape) > 2) and (inp.shape[-3] > TIDL_SOFTMAX_LARGE_DIM_THRESHOLD):
                c, h, w = inp.shape[-3], inp.shape[-2], inp.shape[-1]
                tot_vol = c * h * w
                ch_lower_lim = int(np.ceil(tot_vol / TIDL_GENERIC_SCRATCH_VOLUME))
                logging.debug(f"Lower limit for channel shape = {ch_lower_lim}")
                # calculate ratio to reshape
                new_shape = None
                opt_ratio = 0.5
                logging.debug(f"Searching for optimal shape (base ratio = {opt_ratio})...")
                for new_channel in range(ch_lower_lim, c, 1):
                    new_height = (c*h)/new_channel
                    # check if height is integer
                    if new_height == int(new_height):
                        # check vector optimization
                        t = new_height / TIDL_SOFTMAX_OPTIMIAL_HEIGHT_DIM
                        t = t - int(t)
                        if t > opt_ratio:
                            opt_ratio = t
                            new_shape = inp.shape[:-3] + [new_channel, int(new_height), w]
                            logging.debug(f"Found better optimization {t} with shape {new_shape}")

                # add reshapes before
                reshp_out = gs.Variable(name= f'{inp.name}_Opt_Reshape_out', dtype= np.float32)
                reshp_shape = gs.Constant(name= f'{inp.name}_Opt_Reshape_shape',
                                          values= np.array(new_shape, dtype= np.int64))
                reshp = gs.Node(name= f'{inp.name}_Opt_Reshape', op= 'Reshape',
                                inputs= [inp, reshp_shape], outputs= [reshp_out])

                # add reshape node
                logging.debug(f"Adding reshape node {reshp.name} to reshape input to"
                              f"{new_shape}")
                graph.nodes.append(reshp)


                # add reshape after
                node.inputs[0] = reshp_out
                node_reshaped_out = gs.Variable(name= f'{node.name}_Reshaped_out', dtype= np.float32)
                reshp_shape = gs.Constant(name= f'{node.name}_Opt_Reshape_shape',
                                          values= np.array(inp.shape, dtype= np.int64))
                reshp = gs.Node(name= f'{node.name}_out_Reshape', op= 'Reshape',
                                inputs= [node_reshaped_out, reshp_shape], outputs= node.outputs)

                # add reshape node
                logging.debug(f"Adding reshape node {reshp.name} to reshape output to"
                              f"{inp.shape}")
                graph.nodes.append(reshp)

                # change original softmax output
                node.outputs = [node_reshaped_out]
