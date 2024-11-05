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
Top level api for calling as a package function
"""



import os
import sys
import logging
import onnx_graphsurgeon as gs
import onnx
from onnx import shape_inference
from onnxsim import simplify

from .ops import opt_ops, get_optimizers, get_topological_sorted_key_order, qdq_supported_ops
from .src.common import format_logger

NUM_OPS = len(opt_ops)

def tidl_modify (model_path: str, out_model_path: str, args: dict):
    """
    Wrapper function to modify the passed model network following standard TIDL
    specific constraints
    """
    model = onnx.load(model_path)

    # pre-processing simplification
    if args['shape_inference_mode'] in ["all", "pre"]:
        logging.info("Enabled pre-processing shape inference")
        model = shape_inference.infer_shapes(model, check_type= True, strict_mode= True)

    if args['simplify_mode'] in ["all", "pre"]:
        logging.info("Enabled pre-processing simplification")
        simplify_kwargs = args['simplify_kwargs']
        model, ok = simplify(model, **simplify_kwargs)
        if not ok:
            logging.error("Failed during simplification, aborting...")
            sys.exit(-1)

    onnx_graph = model.graph
    graph = gs.import_onnx(model)

    # check whether a quantized qdq model
    is_quantized_model = any(node.op == "QuantizeLinear" for node in graph.nodes)

    curr_op = 1
    topo_sorted_keys = get_topological_sorted_key_order()
    # logging.debug(topo_sorted_keys)
    for key in topo_sorted_keys:
        disabled_op = True
        if (key in args) and args[key]:
            if not(is_quantized_model) or (is_quantized_model and key in qdq_supported_ops):  
                logging.info(f"[{curr_op}/{NUM_OPS}] {key.capitalize()} optimization : Enabled")
                func = opt_ops[key]
                ret = func(graph, onnx_graph)
                if (type(ret) == gs.Graph):
                    #return an updated graph
                    logging.warning("Graph was updated within optimization function") #fixme
                    graph = ret
                # cleanup
                graph.cleanup().toposort()

                temp_model = gs.export_onnx(graph)
                temp_model = shape_inference.infer_shapes(temp_model, check_type= True, strict_mode= True)
                graph = gs.import_onnx(temp_model)
                disabled_op = False
        if disabled_op:
            logging.info(f"[{curr_op}/{NUM_OPS}] {key.capitalize()} optimization : Disabled")
        curr_op += 1


    # post processing simplification
    out_model = gs.export_onnx(graph)
    if args['shape_inference_mode'] in ["all", "post"]:
        logging.info("Enabled post-processing shape inference")
        out_model = shape_inference.infer_shapes(out_model, check_type= True, strict_mode= True)

    if args['simplify_mode'] in ["all", "post"]:
        logging.info("Enabled post-processing simplification")
        simplify_kwargs = args['simplify_kwargs']
        out_model, ok = simplify(out_model, **simplify_kwargs)
        if not ok:
            logging.error("Failed during simplification, aborting...")
            sys.exit(-1)

    # svae to output path
    onnx.save(out_model, out_model_path)
    

def optimize (model:str, out_model:str = None, verbose:bool= False, custom_optimizers:dict=None, **kwargs):
    """
    Main function
    ---------------------------------------------------------
    Inputs
    ---------------------------------------------------------
    model:                  path to input ONNX model
    out_model:              path to output ONNX model (optional).
                            If not given, saved in same place as the input model
                            with a default name (optimized_<input_model_name>)
    shape_inference_mode:   (pre/post/all/None) flag to use onnx shape inference
                            [pre: run only before graph surgeon optimization,
                            post:run only after graph surgeon optimization,
                            all (default): both pre and post are enabled,
                            None: both disabled]
    simplify_mode:          (pre/post/all/None) flag to use onnxsim simplification
                            [pre : simplify only before graph surgeon
                            optimizations, post:simplify only after graph
                            surgeon optimization, all: both pre and post are
                            enabled, None (default): both disabled]
     ---------------------------------------------------------------
    Output
    ---------------------------------------------------------------
    Empty
    """
    # argument parsing
    args = get_optimizers() if custom_optimizers is None else custom_optimizers
    args['log_level'] = "debug" if verbose else "info"
    for key, val in kwargs.items():
        args[key] = val

    # format logger
    format_logger(args['log_level'])


     # check for valid path
    if not os.path.isfile(model):
        logging.error(f"File {model} not found")
        sys.exit(-1)
    # set output model path
    model_name = model.split('/')[-1]
    out_model_path = '/'.join(model.split('/')[:-1]) + f"/optimized_{model_name}" if out_model is None else out_model
    # call main wrapper function
    tidl_modify(model_path= model, out_model_path= out_model_path, args= args)
    logging.info(f"Saved modified model at {out_model_path}")
