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

### custom imports
from .src.resize import tidl_modify_resize
from .src.attention import tidl_modify_attention
from .src.batch import tidl_modify_batch_dim
from .src.concat import tidl_modify_concat
from .src.maxpool import tidl_modify_maxpool
from .src.reducemean import tidl_modify_reducemean


### function definitions
opt_ops = {
        'attention': tidl_modify_attention,
        'batch': tidl_modify_batch_dim,
        'resize': tidl_modify_resize,
        'concat': tidl_modify_concat,
        'maxpool': tidl_modify_maxpool,
        'reducemean': tidl_modify_reducemean

}

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


    curr_op = 1
    for key, func in opt_ops.items():
        logging.info(f"[{curr_op}/{NUM_OPS}] {key.capitalize()} optimizations")
        func(graph, onnx_graph, args)
        # cleanup
        graph.cleanup().toposort()
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


def format_logger (log_level):
    """
    Format logger
    """

    logging.basicConfig(format='[%(levelname)s]:%(message)s')
    # colored logs
    yellow  = "\x1b[33;20m"
    red     = "\x1b[31;1m"
    reset   = "\x1b[0m"
    logging.addLevelName(logging.WARNING, yellow + logging.getLevelName(logging.WARNING) + reset)
    logging.addLevelName(logging.CRITICAL, yellow + logging.getLevelName(logging.WARNING) + reset)
    logging.addLevelName(logging.ERROR, red + logging.getLevelName(logging.ERROR) + reset)
    # set log level
    if log_level == "info":
        logging.getLogger().setLevel(logging.INFO)
    elif log_level == "debug":
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        print(f"Unknown log level {log_level}")


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
        'convert_reducemean'                        : False,
        # utilities specific
        'shape_inference_mode'      : 'all',
        'simplify_mode'             : None,
        'simplify_kwargs'           : None
    }


def optimize (model:str, out_model:str = None, verbose:bool= False, **kwargs):
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
    args = get_optimizers()
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
