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

from .ops import opt_ops, get_optimizers, get_topological_sorted_key_order, qdq_supported_ops, expand_bucket_flags, get_topological_sorted_bucket_order, BUCKETS, BUCKET_ADJ_LIST
from .src.common import format_logger

NUM_OPS = len(opt_ops)


def print_node_count_table(model1:onnx.ModelProto, model2:onnx.ModelProto):
    
    try:
        import tabulate
    except:
        logging.warning("tabulate is not installed so no node count table will be generated!")
        return
    
    graph1 = gs.import_onnx(model1)
    graph2 = gs.import_onnx(model2)
    ops = []
    
    num1 = {}
    for node in graph1.nodes:
        num1[node.op] = num1.get(node.op, 0) + 1
        ops.append(node.op) if node.op not in ops else None
    
    num2 = {}
    for node in graph2.nodes:
        num2[node.op] = num2.get(node.op, 0) + 1
        ops.append(node.op) if node.op not in ops else None
    
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    
    headers = ["Operation", "Original Model", "Optimized Model"]
    ops = sorted(ops)
    
    data = [ ]
    for op in ops:
        n1 = num1.get(op,0)
        n2 = num2.get(op,0)
        data.append([op, n1, f"{RED if n2>n1 else (GREEN if n2<n1 else '')}{n2}{RESET if n1!=n2 else ''}"])

    table = tabulate.tabulate(data,headers,tablefmt="fancy_grid")
    logging.info(f'After optimization, node count table:\n{table}\n')

def get_bucket_for_opt(opt_name):
    for bucket in BUCKETS.keys():
        if opt_name in BUCKETS[bucket]:
            return bucket
    return None

def log_all_optimizations(args):
    """
    Log the status (enabled/disabled) of all optimizations in args.
    """
    for key, val in args.items():
        if key in opt_ops:
            status = "Enabled" if val else "Disabled"
            logging.info(f"Optimization {key}: {status}")

def enable_bucket_dependencies(args, bucket_order, bucket_adj_list):
    """
    For each enabled bucket in args, recursively enable all its dependency buckets.
    This ensures that if a bucket is enabled, all buckets it depends on are also enabled,
    so their optimizations will be executed in the correct order.
    """   
    for bucket in bucket_order:
        flag = bucket
        if args.get(flag, False):
            # Recursively enable dependencies
            stack = list(bucket_adj_list.get(bucket, []))
            while stack:
                dep = stack.pop()
                dep_flag = dep
                if not args.get(dep_flag, False):
                    args[dep_flag] = True
                    stack.extend(bucket_adj_list.get(dep, []))
                    

def run_optimizations(graph, onnx_graph, args, is_quantized_model, topo_sorted_keys, bucket_order, BUCKETS, mode="auto"):
    """
    Unified optimization runner for both bucket and individual modes.
    mode: "bucket", "individual", or "auto" (auto-detects from args)
    """
    curr_op = 1
    NUM_OPS = len(topo_sorted_keys)
    already_run = set()

    if mode == "auto":
        bucket_mode = any(args.get(bucket, False) for bucket in bucket_order)
        mode = "bucket" if bucket_mode else "individual"

    if mode == "bucket":
        # get (bucket, key) for all enabled buckets and their keys
        def key_iter():
            for bucket in bucket_order:
                val = args.get(bucket, False)
                if not val:
                    yield bucket, key, val
                for key in topo_sorted_keys:
                    if key in BUCKETS[bucket]:
                        yield bucket, key, val
    else:
        # get (bucket, key) for all enabled individual keys
        def key_iter():
            for key in topo_sorted_keys:
                val = args.get(key, False)
                yield get_bucket_for_opt(key), key, val

    for bucket, key, val in key_iter():
        if key in already_run:
            continue
        disabled_op = True
        if val and not is_quantized_model or (is_quantized_model and key in qdq_supported_ops):
            logging.info(f"[{curr_op}/{NUM_OPS}] {key.capitalize()} optimization (bucket: {bucket}) : Enabled")
            if isinstance(val, dict):
                kwargs = val
            else:
                kwargs = {}
            func = opt_ops[key]
            ret = func(graph, onnx_graph, **kwargs)
            if isinstance(ret, gs.Graph):
                logging.warning("Graph was updated within optimization function")
                graph = ret
            graph.cleanup().toposort()
            temp_model = gs.export_onnx(graph)
            temp_model = shape_inference.infer_shapes(temp_model, check_type=True, strict_mode=True)
            graph = gs.import_onnx(temp_model)
            disabled_op = False
        if disabled_op:
            logging.info(f"[{curr_op}/{NUM_OPS}] {key.capitalize()} optimization (bucket: {bucket}) : Disabled")
        curr_op += 1
        already_run.add(key)
    return graph


def tidl_modify(model_path: str, out_model_path: str, args: dict):
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

    is_quantized_model = any(node.op == "QuantizeLinear" for node in graph.nodes)
    topo_sorted_keys = get_topological_sorted_key_order()
    curr_op = 1
    for key, val in args.items():
        if key not in topo_sorted_keys:
            continue
        if args[key] not in (True, False, None) and not isinstance(args[key], dict):
            logging.warning(f"[{curr_op}/{NUM_OPS}] {key.capitalize()} optimization : Wrong input of type{type(args[key])}, defaulting to disabled")
            args[key] =False
            logging.warning(f"only value of True or False or None or a dict of arguments is supported")
        curr_op += 1
            
    bucket_order = get_topological_sorted_bucket_order()
    enable_bucket_dependencies(args, bucket_order, BUCKET_ADJ_LIST)
    bucket_mode = any(args.get(bucket, False) for bucket in bucket_order)
    

    if bucket_mode:
        graph = run_optimizations(graph, onnx_graph, args, is_quantized_model, topo_sorted_keys, bucket_order, BUCKETS)
    else:
        graph = run_optimizations(graph, onnx_graph, args, is_quantized_model, topo_sorted_keys, bucket_order, BUCKETS)

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
            
    print_node_count_table(model, out_model)
    
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

    expand_bucket_flags(args)
    
    # format logger
    format_logger(args['log_level'])


    # check for valid path
    if not os.path.isfile(model):
        logging.error(f"File {model} not found")
        sys.exit(-1)
    # set output model path
    model_name = model.split('/')[-1]
    out_model_path = os.path.join('/'.join(model.split('/')[:-1]) , f"optimized_{model_name}") if out_model is None else out_model
    # call main wrapper function
    tidl_modify(model_path= model, out_model_path= out_model_path, args= args)
    logging.info(f"Saved modified model at {out_model_path}")
