# Copyright (c) {2015 - 2021} Texas Instruments Incorporated
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
Main module providing facility to optimize onnx graph with TIDL-RT specific
contraints
"""


import os
import sys
import logging
import argparse
import onnx_graphsurgeon as gs
import onnx

### custom imports
from src.resize import tidl_modify_resize
from src.attention import tidl_optimize_attention_blocks
from src.batch import tidl_modify_batch_dim


### function definitions

def tidl_modify (model_path: str, out_model_path: str, **kwargs):
    """
    Wrapper function to modify the passed model network following standard TIDL
    specific constraints
    """
    onnx_graph = onnx.load(model_path).graph
    graph = gs.import_onnx(onnx.load(model_path))

    # parse additional args
    is_transformer = False
    for key, value in kwargs.items():
        if (key == "transformer") and value:
            is_transformer = True

    tidl_modify_resize(graph, onnx_graph)
    tidl_modify_batch_dim(graph, onnx_graph)
    if is_transformer:
        tidl_optimize_attention_blocks(graph, onnx_graph)
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), out_model_path)


def main ():
    """
    Main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required= True, help= 'path to onnx model')
    parser.add_argument('--log_level', choices=["info", "debug"], default="info",
                        help='log level [1/2]')
    parser.add_argument('--transformer', action='store_true',
                        help= "flag to enable transformer specific optimizations")
    args = parser.parse_args()
    # set log level
    logging.basicConfig(format='[%(levelname)s]:%(message)s')
    if args.log_level == "info":
        logging.getLogger().setLevel(logging.INFO)
    elif args.log_level == "debug":
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        print(f"Unknown log level {args.log_level}")

     # check for valid path
    if not os.path.isfile(args.model):
        logging.error(f"File {args.model} not found")
        sys.exit(-1)
    # set output model path
    model_name = args.model.split('/')[-1]
    out_model_path = '/'.join(args.model.split('/')[:-1]) + f"/modified_{model_name}"
    # call main wrapper function
    tidl_modify(model_path= args.model, out_model_path= out_model_path, transformer=args.transformer)
    logging.info(f"Saved modified model at {out_model_path}")

if __name__ == "__main__":
    main()
