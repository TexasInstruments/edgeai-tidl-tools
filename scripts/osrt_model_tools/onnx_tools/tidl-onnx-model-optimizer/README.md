# tidl-onnx-model-optimizer
This module contains various modifications and optimization on the original ONNX graph and generates a new optimized ONNX model with the same functionality as the original model but with a more optimized structure for TIDL-RT inference.

## Setup
For setting up execute the command

    source ./setup.sh

This depends on the [onnx-graphsurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon)

## Use
After running setup script, you will have `tidl_onnx_model_optimizer` python package installed in your python environment, which can be used as follows:

    from tidl_onnx_model_optimizer import optimize

    optimize(input_model_path, output_model_path, **kwargs)

Input arguments to `optimize`

    model:                  path to input ONNX model
    out_model:              path to output ONNX model (optional).
                            If not given, saved in same place as the input model
                            with a default name (optimized_<input_model_name>)
    transformer:            (enable/disable) flag to enable/disable transformer
                            optimization (default: disable)
    batch:                  (enable/disable) flag to enable/disable batch input
                            specific optimizations (default: disable)
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


The above command will generate the output model in the same location as mentioned. Some shapes are removed while making changes to graph structure, so shape inference has to be run on the output model before model compilation



### NOTE
1. This module performs some optimizations on the model and one of the optimization is in early stage named as "batch specific optimization". This optimization changes a network with its partial structure with batch to multiple parallel branches in order to have TIDL-RT compatible structure. As of now the "batch specific optimization" is **experimental and at early stage** and require user to provide the start and end node names where the batch dimension needs to be replaced with parallel branches. (*Check batch.py for these two global variables named START_NODE_NAME and END_NODE_NAME*) In future support will be added to automatically detect these nodes and these variables will be removed.
2. Two optimization rules provided in (RGB_YUV_model_converter.py and onnx_model_opt.py) placed  at one directory above shall be combined with this module in future, but currently can be continued to be used as independent optimization scripts 