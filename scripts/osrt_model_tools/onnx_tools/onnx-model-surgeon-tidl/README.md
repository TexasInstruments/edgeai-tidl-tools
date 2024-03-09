# TI-ONNX-Graph-Optimizer
This module contains various modifications and optimization on the original ONNX graph and generates a new optimized ONNX model with the same functionality as the original model but with a more optimized structure for TIDL-RT inference.

## Setup
This script depends on [onnx-graphsurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon). You can install the requirements to run this script via:

          `pip install -r requirements.txt`

## Use
Execute the command below to perform model optimization and generate a modified onnx model

          `python3 onnx-model-surgeon-tidl.py --model=<input_model_path>`.

NOTE : *The input model is expected to have shape inference done*.


The above command will generate the output model in the same location as the input model with name `modified_<input_model_name>`. Some shapes are removed while making changes to graph structure, so shape inference has to be run on the output model before passing it to TIDL-RT.

One of the tools which can be used for shape inference is `onnxsim`, execute the command

          `onnxsim modified_<input model name> <final_model_name>`.

### NOTE
This module performs some optimizations on the model and one of the optimization is in early stage named as "batch specific optimization". This optimization changes a network with its partial structure with batch to multiple parallel branches in order to have TIDL-RT compatible structure. As of now the "batch specific optimization" is **experimental and at early stage** and require user to provide the start and end node names where the batch dimension needs to be replaced with parallel branches. (*Check batch.py for these two global variables named START_NODE_NAME and END_NODE_NAME*) In future support will be added to automatically detect these nodes and these variables will be removed.