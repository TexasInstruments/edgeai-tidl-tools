# onnx-tools
This module consists of two different packages: 

### [tidl_onnx_model_optimizer](tidl_onnx_model_optimizer/README.md)

Consists of various modifications and optimizations on the original ONNX graph and generates a new optimized ONNX model with the same functionality as the original model, making it more suitable for TIDL inference.

### [tidl_onnx_model_utils](tidl_onnx_model_utils/README.md)

Contains various utility functions for ONNX such as getting the nodes between two nodes, pruning the names of the ONNX model, extracting subgraphs, adding intermediate outputs to the model, RGB to YUV input format converter, optimizing model inputs etc.

## Setup
For setting up and installing osrt-model-tools, execute the command:

    source ./setup.sh
