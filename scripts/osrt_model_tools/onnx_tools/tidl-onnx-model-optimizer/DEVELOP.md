# Developer's Guideline

This document describes how you can add your own optimization function to TIDL ONNX Model Optimizer library.


## Creating a function
You have to write a function that contains the optimization you are intending to do.

### Prototype

`def tidl_abc ( graph: gs.Graph, onnx_graph: onnx.GraphProto):`

`abc` should clearly describe what the function is doing in brief. For e.g., `tidl_convert_resize_params_size_to_scale` is name of the function that changes the sizes input of a Resize layer to appropriate scales.

### Where to put your function
In the following section, changes have to be made inside `tidl_onnx_model_optimizer/src` directory.

#### Case - I
If you are writing a function which handles a layer/block which is already present, you are expected to keep the function in the .py file existing for that layer/block. For e.g., if you are adding a optimization function on resize layer, you are supposed to put `tidl_abc` function in `resize.py`.

#### Case - II
If you are creating a function for a completely new layer/block, you will have to create a new .py file, which should be named as `<layer>.py` or `<block>.py`. Put your function in this file.


## Interface
Make the following changes in the file `tidl_onnx_model_optimizer/ops.py`

### Control Flag
You will require a new flag for enable/disbale control over your optimization. For that purpose you have to change the dict returned in the function `get_optimizers` to have a new entry `'abc'`. You can make the default value here True/False as per your need.

### Dependency graph
Your function might need to be run strictly after some other existing optimization function and before some functions. For e.g., say you are converting a unoptimal MatMul to Conv, you want all the optimizations which converts other layers to MatMul to run before this (as then you don't have to run your function multiple number of times).

1. Add a new entry `'abc': []` in the dict `adj_list`.
2. For any other key, `k`, if you need your function to run before function corresposding to k, modify your entry as `'abc': [k]`. Keep adding to this list like `[k1, k2, k3, ...]` for as many functions you need.
3. If you want your funtion to strictly run after some other function corresposnding to key `k`, modify the entry for `k` as `k: [..., 'abc']`

Please add a single line comment justfying your reason of adding a dependency, as these are good when a strict ordering of functions are necessary but costs time when there are lot of functions i.e., nodes in the dependency graph.

### Call to implemented function
Finally you need to change the dict variable `opt_ops` if you have added a new .py file.
Add a entry `"abc" : tidl_abc` and voila! You are done.


## Good Practices
For good practices ensure to use the logging library facility and add useful debug logs wherever you seem necessary 🙂.
Also please try to run pylint on the code as this codebase has been developed with pylint coding guidelines. If you are using pylint in your VS Code, you can add these settings to ensure consistency:
```
"pylint.args": [
        "--disable=E1101,W0613,W1203,E0401,W1201,W1514",
        "--max-line-length=160"
    ],
```