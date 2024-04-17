# Developer's Guideline

This document describes how you can add your own optimization function to TIDL ONNX Model Optimizer library.


## Creating a function
You have to write a function that contains the optimization you are intending to do. T

### Prototype

`def tidl_abc ( graph: gs.Graph, onnx_graph: onnx.GraphProto):`

`abc` should clearly describe what the function is doing in brief. For e.g., `tidl_convert_resize_params_size_to_scale` is name of the function that changes the sizes input of a Resize layer to appropriate scales.

### Where to put your function
In the following section we are making changes inside `tidl_onnx_model_optimizer/src` directory.

#### Case - I
If you are writing a function which handles a layer/block which is already present, you are expected to keep the function in the .py file existing for that layer/block. For e.g., if you are adding a optimization function on resize layer, you are supposed to put `tidl_abc` function in `resize.py`.

In this case you will need to add the function call to the wrapper function present in the .py file. For e.g., in `resize.py` you will have to add the call to `tidl_abc{...)}` call inside `tidl_modify_resize` wrapper function. You will have to add under the conditional flag `args['abc']` which will be added in the interfacing part. Please add logging debug prints to indicate triggering of this specific function. For logging print formats, please refer to existing prints.

#### Case - II
If you are creating a function for a completely new layer/block, you will have to create a new .py file, which should be named as `<layer>.py` or `<block>.py`. Every such .py file will have a wrapper global function as the first function :

`def tidl_modify_<layer/block> (graph: gs.Graph, onnx_graph: onnx.GraphProto, args: dict):`

Once you create this wrapper function, please add your own specific optimization function as mentioned in Case-I.


## Interfacing
Make the following changes in `tidl_onnx_model_optimizer/optimize.py`

### Control Flag
You will need a new flag for enable/disbale control over your optimization. For that purpose you have to change the dict returned in the function `get_optimizers`. You can make the default value here True/False as per your need.

### Call to implemented function
Finally in the function `tidl_modify` you will need to make some changes if you have added a new .py file.
Add a call to the wrapper function you added in your .py file , `tidl_modify_<layer/block>` and pass arguments following the other existing function calls. Please don't forget to change `NUM_OPS` and add info prints similar to existing ones before the function call for a nice console log 🙂.