# tidl-onnx-model-utils
This module contains various utilites for onnx models.

## Setup
For setting up execute the command

    cd ../onnx_tools
    source ./setup.sh

This depends on the [onnx-graphsurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon)

## Uses
After running setup script, you will have `tidl_onnx_model_utils` python package installed in your python environment, which can be used as follows:

    from tidl_onnx_model_utils import get_all_node_names

    deny_list = get_all_node_names(model_path, start_end_layers, **kwargs)

Input arguments to `get_all_node_names`

    model_path:             path to input ONNX model
    start_end_layers:       dictionary of the start and end layer names, between which 
                            (including start and end node) needs to be added to deny list
                            if "None" is passed in the end node (values of dict), then the 
                            model output nodes are assumed as the end nodes


The above command will return a comma separated string of all the nodes between the start and the end nodes.

An example of start_end_layers dictionary can be:

    start_end_layers = {
        '/bbox_head/Sigmoid' : ['/Cast_3', '/GatherElements_1'],
        '/bbox_head/offset_head/offset_head.0/Conv' : None,
        '/Div_2' : ['/Mul_5']
    }

If a particular node specified in end node is not present in the model, it will not return any nodes in that path. If we need to map till the end of the model, None needs to be specified as end_node. 