# tidl-onnx-model-utils
This module contains various utilites for onnx models.

## Table of Contents
- [Setup](#setup)
- [Uses](#uses)
  - [optimize_model_input](#optimize_model_input)
  - [convert_model_to_yuv](#convert_model_to_yuv)
  - [convert_image_to_yuv](#convert_image_to_yuv)
  - [add_intermediate_outputs](#add_intermediate_outputs)
  - [extract_model](#extract_model)
  - [get_all_node_names](#get_all_node_names)
  - [create_batch_model](#create_batch_model)

## Setup
For setting up execute the command

    cd ../onnx_tools
    source ./setup.sh

This depends on the [onnx-graphsurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon)

## Uses
After running setup script, you will have `tidl_onnx_model_utils` python package installed in your python environment

### <u>optimize_model_input</u>

**Optimizes an ONNX model for TIDL inference by adding preprocessing operations (cast, add, multiply) to handle input normalization.**

Usually vision-based deep-learning model training the input image is normalized and resultant float input tensor is used as input for model. The float tensor would need 4 bytes (32-bit) for each element compared to 1 byte which is typical for camera sensor (unsigned 8-bit integer). We propose to update the model offline to accept 8-bit integer inputs and push the required normalization parameters as part of the model. This can be done using the `optimize_model_input` utility function provided.

```python
    from osrt_model_tools.onnx_tools.tidl_onnx_model_utils import optimize_model_input
    optimize_model_input(in_model_path, out_model_path, scale=[0.0078125, 0.0078125, 0.0078125], mean=[128.0, 128.0, 128.0])
```

- `in_model_path` : path to input ONNX model
- `out_model_path` : path to save the optimized ONNX model
- `scale` : list of scale values for each input channel (default: [0.0078125, 0.0078125, 0.0078125])
- `mean` : list of mean values for each input channel (default: [128.0, 128.0, 128.0])

This function modifies the model to:
1. Accept UINT8 input instead of FLOAT
2. Add preprocessing nodes that subtract mean values and apply scaling
3. Handle ArgMax output by casting to UINT8 if it's the final output

<div align="center">
<img src="../../../../docs/assets/onnx_model_opt.png" width="500">
</div>

The figure below shows the conversion of original model with float input to an updated model with 8-bit integer input. The operators inside the dotted box are additional operators added for mean and scale. This model is functionally the same as the original model, but requires lesser memory bandwidth compared original.


### <u>convert_model_to_yuv</u>

**Converts an RGB-trained ONNX model to accept YUV (NV12) image format as input.**

Sometimes a model which is trained with RGB data needs to be run with YUV data. During these scenarios we propose to update model offline to change its input from RGB to YUV.  This can be done using the `convert_model_to_yuv` utility function provided.

```python
    from osrt_model_tools.onnx_tools.tidl_onnx_model_utils import convert_model_to_yuv
    convert_model_to_yuv(model_path, output_path, input_names=None, mean=None, std=None, mode="YUV420SP")
```

- `model_path` : path to input ONNX model (RGB model)
- `output_path` : path to save the YUV-compatible ONNX model
- `input_names` : (optional) list of input names to convert in case of multiple inputs. Default: All inputs of model
- `scale` : (optional) list of scale values for each input channel. Default: None
- `mean` : (optional) list of mean values for each input channel. Default: None
- `mode` : (optional) YUV mode. Default: YUV420SP. Currently supported modes: ["YUV420SP"]

This function modifies the model to:
1. Accept separate Y and UV inputs instead of RGB
2. Add preprocessing nodes that convert YUV to RGB format
3. Apply any specified mean and scale normalization by calline optimize_model_input

<div align="center">
<img src="../../../../docs/assets/onnx_rgb_to_yuv.png" width="300">
</div>

> **Note:** This function currently only supports RGB input models. The convolution layer with name **Conv_YUV_RGB_*** handles the computation of converting the YUV to RGB. If your input is in another colorspace, you will need to update the `weights` and `bias` variables for the Convolution layer either offline or by modifying the code in the `convert_model_to_yuv` function in the `onnx_rgb_to_yuv_convertor.py` file and re-installing osrt-model-tools.

### <u>convert_image_to_yuv</u>

**Converts an image to YUV format and saves separate Y and UV components.**

```python
    from osrt_model_tools.onnx_tools.tidl_onnx_model_utils import convert_image_to_yuv
    convert_image_to_yuv(input_path, input_width, input_height)
```

- `input_path` : path to the input image (e.g., JPG file)
- `input_width` : width of the input image
- `input_height` : height of the input image

This function:
1. Uses ffmpeg to convert the input image to YUV (NV12) format
2. Extracts and saves the Y component as a binary file
3. Extracts and saves the UV component as a binary file

Note: This function requires ffmpeg to be installed on the system.


### <u>add_intermediate_outputs</u>

**Adds output layer to all nodes of onnx model. This is very useful for `debugging` layer level outputs.**

```python   
    from osrt_model_tools.onnx_tools.tidl_onnx_model_utils import add_intermediate_outputs
    add_intermediate_outputs(model_path, output_path)
```

- `model_path` : path to input ONNX model
- `output_path` : path to output ONNX modellayer

### <u>extract_model</u>

**Extracts a subgraph from an ONNX model based on specified input and output nodes. This is useful for creating smaller models or isolating specific parts of a model for debugging or optimization.**

```python   
    from osrt_model_tools.onnx_tools.tidl_onnx_model_utils import extract_model
    extract_model(model_path, output_path, input_names, output_names)
```

- `model_path` : path to input ONNX model
- `output_path` : path to save the extracted ONNX model
- `input_names` : list of input names to extract (these will be the inputs to the extracted model)
- `output_names` : list of output names to extract (these will be the outputs of the extracted model)

### <u>get_all_node_names</u>

**Returns all the nodes in the graph between start and the end nodes**

```python   
    from osrt_model_tools.onnx_tools.tidl_onnx_model_utils import get_all_node_names
    deny_list = get_all_node_names(model_path, start_end_layers, **kwargs)
```


- `model_path` : path to input ONNX model
- `start_end_layers` : dictionary of the start and end layer names, between which 
                            (including start and end node) needs to be added to deny list
                            if "None" is passed in the end node (values of dict), then the 
                            model output nodes are assumed as the end nodes


The above command will return a comma separated string of all the nodes between the start and the end nodes.

Example:

    start_end_layers = {
        '/bbox_head/Sigmoid' : ['/Cast_3', '/GatherElements_1'],
        '/bbox_head/offset_head/offset_head.0/Conv' : None,
        '/Div_2' : ['/Mul_5']
    }

If a particular node specified in end node is not present in the model, it will not return any nodes in that path. If we need to map till the end of the model, None needs to be specified as end_node.


### <u>create_batch_model</u>

**Modifies an ONNX model to support a specific batch dimension.**

```python
    from osrt_model_tools.onnx_tools.tidl_onnx_model_utils import create_batch_model
    create_batch_model(in_model_path, out_model_path, batch_dim)
```

- `in_model_path` : path to input ONNX model
- `out_model_path` : path to save the batch-enabled ONNX model
- `batch_dim` : integer specifying the desired batch dimension

This function updates the batch dimension (first dimension) of all input, output, and intermediate tensors in the model to the specified value.
