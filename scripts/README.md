- [Scripts for Model Optimization and Validation](#scripts-for-model-optimization-and-validation)
  - [Model Optimization](#model-optimization)
- [Scripts for RGB to YUV Model conversion](#scripts-for-rgb-to-yuv-model-conversion)
  - [Extending Support for other Colorspaces](#extending-support-for-other-colorspaces)
  - [RGB_YUV_model_converter](#rgb-to-yuv-model-converter) 

# Scripts for Model Optimization and Validation

## Model Optimization

During vision-based DL model training the input image is normalized and resultant float input tensor is used as input for model. The float tensor would need 4 bytes (32-bit) for each element compared to 1 byte of the element from camera sensor output which is unsigned 8-bit integer.  We propose to update model offline to change this input to 8-bit integer and push the required normalization parameters as part of the model. Below figure shows the example of such original model with float input and an updated model with 8-bit integer. The operators inside the dotted box are additional operators. This model is functionally exactly same as original but would require less memory bandwidth compared original. The additional operators also would be merged into the following convolution layer to reduce overall DL inference latency.  

This optimization is included by default in the Model compilation script in this repository. This is done during model download step.

![Image Normalization Optimization](../docs/images/tidl_model_opt.png)

# Scripts for RGB to YUV Model conversion

## RGB to YUV model converter

Sometimes a model which is trained over RGB data need to be run with YUV data. During these scenarios we propose to update model offline to change its input from RGB to YUV. we provide scripts to do this. Script to convert TFlite model can be found [here](osrt_model_tools/tflite_tools/RGB_YUV_model_converter.py) and for onnx model can be found [here](osrt_model_tools/onnx_tools/tidl_onnx_model_utils/RGB_YUV_model_converter.py) Below figure shows the example of such original model with RGB converted to a model which takes YUV input. The operators inside the box are additional operators added to perform this task. 

![RGB_YUV_model_converter](../docs/images/Resnet_YUV420SP.png) 

One can use [examples](../examples/osrt_cpp/advanced_examples) as a reference to convert a RGB model to YUV model.

### Extending support for other colorspaces

The conv with name **Conv_YUV_RGB_\*** handles the computation of converting the YUV to RGB.  Similarly if your input is in different format, you can change the weights of the conv layer in 

> edgeai-tidl-tools/scripts/osrt_model_tools/onnx_tools/tidl_onnx_model_utils/RGB_YUV_model_converter.py

```python
def addYUVConv(in_model_path, out_model_path, args):
    ...
    # adding conv to convert YUV to RGB
    weights = [1.164, 0.0, 1.596,
                1.164, -0.391, -0.813,
                1.164, 2.018, 0.0 ]
    bias= [-222.912, 135.488, -276.928]
    ...
```

### Usage

To add the YUV input functionality:

```bash
python3 RGB_YUV_model_converter.py -m YUV420SP -i resnet.onnx -o resnet_yuv.onnx -w 224 -l 224
```

To generate the YUV input
```bash
python3 RGB_YUV_model_converter.py -g -i image.jpg -w 224 -l 224
```

> The following options are currently supported only for onnx models

Sometimes the model may have multiple inputs coming from different sources. With this flag you can define specific inputs to convert into YUV

```bash
python3 RGB_YUV_model_converter.py -m YUV420SP -i multiple_input_model.onnx -o resnet_yuv.onnx --input_names input.1 input.5 
```

The output of the conversion will be in the range of (0, 255) you can change this range with the help of **--normalize** flag

```bash
python3 RGB_YUV_model_converter.py -m YUV420SP -i multiple_input_model.onnx -o resnet_yuv.onnx --input_names input.1 input.5 --normalize 255.0
```

Mean and Std deviation of the input can also included into the model as per the [Model Optimization](#model-optimization), with the **--mean** and **--std** flags

```bash
python3 RGB_YUV_model_converter.py -m YUV420SP -i multiple_input_model.onnx -o resnet_yuv.onnx --normalize 255.0 --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225
```
