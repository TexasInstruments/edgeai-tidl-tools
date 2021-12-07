# Scripts for Model Optimization and Validation

- [Scripts for Model Optimization and Validation](#scripts-for-model-optimization-and-validation)
  - [Model Optimization](#model-optimization)
  - [Examples](#examples)

## Model Optimization

During vision-based DL model training the input image is normalized and resultant float input tensor is used as input for model. The float tensor would need 4 bytes (32-bit) for each element compared to 1 byte of the element from camera sensor output which is unsigned 8-bit integer.  We propose to update model offline to change this input to 8-bit integer and push the required normalization parameters as part of the model. This figure 6 shows the example of such original model with float input and an updated model with 8-bit integer. The operators inside the dotted box are additional operators. This model is functionally exactly same as original but would require less memory bandwidth compared original. The additional operators also would be merged into the following convolution layer to reduce overall DL inference latency.  

![Image Normalization Optimization](../../docs/tidl_model_opt.png)

## Examples

1. Optimize TFLite modles	
```
python3 tflite_add_nodes.py --model_path ../../models/public/tflite/mobilenet_v1_1.0_224.tflite --model_path_out ../../models/public/tflite/mobilenet_v1_1.0_224_opt.tflite
python3 tflite_add_nodes.py --model_path ../../models/public/tflite/ssd_mobilenet_v2_300_float.tflite --model_path_out ../../models/public/tflite/ssd_mobilenet_v2_300_float_opt.tflite
python3 tflite_add_nodes.py --model_path ../../models/public/tflite/deeplabv3_mnv2_ade20k_float.tflite --model_path_out ../../models/public/tflite/deeplabv3_mnv2_ade20k_float_opt.tflite
```

2. Optimize ONNX modles	
```
python3 onnx_add_nodes.py --model_path=../../models/public/onnx/deeplabv3lite_mobilenetv2.onnx  --model_path_out=../../models/public/onnx/deeplabv3lite_mobilenetv2_opt.onnx
python3 onnx_add_nodes.py --model_path=../../models/public/onnx/resnet18_opset9.onnx --model_path_out=../../models/public/onnx/resnet18_opset9_opt.onnx
python3 onnx_add_nodes.py --model_path=../../models/public/onnx/ssd-lite_mobilenetv2_fpn.onnx --model_path_out=../../models/public/onnx/ssd-lite_mobilenetv2_fpn_opt.onnx

```
