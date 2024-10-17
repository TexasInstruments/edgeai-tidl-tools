# Advanced examples : RGB model to YUV model conversion

## Introduction 

The intention of this example is to describe the steps on how to convert a RGB trained model to accept YUV data as input. 

You can refer to [Extending support for other colorspaces section](../../../../scripts/README.md#extending-support-for-other-colorspaces) for more information on handling different color spaces

## Usage

1. Perform the setup steps as required (if not done already) for setting up the python examples by following these [steps](../../../../README.md#setup)

2. Follow the steps to generate YUV data from input jpg image

```bash
cd edgeai-tidl-tools/scripts/osrt_model_tools/onnx_tools 
python3 RGB_YUV_model_converter.py -g -i <input_image_path> -w <width> -l <height>
```
eg:
```bash
python3 RGB_YUV_model_converter.py -g -i edgeai-tidl-tools/test_data/ADE_val_00001801.jpg -w 224 -l 224
```

3. Follow the steps to generate YUV model from a RGB trained model
```bash
cd edgeai-tidl-tools/scripts/osrt_model_tools/tflite_tools #for tflite models
cd edgeai-tidl-tools/scripts/osrt_model_tools/onnx_tools/tidl_onnx_model_utils #for onnx models
python3 RGB_YUV_model_converter.py -m <input datalayout> -i <input_model_path> -o <output_model_path>
```
eg:
```bash
python3 RGB_YUV_model_converter.py -m YUV420SP -i edgeai-tidl-tools/models/public/mobilenet_v1_1.0_224.onnx -o edgeai-tidl-tools/models/public/mobilenet_v1_1.0_224_yuv.onnx
```

4. Run the python example with YUV input
```bash
cd edgeai-tidl-tools/examples/osrt_python/advanced_examples/RGB_YUV_model_conversion/ort # for onnx
python3 onnxrt_ep.py -c
python3 onnxrt_ep.py 
cd edgeai-tidl-tools/examples/osrt_python/advanced_examples/RGB_YUV_model_conversion/tfl # for tfl
python3 tflrt_delegate.py -c
python3 tflrt_delegate.py 
```

5. Sometimes the model may have multiple inputs coming from different sources. With this flag you can define specific inputs to convert into YUV 

> Note: This feature is currently supported for onnx models 

```bash
python3 RGB_YUV_model_converter.py -m <mode> -i <input model path> -o <output model path> --input_names <input node names> 
```

eg:
```bash
python3 RGB_YUV_model_converter.py -m YUV420SP -i multiple_input_model.onnx -o resnet_yuv.onnx --input_names input.1 input.5 
```

6. Mean and Std deviation of the input can also included into the model as per the [Model Optimization](../../../../scripts/README.md#model-optimization), with the **--mean** and **--std** flags

```bash
python3 RGB_YUV_model_converter.py -m <mode> -i <input model path> -o <output model path> --mean <space seperaed mean values> --std <space seperated std values>
```
eg:
```bash
python3 RGB_YUV_model_converter.py -m YUV420SP -i multiple_input_model.onnx -o resnet_yuv.onnx --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225
```
