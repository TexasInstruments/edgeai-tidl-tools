# Advanced examples : RGB model to YUV model conversion

## Introduction 

The intention this advanced example folder is to describe te steps on how to convert a RGB trained model to accept YUV data as input. 

**Disclaimer** : These scripts are mainly written/maintained for debug/development validation and may not receive active support in case of issues. Also all example unit test cases may not be necessarily supported on TIDL

## Usage

The usage is specified for tfite runtime, follow corresponding steps for ONNX runtime

1. Perform the setup steps are required for setting up the python examples

2. Follow the steps to generate YUV data from input jpg image

```
cd edgeai-tidl-tools/scripts/osrt_model_tools/onnx_tools 
python3 RGB_YUV_model_converter.py -i <input_image_path>  -g 1 -w <width> -l <height>
```
eg:
```
python3 RGB_YUV_model_converter.py -i edgeai-tidl-tools/test_data/ADE_val_00001801.jpg -w 224 -l 224 -g 1
```

3. Follow the steps to generate YUV model from a RGB trained model

```
cd edgeai-tidl-tools/scripts/osrt_model_tools/tflite_tools #for tflite models
cd edgeai-tidl-tools/scripts/osrt_model_tools/onnx_tools #for onnx models
python3 RGB_YUV_model_converter.py -i <input_model_path> -o <output_model_path>
```

4. Run the python example with YUV input
```
cd edgeai-tidl-tools/examples/osrt_python/advanced_examples/RGB_YUV_model_conversion/ort # for onnx
cd edgeai-tidl-tools/examples/osrt_python/advanced_examples/RGB_YUV_model_conversion/tfl # for tfl
python3 onnxrt_ep.py -c
python3 onnxrt_ep.py 
```