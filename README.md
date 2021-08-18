# Edgeai TIDL Tools and Examples - Work Under progress. Will be Released Soon

This repository contains example developed for Deep learning runtime (DLRT) offering provided by TI’s edge AI solutions. This repository also contains tools that can help in deploying AI applications on TI’s edgeai solutions quickly to achieve most optimal performance.

## Steps to Run Python Exampes

1. Run Setup.sh to install all the dependencies. This would download and intall all the python packages amd PC tools required for compiling models for TI device
2. Set below to environement vairables. If you are building complete SDK from source, then thes path can be set as per the SDK intall paths

```
export TIDL_TOOLS_PATH=/base_repo_path/tidl_tools
export LD_LIBRARY_PATH=$TIDL_TOOLS_PATH 
```
3. Run below script to compile and validate the models in PC
```
./scripts/run_python_examples.sh
```
## Steps to Run CPP Exampes
1. Need to run the Python examples first to generat artifacts for CPP examples.

2. Build the CPP examples using cmake
```
mkdir examples/build
cd examples/build
cmake ../
make
cd  ../../
```

3. Run the CPP examples using the below commands
```
./examples/bin/Release/tfl_clasification -m models/public/tflite/mobilenet_v1_1.0_224.tflite -l test_data/labels.txt -i test_data/airshow.jpg  -f examples/osrt_python/tfl/tflrt-artifacts/mobilenet_v1_1.0_224/ -a 1 -d 1
./examples/bin/Release/ort_clasification  test_data/airshow.jpg models/public/onnx/resnet18_opset9.onnx examples/osrt_python/ort/onnxrt-artifacts/resnet18-v1/ test_data/labels.txt -t
./examples/bin/Release/dlr_clasification  -m examples/osrt_python/tvm_dlr/onnx_mobilenetv2/ -i test_data/airshow.jpg -n input.1
```
