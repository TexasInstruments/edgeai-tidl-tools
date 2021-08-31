# CPP Examples
- [CPP Examples](#cpp-examples)
  - [Introduction](#introduction)
  - [Setup](#setup)
  - [Build](#build)
  - [Run](#run)
  - [Validation on Target](#validation-on-target)


## Introduction

   - CPP APIs os the DL runtime offered by solutions only supports the model inference. So the user is expeted  to run the [Python Examples](../../README.md#python-exampe) on PC to generate the model artifacts.
> Note : We are plannign to clean-up and unify the user inetrface for CPP examples by next release. We are also planning to add more CPP exmaples.

## Setup
- Prepare the Environment for the Model compilation by follwoing the setup section [here](../../README.md#setup)


## Build 
  - Build the CPP examples using cmake from repository base directory
    ```
    mkdir build && cd build
    cmake ../examples/
    make
    cd  ../
    ```

## Run 
  - Run the CPP examples using the below commands
    ```
    ./bin/Release/tfl_clasification -m models/public/tflite/mobilenet_v1_1.0_224.tflite -l test_data/labels.txt -i test_data/airshow.jpg  -f model-artifacts/tfl/mobilenet_v1_1.0_224/ -a 1 -d 1 -c 100
    ./bin/Release/ort_clasification  test_data/airshow.jpg models/public/onnx/resnet18_opset9.onnx model-artifacts/ort/resnet18-v1/ test_data/labels.txt -t
    ./bin/Release/dlr_clasification  -m model-artifacts/dlr/onnx_mobilenetv2/ -i test_data/airshow.jpg -n input.1
    ```
## Validation on Target
- Build and runt steps remains same for PC emaultionn and target. Copy the below folders from PC to the EVM where this repo is cloned before ruunning the examples
  
    ```
    ./model-artifacts
    ./models
    ```



