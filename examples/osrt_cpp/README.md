# CPP Examples
- [CPP Examples](#cpp-examples)
  - [Introduction](#introduction)
  - [Setup](#setup)
  - [Build](#build)
  - [Run](#run)
  - [Validation on Target](#validation-on-target)


## Introduction

   - CPP APIs os the DL runtime offered by solutions only supports the model inference. So the user is expeted  to run the [Python Examples](../../README.md#python-exampe) on PC to generate the model artifacts.
   - CPP API require yaml file reading lib. So the user is expected to install libyaml-cpp-dev by running command "sudo apt-get install libyaml-cpp-dev"
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
    ./bin/Release/ort_main -z "/path/to/artifacts/folder/od-8030_onnxrt_edgeai-mmdet_ssd-lite_mobilenetv2_fpn_512x512_20201110_model_onnx/" -v 1 -i "test_data/ADE_val_00001801.jpg" -l "test_data/labels.txt"
    ./bin/Release/dlr_main -z "/path/to/artifacts/folder/
    od-5020_tvmdlr_gluoncv-mxnet_yolo3_mobilenet1.0_coco-symbol_json/" -v 1 -i "test_data/ADE_val_00001801.jpg"  -l "test_data/labels.txt" -a 1 -d 1 -y "cpu"
    ./bin/Release/tfl_main -z "/path/to/artifacts/folder/od-2010_tflitert_mlperf_ssd_mobilenet_v2_300_float_tflite/" -v 1 -i "test_data/ADE_val_00001801.jpg" -l "test_data/labels.txt" -a 1 -d 1
    ```
## Validation on Target
- Build and runt steps remains same for PC emaultionn and target. Copy the below folders from PC to the EVM where this repo is cloned before ruunning the examples
  
    ```
    ./model-artifacts
    ./models
    ```



