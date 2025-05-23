# TIDLRT CPP Examples
- [TIDLRT CPP Examples](#tidlrt-cpp-examples)
  - [Introduction](#introduction)
  - [Setup](#setup)
  - [Build](#build)
  - [Run](#run)
  - [Validation on Target](#validation-on-target)


## Introduction
   - TIDL RT CPP APIs only supports the model inference for the models which can be fully offloaded to DSP. The user is expected  to run the [Python Examples](../osrt_python/README.md#python-example) on PC to generate the model artifacts.
  - The model artifacts is expected to be generated with the following advanced options:
    1. "advanced_options:add_data_convert_ops": 3
  - For model compiled using ONNXRT, the order of outputs generated from TIDL-RT run can be obtained from 
    `onnxrtMetaData.txt` file which can be found in the compiled model artifacts folder. The `outDataNames` defines the order of output.
## Setup
- Prepare the Environment for the Model compilation by following the setup section [here](../../README.md#setup)

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
    ./bin/Release/tidlrt_clasification -l test_data/labels.txt -i test_data/airshow.jpg  -f model-artifacts/cl-tfl-mobilenet_v1_1.0_224/artifacts/ -d 1
    ```
## Validation on Target
- Build and run steps remains same for PC emulation and target. Copy the below folders from PC to the EVM where this repo is cloned before running the examples
  
    ```
    ./model-artifacts
    ./models
    ```



