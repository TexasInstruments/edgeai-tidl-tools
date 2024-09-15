# TIDLRT CPP Advanced Examples
- [TIDLRT CPP Examples](#tidlrt-cpp-examples)
  - [Introduction](#introduction)
  - [Setup](#setup)
  - [Test case addition](#test-case-addition)
  - [Build](#build)
  - [Run](#run)
  - [Validation on Target](#validation-on-target)


## Introduction
   - TIDL RT CPP APIs only supports the model inference for the models which can be fully offloaded to DSP. This specific advanced application is meant to test the TIDL feature capability of a higher priority model to pre-empt another lower priority model. This example requires user to run the [Python Examples](../../osrt_python/README.md#python-example) on PC to generate the model artifacts.
## Setup
- Prepare the Environment for the Model compilation by following the setup section [here](../../../README.md#setup)

## Test case addition
- "gPriorityMapping" is a vector of tests to be run by user. Each test can specify a vector of threads, with each thread hosting multiple models.
- To run a new test, add a new entry in the vector with model information as described by "model_input_info" struct.
- Application as it is uses the models "cl-ort-resnet18-v1" and "ss-ort-deeplabv3lite_mobilenetv2" which are compiled as part of default python compilation example models [Python Examples](../../osrt_python/README.md#python-example).

## Build 
  - Build the CPP examples using cmake from repository base directory
    ```
    mkdir build && cd build
    cmake ../examples/
    #If building with backward compatibility please use the below cmake command: 
    #cmake ../examples -DENABLE_SDK_9_2_COMPATIBILITY=1
    make -j 
    cd  ../
    ```

## Run 
  - Run the CPP examples using the below commands
    ```
    ./bin/Release/tidlrt_priority_scheduling
    ```
  - The current application is written with intention to do TIDL internal testing with a post processing result analysis. For a customer application, the result analysis can be disabled by setting option --disable_result_analysis ( -r ). Similarly duration of the pre-emption test(in minutes) can be specified with option --test_duration ( -t ) as follows:
  ```
  ./bin/Release/tidlrt_priority_scheduling -r 1 -t 1
  ```

## Validation on Target
- This application is meant to be tested only on target and not on PC emulation. Copy the below folders from PC to the EVM where this repo is cloned before running the application
  
    ```
    ./model-artifacts
    ```



