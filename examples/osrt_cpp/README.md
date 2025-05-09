# CPP Examples
- [CPP Examples](#cpp-examples)
  - [Introduction](#introduction)
  - [Setup](#setup)
  - [Build](#build)
  - [Run](#run)
  - [Validation on Target](#validation-on-target)
  - [Running pre-compiled model from modelzoo](#running-pre-compiled-model-from-modelzoo)


## Introduction

   - CPP APIs for DL runtimes only support model inference. So the user is expected to run the [Python Examples](../../README.md#python-exampe) on PC to generate the model artifacts.
   - CPP API require yaml file reading lib. So the user is expected to install libyaml-cpp-dev by running command "sudo apt-get install libyaml-cpp-dev"


## Setup
- Prepare the Environment for the Model compilation by following the setup section [here](../../README.md#setup)


## Build 
  - Build the CPP examples using cmake from repository base directory. Create a build folder for your generated build files
  
    ```
    mkdir build && cd build
    ```
  - Below tables depicts the flags required for different compilation mode and different target device use these with cmake inside build directory.
  - During cmake build following paths are expected at $HOME(to override default path use cmake flags during cmake build).
      - TENSORFLOW_INSTALL_DIR : defaults check at ~/tensorflow 
      - ONNXRT_INSTALL_DIR: defaults check at ~/onnxruntime
      - DLR_INSTALL_DIR: defaults check at ~/neo-ai-dlr
      - OPENCV_INSTALL_DIR: defaults check at ~/opencv-4.1.0
      - ARMNN_PATH: defaults check at ~/armnn
      - TARGET_FS_PATH: defaults check ~/targetfs
      - CROSS_COMPILER_PATH: defaults check ~/arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu
    ```
    cmake -DFLAG1=val -DFLAG2=val ../examples
    make -j2
    cd ../
    ```


    | HOST CPU        | TARGET CPU           | TARGET DEVICE  | CMAKE OPTIONS  |
    | ------- |:------:| :-----:|:------------:|
    | X86      | X86 | AM62 | TARGET_DEVICE=am62 |
    | X86      | X86 | J7 | <none> |
    | X86      | ARM | AM62 | TARGET_DEVICE=am62<br> TARGET_CPU=arm |
    | X86      | ARM | J7 |  TARGET_CPU=arm |
    | ARM      | ARM | AM62 | TARGET_DEVICE=am62|
    | ARM      | ARM | J7 | 

  - Example: for cross compiling cpp examples for am62 device on x86 host
    ```
    cmake -DTARGET_DEVICE=am62 TARGET_CPU=arm ../examples
    make -j2
    cd ../
    ```
  - To compile ARMNN delegate option in AM62 native on AM62 device
    ```
    cmake -DTARGET_DEVICE=am62  -DARMNN_ENABLE=1../examples    
    ```
  - To compile ARMNN delegate option in AM62 cross compile on x86 device
    ```
    cmake -DTARGET_DEVICE=am62 TARGET_CPU=arm -DARMNN_ENABLE=1../examples
    ```
  - Note: If during cross-compiling/native-compiling cpp aplication for J7 device error appears "onnxruntime library not found" create symbolic for the same 
    ```
    cd <target_fs>/usr/lib
    ln -s libonnxruntime.so.1.7.0 libonnxruntime.so
    ```

## Run 
  - Run the CPP examples using the below commands
    ```
    ./bin/Release/ort_main -f model-artifacts/cl-ort-resnet18-v1/artifacts  -i test_data/airshow.jpg
    ./bin/Release/tfl_main -f model-artifacts/cl-tfl-mobilenet_v1_1.0_224/artifacts -i test_data/airshow.jpg
    ./bin/Release/dlr_main -f model-artifacts/cl-dlr-tflite_inceptionnetv3/artifacts  -i test_data/airshow.jpg
    ./bin/Release/dlr_main -f model-artifacts/cl-dlr-onnx_mobilenetv2/artifacts  -i test_data/airshow.jpg
    ./bin/Release/ort_main -f model-artifacts/od-ort-ssd-lite_mobilenetv2_fpn/artifacts -i test_data/ADE_val_00001801.jpg
    ./bin/Release/tfl_main -f model-artifacts/od-tfl-ssd_mobilenet_v2_300_float/artifacts -i test_data/ADE_val_00001801.jpg
    ./bin/Release/tfl_main -f model-artifacts/ss-tfl-deeplabv3_mnv2_ade20k_float/artifacts -i test_data/ADE_val_00001801.jpg
    ```
  - Available cmd line arguments for cpp
    - -h : for help
    - -i : input image path
    - -c : number of iteration to run for model
    - -l: labels path for classification model
    - -y: device_type for dlr models can be cpu,gpu
    - -a : accelarate [0|1|2|3]
      - 0 : None valid on all device
      - 1 : TIDL valid on J7 and x86 in runtimes(tfl and onnx)
      - 2 : XNN valid on Am62 and x86(Am62 config) in runtimes(tfl only)
      - 3 : ARMNN valid on Am62 in runtimes(tfl only)

    - -v : verbose (set to 1) 

## Validation on Target
- Build and run steps remains same for PC emulation and target. Copy the below folders from PC to the EVM where this repo is cloned before ruunning the examples
    ```
    ./model-artifacts
    ./models
    ```
- For ONNX runtime export the following on device, prior to execution:
    ```
    export TIDL_RT_ONNX_VARDIM=1
    ```
  
## Running pre-compiled model from modelzoo
- To run precomiled model from model zoo run the follwoing commands( as an example: cl-0000_tflitert_mlperf_mobilenet_v1_1.0_224_tflite)
- Fetch the tar link from model zoo and wget the file
- create a dir for untarring the files
- move the downloaded tar to created folder and then untar
- run the model 
  
    ```
    wget http://software-dl.ti.com/jacinto7/esd/modelzoo/08_00_00_05/modelartifacts/8bits/cl-0000_tflitert_mlperf_mobilenet_v1_1.0_224_tflite.tar.gz
    mkdir cl-0000_tflitert_mlperf_mobilenet_v1_1.0_224_tflite
    mv cl-0000_tflitert_mlperf_mobilenet_v1_1.0_224_tflite.tar.gz cl-0000_tflitert_mlperf_mobilenet_v1_1.0_224_tflite
    cd cl-0000_tflitert_mlperf_mobilenet_v1_1.0_224_tflite
    tar -xvf cl-0000_tflitert_mlperf_mobilenet_v1_1.0_224_tflite
    cd ../
    ./bin/Release/tfl_main -z "cl-0000_tflitert_mlperf_mobilenet_v1_1.0_224_tflite/" -v 1 -i "test_data/airshow.jpg" -l "test_data/labels.txt" -a 1 -d 1
    ```
- To run on target , copy the below folders from PC to the EVM where this repo is cloned before running the examples
    ```
    ./model-artifacts
    ./models
    ```


