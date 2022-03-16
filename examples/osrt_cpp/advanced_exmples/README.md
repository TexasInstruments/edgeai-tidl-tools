# CPP Advanced Examples
- [CPP Advanced Examples](#cpp-advanced-examples)
  - [Introduction](#introduction)
  - [Setup](#setup)
  - [Build](#build)
  - [Run](#run)
  - [Validation on Target](#validation-on-target)
  - [Running pre-compiled model from modelzoo](#running-pre-compiled-model-from-modelzoo)


## Introduction

   - CPP Advanced APIs os the DL runtime offered by solutions only supports the model inference. So the user is expeted  to run the [Python Examples](../../README.md#python-exampe) on PC to generate the model artifacts.
   - CPP Advanced API require yaml file reading lib. So the user is expected to install libyaml-cpp-dev by running command "sudo apt-get install libyaml-cpp-dev"
> Note : We are plannign to clean-up and unify the user inetrface for CPP Advanced examples by next release. We are also planning to add more CPP exmaples.

## Setup
- Prepare the Environment for the Model compilation by follwoing the setup section [here](../../README.md#setup)


## Build 
  - Build the CPP Advanced examples using cmake from repository base directory
    ```
    mkdir build && cd build
    cmake ../examples/
    make
    cd  ../
    ```

## Run 
  - Run the advanced application with -h flag to see supported command line argument.
  ```
    ./bin/Release/tfl_main_adv -h
    ./bin/Release/ort_main_adv -h
  ```
  - Run the CPP Advanced examples using the below commands
    ```
    ./bin/Release/tfl_main_adv -i test_data/ADE_val_00001801.jpg test_data/airshow.jpg -m  model-artifacts/ss-tfl-deeplabv3_mnv2_ade20k_float/ model-artifacts/
    cl-tfl-mobilenet_v1_1.0_224/ -p 1 0 -t 1  -a 1  -d 1 -e 3
     ./bin/Release/ort_main_adv -i test_data/ADE_val_00001801.jpg test_data/airshow.jpg -m model-artifacts/ss-ort-deeplabv3lite_mobilenetv2/ model-artifacts/cl-ort-resnet18-v1/  -p 1 0 -t 1  -a 1  -d 1 -e 3
    ```
  - The above commands will run tfl and ort advance appliation.
  - Command line argument specifies:
    - -m model1_path model2_path
    - -p 1 0 (model1 with priority 1 and model0 priority 0. 0 for higher priority and  7 for lowest).Ignore for default(priority 0) prioirty.
    - -t n (number of threads to spawn for each model)
    - -a 1 (offload to j7)
    - -d 1 (use device mem)
    - -e 3 (model1 preempt delay is 3 ms and model2 preempt delay is deafult(FLT_MAX))
## Validation on Target
- Build and run steps remains same for PC emaultionn and target. Copy the below folders from PC to the EVM where this repo is cloned before ruunning the examples
  
    ```
    ./model-artifacts
    ./models
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
    ```


