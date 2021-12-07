# CPP Examples
- [CPP Examples](#cpp-examples)
  - [Introduction](#introduction)
  - [Setup](#setup)
  - [Build](#build)
  - [Run](#run)
  - [Validation on Target](#validation-on-target)
  - [Running pre-compiled model from modelzoo](#running-pre-compiled-model-from-modelzoo)


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
    ./bin/Release/ort_main -z "model-artifacts/ort/resnet18-v1/" -v 1 -i "test_data/airshow.jpg" -l "test_data/labels.txt" -a 1
    ./bin/Release/tfl_main -z "model-artifacts/tfl/mobilenet_v1_1.0_224/" -v 1 -i "test_data/airshow.jpg" -l "test_data/labels.txt" -a 1
    ./bin/Release/dlr_main -z "model-artifacts/dlr/tflite_inceptionnetv3" -v 1 -i "test_data/airshow.jpg"  -l "test_data/labels.txt"  -y "cpu"
    ```
## Validation on Target
- Build and runt steps remains same for PC emaultionn and target. Copy the below folders from PC to the EVM where this repo is cloned before ruunning the examples
  
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
    cd ../
    ./bin/Release/tfl_main -z "cl-0000_tflitert_mlperf_mobilenet_v1_1.0_224_tflite/" -v 1 -i "test_data/airshow.jpg" -l "test_data/labels.txt" -a 1 -d 1
    ```


