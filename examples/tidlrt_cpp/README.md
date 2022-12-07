# TIDLRT CPP Examples
- [TIDLRT CPP Examples](#tidlrt-cpp-examples)
  - [Introduction](#introduction)
  - [Setup](#setup)
  - [Build](#build)
  - [Run](#run)
  - [Validation on Target](#validation-on-target)


## Introduction
   - TIDL RT CPP APIs only supports the model inference for the models which can be fully offloaded to DSP. The user is expected  to run the [Python Examples](../osrt_python/README.md#python-example) on PC to generate the model artifacts.
> Note : We are planing to clean-up and unify the user interface for CPP examples by next release. We are also planning to add more CPP examples.

## Setup
- Prepare the Environment for the Model compilation by following the setup section [here](../../README.md#setup)
<<<<<<< HEAD
=======

>>>>>>> README link fix TIDL-2747

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
    ./bin/Release/tidlrt_clasification -l test_data/labels.txt -i test_data/airshow.jpg  -f model-artifacts/tfl/mobilenet_v1_1.0_224/ -d 1
    ```
## Validation on Target
<<<<<<< HEAD
<<<<<<< HEAD
- Build and run steps remains same for PC emulation and target. Copy the below folders from PC to the EVM where this repo is cloned before running the examples

=======
<<<<<<< HEAD
- Build and runt steps remains same for PC emulation and target. Copy the below folders from PC to the EVM where this repo is cloned before running the examples
=======
- Build and run steps remains same for PC emulation and target. Copy the below folders from PC to the EVM where this repo is cloned before running the examples
>>>>>>> README link fix TIDL-2747
>>>>>>> README link fix TIDL-2747
=======
- Build and run steps remains same for PC emulation and target. Copy the below folders from PC to the EVM where this repo is cloned before running the examples
>>>>>>> need link update
  
    ```
    ./model-artifacts
    ./models
    ```



