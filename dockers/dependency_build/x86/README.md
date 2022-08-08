# X86 Docker Setup
- [X86 Docker Setup](#x86-docker-setups)
  - [Introduction](#introduction)
  - [Setup](#setup)
  - [Copying the libraries](#copying-the-libraries)



## Introduction

   - This folder consist of following x86 docker setups needed for building aarch64 libraries on Ubuntu 18.04 docker
      - tflite_2.4
  - The commands in this folder are intended to run on x86 PC
  - Libraries generated need to copied to the target machine after the build.

## Setup

- To start the aarch64 Ubuntu18 container on X86 PC

  ```
  ./tflite_2.4_prepare.sh # This step will download dependencies for tflite_2.4 compilation
  cd docker
  ./docker_build_u18.sh # This step will build the docker image needed only once
  ./docker_run_u18.sh # This step will run the container and log you in
  ```
- The container volume is mounted to host machine at /root/dlrt-build directory to  access the building script.
- Inside the container run the commands to build teh libraries.
    ```
    cd ~/dlrt-build/
    ./tflite_2.4_build.sh # this step will cross compile tflite_2.4 for aarch64
    ```


## Copying the libraries
- Exit the container 
- Inside edgeai-tidl-tools/dockers/dependency_build/qemu opencv, dlr and onnxrt libraries will be present
- Following libraries are required:
    - libdlr.so : 
    - opencv libs : opencv/opencv-4.2.0/build/lib/*
    - onnxrt libs : onnxruntime/build_aarch64/Release/libonnxruntime.so
    -  onnxrt py whl(not used currently) : onnxruntime/build_aarch64/Release/dist/onnxruntime_tidl-1.7.0-cp36-cp36m-linux_aarch64.whl

  