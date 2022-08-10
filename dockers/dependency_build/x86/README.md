# X86 Docker Setup
- [X86 Docker Setup](#x86-docker-setups)
  - [Introduction](#introduction)
  - [Setup](#setup)
  - [Copying the libraries](#copying-the-libraries)



## Introduction

   - This folder consist of following x86 docker setups needed for building (cross-compiling) aarch64 libraries on Ubuntu 18.04 docker for arago linux
      - tflite_2.4
      - onnxrt
      - dlr
  - The xx_prepare.sh commands in this folder are intended to run on docker host PC
  - The xx_build.sh commands in this folder are intended to run on docker container
  - Libraries generated need to copied to the target machine after the build.

## Setup
- To download the dependency for building onnxrt, tflite2.4 and dlr run

  ```
  ./tflite_2.4_prepare.sh # This step will download dependencies for tflite_2.4 cross-compilation
  ./dlr_prepare.sh # This step will download dependencies for dlr cross-compilation
  ./onnxrt_prepare.sh # This step will download dependencies for onnxrt cross-compilation
  ```
- To start the Ubuntu18 container on X86 PC

  ```
  cd docker
  ./docker_build.sh ubuntu18 # This step will build the docker image needed only once
  ./docker_run.sh ubuntu18# This step will run the container and log you in
  ```
- The container volume is mounted to host machine at /root/dlrt-build directory to  access the building script.
- Inside the container run the commands to build the libraries.
    ```
    cd ~/dlrt-build/
    ./tflite_2.4_build.sh # this step will cross compile tflite_2.4 for aarch64
    ./dlr_buld.sh # this step will cross compile onnxrt for aarch64
    ./onnxrt_build.sh # this step will cross compile onnxrt for aarch64
    ```


## Copying the libraries
- Exit the container 
- Inside edgeai-tidl-tools/dockers/dependency_build/x86 tflite2.4, dlr and onnxrt libraries will be present
- Following libraries are generated:
    - dlr
        - neo-ai-dlr/python/dist/dlr-1.10.0-py3-none-any.whl
        - neo-ai-dlr/python/build/lib/dlr/libdlr.so
    - onnxrt
        - onnxruntime/build_aarch64/Release/libonnxruntime.so
        - onnxruntime/build_aarch64/Release/dist/onnxruntime_tidl-1.7.0-cp36-cp36m-linux_aarch64.whl
    - tflite_2.4
        - tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a
        - tensorflow/lite/tools/pip_package/gen/tflite_pip/python3/dist/tflite_runtime-2.4.0-py3-none-linux_aarch64.whl

  