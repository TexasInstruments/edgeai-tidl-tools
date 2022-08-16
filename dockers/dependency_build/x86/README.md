# X86 Docker Setup
- [X86 Docker Setup](#x86-docker-setups)
  - [Introduction](#introduction)
  - [Setup](#setup)
  - [Copying the libraries](#copying-the-libraries)



## Introduction

   - This folder consist of following x86 docker setups needed for building (cross-compiling) aarch64 libraries on Ubuntu 18.04 docker for arago linux
      - tflite_2.4 (Used in native arago and Ubuntu containers)
      - onnxrt
      - dlr
  - Libraries generated need to copied to the target machine after the build.

## Setup
- To build the dependency for building onnxrt, opencv and dlr run

  ```
  ./build_tflite_2.4.sh ubuntu18 # This will dwld required files and cross compile tflite_2.4 for aarch64 Ubuntu18.04
  ./build_onnxrt.sh ubuntu18# This will dwld required files and cross compile onnxrt for aarch64 Ubuntu18.04
  ./build_dlr.sh ubuntu18 # This will dwld required files and cross compile dlr for aarch64 Ubuntu18.04

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

  