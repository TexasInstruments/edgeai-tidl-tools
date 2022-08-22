# QEMU Docker Setup
- [QEMU Docker Setup](#qemu-docker-setups)
  - [Introduction](#introduction)
  - [Setup](#setup)
  - [Installing to TargetFs](#installing-j7-targetfs)
  - [Copying the libraries](#copying-the-libraries)



## Introduction

   - This folder consist of following qemu docker setups needed for building aarch64 libraries on Ubuntu 18.04 and Ubuntu 20.04 using QEMU emulation on X86
      - dlr
      - onnxruntime
      - opencv
  - Libraries generated need to copied to the target machine after the build.
  - For installing required dependencies on J721E target fs

## Setup
- Few packages need to be installed for qemu emulation
  ```
  sudo apt-get install qemu binfmt-support qemu-user-static
  ```
- To build the docker containers (needed only once)
  ```
  cd docker
  ./docker_build_arm.sh ubuntu18 #This will build arm Ubuntu 18.04 container
  ./docker_build_arm.sh ubuntu20 #This will build arm Ubuntu 20.04 container
  ```

- To build the dependency for building onnxrt, opencv and dlr run

  ```
  ./build_opencv.sh ubuntu18 # This will dwld required files for opencv and build the libraries for Ubunut18.04 arm
  ./build_opencv.sh ubuntu20 # This will dwld required files for opencv and build the libraries for Ubunut20.04 arm
  ./build_onnxrt.sh ubuntu18# This will dwld required files for onnxrt build and build the libraries,py whls for Ubunut18.04 arm
  ./build_onnxrt.sh ubuntu18# This will dwld required files for onnxrt build and build the libraries,py whls for Ubunut20.04 arm
  ./build_dlr.sh ubuntu18 # This will dwld required files for dlr build and build the libraries,py whls for Ubunut18.04 arm
  ./build_dlr.sh ubuntu20 # This will dwld required files for dlr build and build the libraries,py whls for Ubunut20.04 arm
  ```


## Installing to TargetFs
- To Install the dependency libs and python wheels to target fs 
  ```
  ./install_j7_targetfs.sh /home/path/to/targetfs/
  ```
## Copying the libraries
- Exit the container 
- Inside edgeai-tidl-tools/dockers/dependency_build/qemu opencv, dlr and onnxrt libraries will be present
- Following libraries are required:
    - libdlr.so
      - dlr/neo-ai-dlr/build/lib/libdlr.so
      - dlr/neo-ai-dlr/python/dist/dlr-1.10.0-py3-none-any.whl 
    - opencv libs 
      - opencv/opencv-4.2.0/build/lib/*
    - onnxrt 
      - onnxruntime/build_aarch64/Release/libonnxruntime.so
      - onnxruntime/build_aarch64/Release/dist/onnxruntime_tidl-1.7.0-cp36-cp36m-linux_aarch64.whl

  