# QEMU Docker Setup
- [QEMU Docker Setup](#qemu-docker-setups)
  - [Introduction](#introduction)
  - [Setup](#setup)
  - [Copying the libraries](#copying-the-libraries)



## Introduction

   - This folder consist of following qemu docker setups needed for building aarch64 libraries on Ubuntu 18 using QEMU emulation on X86
      - dlr
      - onnxruntime
      - opencv
  - The commands in this folder are intended to run on x86 PC
  - Libraries generated need to copied to the target machine after the build.

## Setup
- Few packages need to be installed for qemu emulation
  ```
  sudo apt-get install qemu binfmt-support qemu-user-static
  ```
- To start the aarch64 Ubuntu18 container on X86 PC

  ```
  cd qemu
  ./qemu_init.sh # This step will execute the registering scripts
  ./opencv_prepare.sh # This will dwld required files for opencv build
  ./onnxrt_prepare.sh # This will dwld required files for onnxrt build
  ./dlr_prepare.sh # This will dwld required files for dlr build
  cd docker
  ./docker_build.sh # This step will build the docker image
  ./docker_run.sh # This step will run the container and log you in
  ```
- The container volume is mounted to host machine at /root/dlrt-build directory to  access the building script.
- Inside the container run the commands to build teh libraries.
    ```
    cd ~/dlrt-build/
    ./opencv_build.sh
    ./onnxrt_build.sh
    ./dlr_build.sh
    ```


## Copying the libraries
- Exit the container 
- Inside edgeai-tidl-tools/dockers/qemu opencv, dlr and onnxrt libraries will be present
- Following libraries are required:
    - libdlr.so : 
    - opencv libs : opencv/opencv-4.2.0/build/lib/*
    - onnxrt libs : onnxruntime/build_aarch64/Release/libonnxruntime.so
    -  onnxrt py whl(not used currently) : onnxruntime/build_aarch64/Release/dist/onnxruntime_tidl-1.7.0-cp36-cp36m-linux_aarch64.whl

  