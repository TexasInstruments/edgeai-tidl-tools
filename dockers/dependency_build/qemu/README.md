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
  - The xx_prepare.sh commands in this folder are intended to run on docker host PC
  - The xx_build.sh commands in this folder are intended to run on docker container
  - Libraries generated need to copied to the target machine after the build.
  - For installing required dependencies on J721E target fs

## Setup
- Few packages need to be installed for qemu emulation
  ```
  sudo apt-get install qemu binfmt-support qemu-user-static
  ```
- To start the aarch64 Ubuntu18 container on X86 PC

  ```
  cd qemu
  ./qemu_init.sh # This step will execute the registering scripts
  ```

- To download the dependency for building onnxrt, opencv and dlr run

  ```
  ./opencv_prepare.sh # This will dwld required files for opencv build
  ./onnxrt_prepare.sh # This will dwld required files for onnxrt build
  ./dlr_prepare.sh # This will dwld required files for dlr build
  ```
- To start the arm UbuntuX container on X86 PC, based on your requirement build and run teh required container by changing the script argument

  ```
  cd docker
  ./docker_build.sh ubuntu18 # This step will build the docker image needed only once
  ./docker_build.sh ubuntu20 # This step will build the docker image needed only once
  ./docker_run.sh ubuntu18# This step will run the container and log you in
  ./docker_run.sh ubuntu20# This step will run the container and log you in
  ```
- The container volume is mounted to host machine at /root/dlrt-build directory to  access the building script.
- Inside the container run the commands to build the libraries.
    ```
    cd ~/dlrt-build/
    ./opencv_build.sh
    ./onnxrt_build.sh
    ./dlr_build.sh
    ```

## Installing to TargetFs
- To Install the deoendeency libs and python wheels to target fs 
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

  