#!/bin/bash

# This script should be run inside the docker host machine 
# Outputs:
# - This script will clone required files for corresponding xxx_build.sh file 
mkdir onnx
cd onnx
if [ ! -f miniconda.sh   ];then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh 
fi

if [ ! -d gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu  ];then    
    wget https://developer.arm.com/-/media/Files/downloads/gnu/11.3.rel1/binrel/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu.tar.xz
    tar -xf arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu.tar.xz
    rm arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu.tar.xz
fi

if [ ! -d targetfs  ];then
    wget https://software-dl.ti.com/jacinto7/esd/processor-sdk-linux-jacinto7/08_00_00_08/exports/tisdk-default-image-j7-evm.tar.xz
    mkdir targetfs
    tar -xf tisdk-default-image-j7-evm.tar.xz -C targetfs/
    rm tisdk-default-image-j7-evm.tar.xz
fi

if [ ! -d protobuf-3.11.3  ];then
    wget https://github.com/protocolbuffers/protobuf/releases/download/v3.11.3/protobuf-cpp-3.11.3.tar.gz
    tar -xzvf protobuf-cpp-3.11.3.tar.gz 
    rm protobuf-cpp-3.11.3.tar.gz 
    cd protobuf-3.11.3/
    ./configure CXXFLAGS=-fPIC --enable-shared=no LDFLAGS="-static"
    make -j 32
    cd ../
fi

if [ ! -f cmake-3.22.1-linux-x86_64.sh ];then
    wget https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-linux-x86_64.sh
    chmod u+x cmake-3.22.1-linux-x86_64.sh 
fi


if [ ! -d onnxruntime ];then
    git clone --depth 1 --single-branch -b tidl-j7 https://github.com/TexasInstruments/onnxruntime.git
else
    cd onnxruntime
    git clean -fdx
fi

cd -