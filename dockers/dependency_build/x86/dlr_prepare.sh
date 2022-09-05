#!/bin/bash

# This script should be run in the docker host machine 
# Outputs:
# - This script will clone required files for corresponding xxx_build.sh file 
if [ ! -d dlr ];then
    mkdir dlr
fi
cd dlr

if [ ! -f miniconda.sh   ];then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh 
fi

if [ ! -f cmake-3.22.1-linux-x86_64.sh ];then
    wget https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-linux-x86_64.sh
    chmod u+x cmake-3.22.1-linux-x86_64.sh 
fi

if [ ! -f ti-processor-sdk-rtos-j721e-evm-08_04_00_02.tar.gz ];then 
    wget https://dr-download.ti.com/software-development/software-development-kit-sdk/MD-bA0wfI4X2g/08.04.00.02/ti-processor-sdk-rtos-j721e-evm-08_04_00_02.tar.gz
fi

if [ ! -d ti-processor-sdk-rtos-j721e-evm-08_04_00_02 ];then 
    tar -xf ti-processor-sdk-rtos-j721e-evm-08_04_00_02.tar.gz
fi
export PSDK_RTOS=$(pwd)/ti-processor-sdk-rtos-j721e-evm-08_04_00_02

if [ ! -f clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz ];then 
    wget https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
fi

if [ ! -d clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04 ];then 
    tar -xf clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
fi

if [ ! -d tvm  ];then
    #change the tag for each release
    git clone --depth 1 --single-branch -b TIDL_PSDK_8.4 https://github.com/TexasInstruments/tvm.git
    cd tvm
    git submodule init
    git submodule update --init --recursive
    cd ../
fi

if [ ! -d neo-ai-dlr  ];then
    git clone --depth 1 --single-branch -b TIDL_PSDK_8.4 https://github.com/TexasInstruments/neo-ai-dlr.git
    cd neo-ai-dlr
    git submodule init
    git submodule update --init --recursive
    mkdir build
    cd ../
fi

if [ ! -f gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz ];then
    wget https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
fi

if [ ! -d gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu  ];then
    tar -xf gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
fi

cd ../