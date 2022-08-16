#!/bin/bash

# This script should be run inside the docker host machine 
# Outputs:
# - This script will clone required files for corresponding xxx_build.sh file 
mkdir dlr
cd dlr
if [ ! -f miniconda.sh   ];then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh 
fi

if [ ! -f cmake-3.22.1-linux-x86_64.sh ];then
    wget https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-linux-x86_64.sh
    chmod u+x cmake-3.22.1-linux-x86_64.sh 
fi

if [ ! -d meta-psdkla  ];then
    git clone --depth 1 --single-branch -b master  https://git.ti.com/git/jacinto-linux/meta-psdkla.git
    
fi

if [ ! -d tvm  ];then
    git clone --depth 1 --single-branch -b tidl-j7 https://github.com/TexasInstruments/tvm.git
    cd tvm
    git submodule init
    git submodule update --init --recursive
    cd ../
fi


if [ ! -d neo-ai-dlr  ];then
    git clone --depth 1 --single-branch -b tidl-j7 https://github.com/TexasInstruments/neo-ai-dlr.git
    cd neo-ai-dlr
    git submodule init
    git submodule update --init --recursive
    mkdir build
    cd ../
fi

if [ ! -d gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu  ];then
    wget https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
    tar -xf gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
    rm gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
fi

if [ ! -d dlr_tidl_include  ];then
    mkdir dlr_tidl_include
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/tidl_tools.tar.gz
    tar xf tidl_tools.tar.gz
    rm tidl_tools.tar.gz
    cp tidl_tools/itidl_rt.h dlr_tidl_include/
    cp tidl_tools/itvm_rt.h dlr_tidl_include/
fi




cd ../