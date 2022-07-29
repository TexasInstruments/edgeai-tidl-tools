#!/bin/bash

# This script should be run inside the docker host machine 
# Outputs:
# - This script will clone required files for corresponding xxx_build.sh file 
mkdir tflite_2.4
cd tflite_2.4
if [ ! -d tensorflow  ];then
    git clone --depth 1 --single-branch -b tidl-j7 https://github.com/TexasInstruments/tensorflow.git 
else 
    cd tensorflow
    git clean -fdx
    cd -
fi
if [ ! -f miniconda.sh  ];then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh 
fi
if [ ! -f bazel-3.1.0-installer-linux-x86_64.sh   ];then
    wget https://github.com/bazelbuild/bazel/releases/download/3.1.0/bazel-3.1.0-installer-linux-x86_64.sh 
fi
if [ ! -f gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz   ];then
    wget https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
fi
cd -