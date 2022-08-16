#! /bin/bash
# This script should be run inside the docker host machine 
# Outputs:
# - This script will clone required files for corresponding xxx_build.sh file 

mkdir dlr
cd dlr
if [ ! -d meta-psdkla   ];then
git clone --depth 1 --single-branch -b master  https://git.ti.com/git/jacinto-linux/meta-psdkla.git
fi
if [ ! -d meta-psdkla   ];then
    git clone --depth 1 --single-branch -b tidl-j7 https://github.com/TexasInstruments/neo-ai-dlr.git
    cd neo-ai-dlr
    git submodule init
    git submodule update --init --recursive
    mkdir build
    cd ../
else
    cd neo-ai-dlr
    git clean -fdx
    cd ../
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