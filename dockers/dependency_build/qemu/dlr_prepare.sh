#! /bin/bash
# This script should be run inside the docker host machine 
# Outputs:
# - This script will clone required files for corresponding xxx_build.sh file 

mkdir dlr
cd dlr

if [ ! -d neo-ai-dlr  ];then
    git clone --depth 1 --single-branch -b TIDL_PSDK_8.4 https://github.com/TexasInstruments/neo-ai-dlr.git
    cd neo-ai-dlr
    git submodule init
    git submodule update --init --recursive
    cd ../
else
    cd neo-ai-dlr
    git clean -fdx
    cd ../
fi

if [ ! -f ti-processor-sdk-rtos-j721e-evm-08_04_00_02.tar.gz ];then 
    wget https://dr-download.ti.com/software-development/software-development-kit-sdk/MD-bA0wfI4X2g/08.04.00.02/ti-processor-sdk-rtos-j721e-evm-08_04_00_02.tar.gz
fi

if [ ! -d ti-processor-sdk-rtos-j721e-evm-08_04_00_02 ];then 
    tar -xf ti-processor-sdk-rtos-j721e-evm-08_04_00_02.tar.gz
fi

cd ../