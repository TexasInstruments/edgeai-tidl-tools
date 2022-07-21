#!/bin/bash

SCRIPTDIR=`pwd`


cd $HOME
if [  ! -d tensorflow ];then
    git clone --depth 1 --single-branch -b tidl-j7 https://github.com/TexasInstruments/tensorflow.git
    mkdir -p tensorflow/tensorflow/lite/tools/make/downloads
    cd tensorflow/tensorflow/lite/tools/make/downloads
    wget https://github.com/google/flatbuffers/archive/v1.12.0.tar.gz
    tar -xzf v1.12.0.tar.gz
    rm v1.12.0.tar.gz
    mv  flatbuffers-1.12.0 flatbuffers
    cd $HOME
fi
if [  ! -d opencv-4.2.0 ];then
    # wget -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip
    # unzip opencv.zip
    # rm opencv.zip
    # cd opencv-4.2.0
    # export OPENCV_INSTALL_DIR=$(pwd)
    cd $HOME
fi
cd ~/opencv-4.2.0
export OPENCV_INSTALL_DIR=$(pwd)
cd $HOME
if [  ! -d onnxruntime ];then
    git clone --depth 1 --single-branch -b tidl-j7 https://github.com/TexasInstruments/onnxruntime.git
    cd $HOME
fi
if [  ! -d neo-ai-dlr ];then
    git clone --depth 1 --single-branch -b tidl-j7 https://github.com/TexasInstruments/neo-ai-dlr
    cd neo-ai-dlr
    git submodule init
    git submodule update --init --recursive
    cd $HOME
fi


cd $SCRIPTDIR/../../tidl_tools
export TIDL_TOOLS_PATH=$(pwd)
cd $HOME


if [  ! -d required_lib_18 ];then
    cp -r /mnt/temp/container/required_lib_18  required_lib_18
    # required_lib_18 contents:
    # Taken from qemu ubuntu 18 compilation
    # libdlr.so*  
    # libonnxruntime.so -> libonnxruntime.so.1.7.0* 
    # libonnxruntime.so.1.7.0* 
    # opencv/ 

    # Taken from j7 target fs
    # libtensorflow-lite.a 
    # libtidl_onnxrt_EP.so* 
    # libtidl_tfl_delegate.so* 
    # libti_rpmsg_char.so* 
    # libti_rpmsg_char.so.0 -> libti_rpmsg_char.so*
    # libvx_tidl_rt.so* 
    # libvx_tidl_rt.so.1.0 -> libvx_tidl_rt.so*  
    
fi
cd ~/required_lib_18 
export LD_LIBRARY_PATH=$(pwd)
cd /usr/lib/aarch64-linux-gnu/
if [  ! -L libwebp.so ];then
    ln -s libwebp.so.6 libwebp.so
fi
if [  ! -L libjpeg.so ];then
    ln -s libjpeg.so.8 libjpeg.so
fi
if [  ! -L libpng16.so ];then
    ln -s libpng16.so.16 libpng16.so
fi
if [  ! -L libtiff.so ];then
    ln -s libtiff.so.5 libtiff.so
fi

export DEVICE=j7

if [  ! -d /usr/dlr ];then
    mkdir /usr/dlr
    cp ~/required_lib_18/libdlr.so /usr/dlr/
fi
#TODO Upload the whl and wget it and install 
pip3 install /mnt/onnx_artifacts/arm/tfl_3.6/tflite_runtime-2.4.0-py3-none-linux_aarch64.whl


cd $SCRIPTDIR


