#!/bin/bash

SCRIPTDIR=`pwd`


export DEVICE=j7
pip3 install /mnt/onnx_artifacts/arm/tfl_3.8/tflite_runtime-2.4.0-py3-none-linux_aarch64.whl
pip3 install /mnt/onnx_artifacts/arm/onnxruntime-1.7.0-cp38-cp38-linux_aarch64.whl

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
if [  ! -d opencv4 ];then
    cp -r /mnt/work/psdkra_new/targetfs/usr/include/opencv4 .
    export OPENCV_INSTALL_DIR=$(pwd)
    cd $HOME
fi
cd ~/opencv4
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


if [  ! -d required_lib_20 ];then
    cp -r /mnt/temp/container/required_lib_20  required_lib_20
    cd ~/required_lib_20 
    # required_lib_18 contents:
    # Taken from qemu ubuntu 20 compilation
    # libdlr.so*  
    # libonnxruntime.so -> libonnxruntime.so.1.7.0* 
    # libonnxruntime.so.1.7.0* 
    

    # Taken from j7 target fs
    # libtensorflow-lite.a 
    # libtidl_onnxrt_EP.so* 
    # libtidl_tfl_delegate.so* 
    # libti_rpmsg_char.so* 
    # libti_rpmsg_char.so.0 -> libti_rpmsg_char.so*
    # libvx_tidl_rt.so* 
    # libvx_tidl_rt.so.1.0 -> libvx_tidl_rt.so*      
fi
cd ~/required_lib_20
export LD_LIBRARY_PATH=$(pwd)

cd /usr/lib/aarch64-linux-gnu/
if [  ! -L libopencv_imgcodecs.so ];then
    ln -s libopencv_imgcodecs.so.4.2 libopencv_imgcodecs.so
fi
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

if [  ! -d /usr/dlr ];then
    mkdir /usr/dlr
    cp ~/required_lib_20/libdlr.so /usr/dlr/
fi

if [ ! -d  /usr/local/lib/python3.8/dist-packages/dlr/counter/ccm_config.json ];then
    rm /usr/local/lib/python3.8/dist-packages/dlr/counter/ccm_config.json
    cd /usr/local/lib/python3.8/dist-packages/dlr/counter/
    echo "{\"enable_phone_home\" : false}" > ccm_config.json
    cd $HOME
fi

rm -r  /usr/lib/python3/dist-packages/numpy*
pip3 install numpy
cd $SCRIPTDIR