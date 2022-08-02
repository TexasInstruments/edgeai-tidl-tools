#!/bin/bash

SCRIPTDIR=`pwd`

cd $HOME
if [ ! -d required_lib_20 ];then
    mkdir required_lib_20
fi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/required_lib_20
export DEVICE=j7
#For libdlr.so showing error 
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

if [ ! -d u_20_pywhl ];then
    mkdir u_20_pywhl
fi
cd u_20_pywhl 
#remove existing numpy dlt coz outside env
rm -r /usr/lib/python3/dist-packages/numpy*

wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/ubuntu20_04/pywhl/dlr-1.10.0-py3-none-any.whl
wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/ubuntu20_04/pywhl/onnxruntime_tidl-1.7.0-cp38-cp38-linux_aarch64.whl
wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/ubuntu20_04/pywhl/tflite_runtime-2.4.0-py3-none-linux_aarch64.whl
pip3 install --upgrade --force-reinstall dlr-1.10.0-py3-none-any.whl
pip3 install onnxruntime_tidl-1.7.0-cp38-cp38-linux_aarch64.whl
pip3 install --upgrade --force-reinstall tflite_runtime-2.4.0-py3-none-linux_aarch64.whl
cd $HOME

if [  ! -d tensorflow ];then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/ubuntu20_04/tflite_2.4_u20.tar.gz
    tar xf tflite_2.4_u20.tar.gz
    rm tflite_2.4_u20.tar.gz
    cp tflite_2.4_u20/libtidl_tfl_delegate.so* $HOME/required_lib_20/
    cp tflite_2.4_u20/libtensorflow-lite.a  $HOME/required_lib_20/
    mv tflite_2.4_u20/tensorflow .
    rm -r tflite_2.4_u20
    cd $HOME
fi

if [  ! -d opencv4 ];then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/ubuntu20_04/opencv_4.2.0_u20.tar.gz
    tar -xf opencv_4.2.0_u20.tar.gz
    rm opencv_4.2.0_u20.tar.gz
    cp opencv_4.2.0_u20/opencv $HOME/required_lib_20/
    mv opencv_4.2.0_u20/opencv-4.2.0 .
    cd opencv-4.2.0
    export OPENCV_INSTALL_DIR=$(pwd)
    cd $HOME
    rm -r opencv_4.2.0_u20
fi

cd ~/opencv-4.2.0
export OPENCV_INSTALL_DIR=$(pwd)
cd $HOME

if [  ! -d onnxruntime ];then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/ubuntu20_04/onnx_1.7.0_u20.tar.gz
    tar xf onnx_1.7.0_u20.tar.gz
    rm onnx_1.7.0_u20.tar.gz
    cp -r  onnx_1.7.0_u20/libonnxruntime.so* $HOME/required_lib_20/
    cp -r  onnx_1.7.0_u20/libtidl_onnxrt_EP.so* $HOME/required_lib_20/
    cd $HOME/required_lib_20/
    ln -s libonnxruntime.so libonnxruntime.so.1.7.0
    cd $HOME
    mv onnx_1.7.0_u20/onnxruntime .
    rm -r onnx_1.7.0_u20
    cd $HOME
fi

if [  ! -d neo-ai-dlr ];then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/ubuntu20_04/dlr_1.10.0_u20.tar.gz
    tar xf dlr_1.10.0_u20.tar.gz 
    rm dlr_1.10.0_u20.tar.gz 
    cp -r  dlr_1.10.0_u20/libdlr.so* $HOME/required_lib_20/
    mv dlr_1.10.0_u20/neo-ai-dlr .
    rm -r dlr_1.10.0_u20
    cd $HOME
fi


cd $SCRIPTDIR/../../tidl_tools
export TIDL_TOOLS_PATH=$(pwd)
cd $HOME


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
cd /usr/lib/
if [  ! -L libti_rpmsg_char.so.0 ];then
    ln -s /host/usr/lib/libti_rpmsg_char.so.0.4.0* libti_rpmsg_char.so.0
fi
if [  ! -L libvx_tidl_rt.so ];then
    ln -s /host/usr/lib/libvx_tidl_rt.so.1.0  libvx_tidl_rt.so
    ln -s libvx_tidl_rt.so libvx_tidl_rt.so.1.0
fi
if [  ! -f /usr/dlr/libdlr.so ];then
    mkdir /usr/dlr
    cp ~/required_lib_20/libdlr.so /usr/dlr/
fi

cd $SCRIPTDIR