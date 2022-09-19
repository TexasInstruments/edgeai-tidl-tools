#!/bin/bash

SCRIPTDIR=`pwd`

cd $HOME
if [ ! -d required_libs ];then
    mkdir required_libs
fi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/required_libs
export DEVICE=j7
export TIDL_TOOLS_PATH=
#For libdlr.so showing error 
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

if [ ! -d u_20_pywhl ];then
    mkdir u_20_pywhl
fi
cd u_20_pywhl 
#remove existing numpy dlt coz outside env
rm -r /usr/lib/python3/dist-packages/numpy*

STR=`pip3 list | grep dlr`
SUB='dlr'
if [[ "$STR" != *"$SUB"* ]]; then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/ubuntu20_04/pywhl/dlr-1.10.0-py3-none-any.whl
    pip3 install --upgrade --force-reinstall dlr-1.10.0-py3-none-any.whl
fi

STR=`pip3 list | grep onnxruntime-tidl`
SUB='onnxruntime-tidl'
if [[ "$STR" != *"$SUB"* ]]; then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/ubuntu20_04/pywhl/onnxruntime_tidl-1.7.0-cp38-cp38-linux_aarch64.whl
    pip3 install onnxruntime_tidl-1.7.0-cp38-cp38-linux_aarch64.whl
fi

STR=`pip3 list | grep tflite-runtime`
SUB='tflite-runtime'
if [[ "$STR" != *"$SUB"* ]]; then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/ubuntu20_04/pywhl/tflite_runtime-2.8.2-cp38-cp38-linux_aarch64.whl
    pip3 install --upgrade --force-reinstall tflite_runtime-2.8.2-cp38-cp38-linux_aarch64.whl
fi

cd $HOME
rm -r u_20_pywhl
if [  ! -d /usr/include/tensorflow ];then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/ubuntu20_04/tflite_2.8_aragoj7.tar.gz
    tar xf tflite_2.8_aragoj7.tar.gz
    rm tflite_2.8_aragoj7.tar.gz
    mv tflite_2.8_aragoj7/tensorflow /usr/include
    mv tflite_2.8_aragoj7/tflite_2.8 /usr/lib/
    cp ~/tflite_2.8_aragoj7/libtensorflow-lite.a $HOME/required_libs/
    rm -r tflite_2.8_aragoj7  
    cd $HOME
else
    echo "skipping tensorflow setup: found /usr/include/tensorflow"
    echo "To redo the setup delete: /usr/include/tensorflow and run this script again"
fi

if [  ! -d /usr/include/opencv-4.2.0 ];then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/ubuntu20_04/opencv_4.2.0_u20.tar.gz
    tar -xf opencv_4.2.0_u20.tar.gz
    rm opencv_4.2.0_u20.tar.gz
    cp opencv_4.2.0_u20/opencv $HOME/required_libs/
    mv opencv_4.2.0_u20/opencv-4.2.0 /usr/include/
    cd opencv-4.2.0
    export OPENCV_INSTALL_DIR=$(pwd)
    cd $HOME
    rm -r opencv_4.2.0_u20
else
    echo "skipping opencv-4.2.0 setup: found /usr/include/opencv-4.2.0"
    echo "To redo the setup delete: /usr/include/opencv-4.2.0 and run this script again"
fi


if [  ! -d /usr/include/onnxruntime ];then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/ubuntu20_04/onnx_1.7.0_u20.tar.gz
    tar xf onnx_1.7.0_u20.tar.gz
    rm onnx_1.7.0_u20.tar.gz
    cp -r  onnx_1.7.0_u20/libonnxruntime.so* $HOME/required_libs/
    cp -r  onnx_1.7.0_u20/libtidl_onnxrt_EP.so* $HOME/required_libs/
    mv onnx_1.7.0_u20/onnxruntime /usr/include/
    rm -r onnx_1.7.0_u20
    cd $HOME
else
    echo "skipping onnxruntime setup: found /usr/include/onnxruntime"
    echo "To redo the setup delete: /usr/include/onnxruntime and run this script again"
fi

if [  ! -d /usr/include/neo-ai-dlr ];then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/ubuntu20_04/dlr_1.10.0_u20.tar.gz
    tar xf dlr_1.10.0_u20.tar.gz 
    rm dlr_1.10.0_u20.tar.gz 
    cp -r  /usr/local/lib/python3.8/dist-packages/dlr/libdlr.so $HOME/required_libs/
    mv dlr_1.10.0_u20/neo-ai-dlr /usr/include/
    rm -r dlr_1.10.0_u20
    cd $HOME
else    
    echo "skipping neo-ai-dlr setup: found /usr/include/neo-ai-dlr"
    echo "To redo the setup delete: /usr/include/neo-ai-dlr and run this script again"
fi


if [  ! -f /usr/include/itidl_rt.h ];then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/tidl_tools.tar.gz
    tar xf tidl_tools.tar.gz
    rm tidl_tools.tar.gz
    cp tidl_tools/itidl_rt.h /usr/include/
    cp  tidl_tools/itvm_rt.h /usr/include/
    cd $HOME
else
    echo "skipping itidl_rt.h setup: found /usr/include/itidl_rt.h"
    echo "To redo the setup delete: /usr/include/itidl_rt.h and run this script again"
fi  
#symbolic link creation
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
    ln -s /host/usr/lib/libti_rpmsg_char.so.0.4.1 libti_rpmsg_char.so.0
fi
if [  ! -L libvx_tidl_rt.so ];then
    ln -s /host/usr/lib/libvx_tidl_rt.so.1.0  libvx_tidl_rt.so
    ln -s libvx_tidl_rt.so libvx_tidl_rt.so.1.0
fi
if [  ! -f /usr/dlr/libdlr.so ];then
    mkdir /usr/dlr
    cp ~/required_libs/libdlr.so /usr/dlr/
fi


if [   -d $HOME/required_libs ];then
    cp -r $HOME/required_libs/* /usr/lib/
    ln -s /usr/lib/libonnxruntime.so /usr/lib/libonnxruntime.so.1.7.0
fi

#Cleanup
cd $HOME
rm -rf required_libs
rm -rf tidl_tools

cd $SCRIPTDIR