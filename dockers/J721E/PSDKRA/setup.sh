#!/bin/bash

SCRIPTDIR=`pwd`
cd $HOME
if [ ! -d required_libs ];then
    mkdir required_libs
fi
export DEVICE=j7

if [ ! -d arago_j7_pywhl ];then
    mkdir arago_j7_pywhl
fi
cd arago_j7_pywhl
STR=`pip3 list | grep dlr`
SUB='dlr'

if [[ "$STR" != *"$SUB"* ]]; then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/psdkr/pywhl/dlr-1.10.0-py3-none-any.whl
    pip3 install --upgrade --force-reinstall dlr-1.10.0-py3-none-any.whl
fi
STR=`pip3 list | grep onnxruntime-tidl`
SUB='onnxruntime-tidl'
if [[ "$STR" != *"$SUB"* ]]; then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/psdkr/pywhl/onnxruntime_tidl-1.7.0-cp38-cp38-linux_aarch64.whl
    pip3 install onnxruntime_tidl-1.7.0-cp38-cp38-linux_aarch64.whl
fi
STR=`pip3 list | grep tflite-runtime`
SUB='tflite-runtime'
if [[ "$STR" != *"$SUB"* ]]; then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/psdkr/pywhl/tflite_runtime-2.4.0-py3-none-linux_aarch64.whl
    pip3 install --upgrade --force-reinstall tflite_runtime-2.4.0-py3-none-linux_aarch64.whl
    # to sync with tensor flow build version
    pip3 uninstall  numpy
    pip3 install numpy
fi

cd $HOME
rm -r arago_j7_pywhl
if [  ! -d /usr/include/tensorflow ];then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/psdkr/tflite_2.4_aragoj7.tar.gz
    tar xf tflite_2.4_aragoj7.tar.gz
    rm tflite_2.4_aragoj7.tar.gz
    cp tflite_2.4_aragoj7/libtensorflow-lite.a  $HOME/required_libs/
    mv tflite_2.4_aragoj7/tensorflow /usr/include/
    rm -r tflite_2.4_aragoj7
    cd $HOME
else
    echo "skipping tensorflow setup: found /usr/include/tensorflow"
    echo "To redo the setup delete: /usr/include/tensorflow and run this script again"
fi

if [  ! -d /usr/include/opencv-4.2.0 ];then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/psdkr/opencv_4.2.0_aragoj7.tar.gz
    tar -xf opencv_4.2.0_aragoj7.tar.gz
    rm opencv_4.2.0_aragoj7.tar.gz
    cp opencv_4.2.0_aragoj7/opencv $HOME/required_libs/
    mv opencv_4.2.0_aragoj7/opencv-4.2.0 /usr/include/
    cd opencv-4.2.0
    cd $HOME
    rm -r opencv_4.2.0_aragoj7
else
    echo "skipping opencv-4.2.0 setup: found /usr/include/opencv-4.2.0"
    echo "To redo the setup delete: /usr/include/opencv-4.2.0 and run this script again"
fi

if [  ! -d /usr/include/onnxruntime ];then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/psdkr/onnx_1.7.0_aragoj7.tar.gz
    tar xf onnx_1.7.0_aragoj7.tar.gz
    rm onnx_1.7.0_aragoj7.tar.gz
    cp -r  onnx_1.7.0_aragoj7/libonnxruntime.so $HOME/required_libs/
    mv onnx_1.7.0_aragoj7/onnxruntime /usr/include/
    rm -r onnx_1.7.0_aragoj7
    cd $HOME
else
    echo "skipping onnxruntime setup: found /usr/include/onnxruntime"
    echo "To redo the setup delete: /usr/include/onnxruntime and run this script again"
fi

if [  ! -d /usr/include/neo-ai-dlr ];then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/psdkr/dlr_1.10.0_aragoj7.tar.gz
    tar xf dlr_1.10.0_aragoj7.tar.gz 
    rm dlr_1.10.0_aragoj7.tar.gz 
    cp -r  dlr_1.10.0_aragoj7/libdlr.so* $HOME/required_libs/
    mv dlr_1.10.0_aragoj7/neo-ai-dlr /usr/include/
    rm -r dlr_1.10.0_aragoj7
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

if [   -d ~/required_libs ];then
    cp ~/required_libs/* /usr/lib/    
fi
if [  ! -f /usr/dlr/libdlr.so ];then
    mkdir /usr/dlr
    cp ~/required_libs/libdlr.so /usr/dlr/
fi
if [  ! -f /usr/lib/libonnxruntime.so.1.7.0 ];then
    ln -s /usr/lib/libonnxruntime.so /usr/lib/libonnxruntime.so.1.7.0
fi

rm -r $HOME/required_libs


echo "export the following vars"
echo "export DEVICE=j7"
echo "export TIDL_TOOLS_PATH="
echo "export LD_LIBRARY_PATH=/usr/lib"
cd $SCRIPTDIR