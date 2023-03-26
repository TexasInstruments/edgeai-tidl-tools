#!/bin/bash

SCRIPTDIR=`pwd`
cd $HOME
if [ ! -d required_libs ];then
    mkdir required_libs
fi
export SOC=j7
REL=08_06_00_24

if [ ! -d arago_j7_pywhl ];then
    mkdir arago_j7_pywhl
fi
cd arago_j7_pywhl
STR=`pip3 list | grep dlr`
SUB='dlr'

if [[ "$STR" != *"$SUB"* ]]; then
    wget --quiet  --proxy off https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/ARAGO//dlr-1.10.0-py3-none-any.whl
    yes | pip3 install --upgrade --force-reinstall dlr-1.10.0-py3-none-any.whl
fi
STR=`pip3 list | grep onnxruntime-tidl`
SUB='onnxruntime-tidl'
if [[ "$STR" != *"$SUB"* ]]; then
    wget --quiet  --proxy off https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/ARAGO//onnxruntime_tidl-1.7.0-cp38-cp38-linux_aarch64.whl
    yes | pip3 install onnxruntime_tidl-1.7.0-cp38-cp38-linux_aarch64.whl
fi
STR=`pip3 list | grep tflite-runtime`
SUB='tflite-runtime'
if [[ "$STR" != *"$SUB"* ]]; then
    wget --quiet  --proxy off https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/ARAGO//tflite_runtime-2.8.2-cp38-cp38-linux_aarch64.whl
    yes | pip3 install --upgrade --force-reinstall tflite_runtime-2.8.2-cp38-cp38-linux_aarch64.whl
    # to sync with tensor flow build version
    y | pip3 uninstall  numpy
    yes | pip3 install numpy
fi

cd $HOME
rm -r arago_j7_pywhl
if [  ! -d /usr/include/tensorflow ];then
    wget --quiet  --proxy off https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/ARAGO//tflite_2.8_aragoj7.tar.gz
    tar xf tflite_2.8_aragoj7.tar.gz
    rm tflite_2.8_aragoj7.tar.gz
    mv tflite_2.8_aragoj7/tensorflow /usr/include
    mv tflite_2.8_aragoj7/tflite_2.8 /usr/lib/
    cp tflite_2.8_aragoj7/libtensorflow-lite.a $HOME/required_libs/
    rm -r tflite_2.8_aragoj7    
    cd $HOME
else
    echo "skipping tensorflow setup: found /usr/include/tensorflow"
    echo "To redo the setup delete: /usr/include/tensorflow and run this script again"
fi


if [  ! -d /usr/include/onnxruntime ];then
    wget --quiet  --proxy off https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/ARAGO//onnx_1.7.0_aragoj7.tar.gz
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
    wget --quiet  --proxy off https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/ARAGO//dlr_1.10.0_aragoj7.tar.gz
    tar xf dlr_1.10.0_aragoj7.tar.gz 
    rm dlr_1.10.0_aragoj7.tar.gz 
    cd dlr_1.10.0_aragoj7
    unzip dlr-1.10.0-py3-none-any.whl
    cp ./dlr/libdlr.so $HOME/required_libs/
    cd -
    mv dlr_1.10.0_aragoj7/neo-ai-dlr /usr/include/
    rm -r dlr_1.10.0_aragoj7
    cd $HOME
else
    echo "skipping neo-ai-dlr setup: found /usr/include/neo-ai-dlr"
    echo "To redo the setup delete: /usr/include/neo-ai-dlr and run this script again"
fi

if [  ! -f /usr/include/itidl_rt.h ];then    
    wget --quiet --proxy off   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/TIDL_TOOLS/AM68PA/tidl_tools.tar.gz 
    tar xf tidl_tools.tar.gz
    rm tidl_tools.tar.gz
    cp tidl_tools/itidl_rt.h /usr/include/
    cp  tidl_tools/itvm_rt.h /usr/include/
    cd $HOME
else
    echo "skipping itidl_rt.h setup: found /usr/include/itidl_rt.h"
    echo "To redo the setup delete: /usr/include/itidl_rt.h and run this script again"
fi

if [ -d ~/required_libs ];then
    cp -r ~/required_libs/* /usr/lib/    
fi

if [  ! -f /usr/dlr/libdlr.so ];then
    mkdir /usr/dlr
    cp ~/required_libs/libdlr.so /usr/dlr/
fi

if [  ! -f /usr/lib/libonnxruntime.so.1.7.0 ];then
    ln -s /usr/lib/libonnxruntime.so /usr/lib/libonnxruntime.so.1.7.0
fi
#Cleanup
cd $HOME
rm -rf required_libs
rm -rf tidl_tools

echo "export the following vars"
echo "export SOC=j7"
echo "export TIDL_TOOLS_PATH="
echo "export LD_LIBRARY_PATH=/usr/lib"
cd $SCRIPTDIR