#!/bin/bash

if [ $# -eq 0 ]
  then
    echo "usage ./targetfs_load /path/to/targetfs/"
    exit
fi

REL=09_00_00_01
SCRIPTDIR=`pwd`
TARGET_FS_PATH=$1
echo "installing dependedcies at $TARGET_FS_PATH"
cd $TARGET_FS_PATH/$HOME
if [ ! -d required_libs ];then
    mkdir required_libs
fi

if [ ! -d arago_j7_pywhl ];then
    mkdir arago_j7_pywhl
fi
cd $TARGET_FS_PATH/$HOME/arago_j7_pywhl

wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/ARAGO/dlr-1.13.0-py3-none-any.whl
wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/ARAGO/onnxruntime_tidl-1.7.0-cp310-cp310-linux_aarch64.whl
wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/ARAGO/tflite_runtime-2.8.2-cp310-cp310-linux_aarch64.whl

ln -s /usr/bin/pip3 /usr/bin/pip3.10
pip3 install --upgrade --force-reinstall dlr-1.13.0-py3-none-any.whl  -t $PYTHONPATH --disable-pip-version-check
pip3 install onnxruntime_tidl-1.7.0-cp310-cp310-linux_aarch64.whl  -t $PYTHONPATH --disable-pip-version-check
pip3 install --upgrade --force-reinstall tflite_runtime-2.8.2-cp310-cp310-linux_aarch64.whl -t $PYTHONPATH --disable-pip-version-check
pip3 install --upgrade --force-reinstall --no-cache-dir numpy -t $PYTHONPATH --disable-pip-version-check
cd $TARGET_FS_PATH/$HOME
rm -r $TARGET_FS_PATH/$HOME/arago_j7_pywhl

if [  ! -d $TARGET_FS_PATH/usr/include/tensorflow ];then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/ARAGO/tflite_2.8_aragoj7.tar.gz
    tar xf tflite_2.8_aragoj7.tar.gz
    rm tflite_2.8_aragoj7.tar.gz
    mv tflite_2.8_aragoj7/tensorflow  $TARGET_FS_PATH/usr/include
    mv tflite_2.8_aragoj7/tflite_2.8  $TARGET_FS_PATH/usr/lib/
    cp tflite_2.8_aragoj7/libtensorflow-lite.a $TARGET_FS_PATH/usr/lib/
    rm -r tflite_2.8_aragoj7 
    cd $TARGET_FS_PATH/$HOME
else
    echo "skipping tensorflow setup: found /usr/include/tensorflow"
    echo "To redo the setup delete: /usr/include/tensorflow and run this script again"
fi

if [  ! -d $TARGET_FS_PATH/usr/include/opencv-4.2.0 ];then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/ARAGO/opencv_4.2.0_aragoj7.tar.gz
    tar -xf opencv_4.2.0_aragoj7.tar.gz
    rm opencv_4.2.0_aragoj7.tar.gz
    cp -r opencv_4.2.0_aragoj7/opencv $TARGET_FS_PATH/usr/lib/
    mv opencv_4.2.0_aragoj7/opencv-4.2.0 $TARGET_FS_PATH/usr/include/
    cd $TARGET_FS_PATH/$HOME
    rm -r opencv_4.2.0_aragoj7
else
    echo "skipping opencv-4.2.0 setup: found /usr/include/opencv-4.2.0"
    echo "To redo the setup delete: /usr/include/opencv-4.2.0 and run this script again"
fi

if [  ! -d $TARGET_FS_PATH/usr/include/onnxruntime ];then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/ARAGO/onnx_1.7.0_aragoj7.tar.gz
    tar xf onnx_1.7.0_aragoj7.tar.gz
    rm onnx_1.7.0_aragoj7.tar.gz
    cp -r  onnx_1.7.0_aragoj7/libonnxruntime.so*   $TARGET_FS_PATH/usr/lib/
    cd   $TARGET_FS_PATH/usr/lib/
    ln -s libonnxruntime.so.1.7.0 libonnxruntime.so
    cd  $TARGET_FS_PATH/$HOME
    mv onnx_1.7.0_aragoj7/onnxruntime $TARGET_FS_PATH/usr/include/
    rm -r onnx_1.7.0_aragoj7
    cd  $TARGET_FS_PATH/$HOME
else
    echo "skipping onnxruntime setup: found /usr/include/onnxruntime"
    echo "To redo the setup delete: /usr/include/onnxruntime and run this script again"
fi

if [  ! -f  $TARGET_FS_PATH/usr/include/itidl_rt.h ];then
    git clone -b master git://git.ti.com/processor-sdk-vision/arm-tidl.git
    cp arm-tidl/rt/inc/itidl_rt.h  $TARGET_FS_PATH/usr/include/
    cp arm-tidl/rt/inc/itvm_rt.h $TARGET_FS_PATH/usr/include/
    rm -r arm-tidl
    cd $TARGET_FS_PATH/$HOME/
else
    echo "skipping itidl_rt.h setup: found /usr/include/itidl_rt.h"
    echo "To redo the setup delete: /usr/include/itidl_rt.h and run this script again"
fi

if [  ! -f  $TARGET_FS_PATH/usr/lib/libdlr.so ];then
    cp  $TARGET_FS_PATH/$PYTHONPATH/dlr/libdlr.so  $TARGET_FS_PATH/usr/lib/
fi

if [  ! -f  $TARGET_FS_PATH/usr/dlr/libdlr.so ];then
    mkdir  $TARGET_FS_PATH/usr/dlr
    cp  $TARGET_FS_PATH/usr/lib/libdlr.so  $TARGET_FS_PATH/usr/dlr/
fi

#Cleanup
cd $TARGET_FS_PATH/$HOME/
rm -rf required_libs
rm -rf tidl_tools

echo "export the following vars with correc value in target machine"
echo "export TIDL_TOOLS_PATH="
echo "export DEVICE=j7"
cd $SCRIPTDIR
