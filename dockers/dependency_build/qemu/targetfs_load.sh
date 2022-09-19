#!/bin/bash


if [ $# -eq 0 ]
  then
    echo "usage ./targetfs_load /path/to/targetfs/"
    exit
fi

SCRIPTDIR=`pwd`
TARGET_FS_PATH=$1
echo "installing dependedcies at $TARGET_FS_PATH"
cd $TARGET_FS_PATH/home/root
if [ ! -d required_libs ];then
    mkdir required_libs
fi

if [ ! -d arago_j7_pywhl ];then
    mkdir arago_j7_pywhl
fi
cd $TARGET_FS_PATH/home/root/arago_j7_pywhl

wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/psdkr/pywhl/dlr-1.10.0-py3-none-any.whl
wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/psdkr/pywhl/onnxruntime_tidl-1.7.0-cp38-cp38-linux_aarch64.whl
wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/psdkr/pywhl/tflite_runtime-2.8.2-cp38-cp38-linux_aarch64.whl

ln -s /usr/bin/pip3 /usr/bin/pip3.8 
pip3 install --upgrade --force-reinstall dlr-1.10.0-py3-none-any.whl  --root $TARGET_FS_PATH --disable-pip-version-check
pip3 install onnxruntime_tidl-1.7.0-cp38-cp38-linux_aarch64.whl  --root $TARGET_FS_PATH --disable-pip-version-check
pip3 install --upgrade --force-reinstall tflite_runtime-2.8.2-cp38-cp38-linux_aarch64.whl  --root $TARGET_FS_PATH --disable-pip-version-check
pip3 install --upgrade --force-reinstall --no-cache-dir numpy  --root $TARGET_FS_PATH --disable-pip-version-check
cd $TARGET_FS_PATH/home/root
rm -r $TARGET_FS_PATH/home/root/arago_j7_pywhl

if [  ! -d $TARGET_FS_PATH/usr/include/tensorflow ];then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/psdkr/tflite_2.8_aragoj7.tar.gz
    tar xf tflite_2.8_aragoj7.tar.gz
    rm tflite_2.8_aragoj7.tar.gz
    mv tflite_2.8_aragoj7/tensorflow  $TARGET_FS_PATH/usr/include
    mv tflite_2.8_aragoj7/tflite_2.8  $TARGET_FS_PATH/usr/lib/
    cp tflite_2.8_aragoj7/libtensorflow-lite.a $TARGET_FS_PATH/usr/lib/
    rm -r tflite_2.8_aragoj7 
    cd $TARGET_FS_PATH/home/root
else
    echo "skipping tensorflow setup: found /usr/include/tensorflow"
    echo "To redo the setup delete: /usr/include/tensorflow and run this script again"
fi

if [  ! -d $TARGET_FS_PATH/usr/include/opencv-4.2.0 ];then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/psdkr/opencv_4.2.0_aragoj7.tar.gz
    tar -xf opencv_4.2.0_aragoj7.tar.gz
    rm opencv_4.2.0_aragoj7.tar.gz
    cp -r opencv_4.2.0_aragoj7/opencv $TARGET_FS_PATH/usr/lib/
    mv opencv_4.2.0_aragoj7/opencv-4.2.0 $TARGET_FS_PATH/usr/include/
    cd $TARGET_FS_PATH/home/root
    rm -r opencv_4.2.0_aragoj7
else
    echo "skipping opencv-4.2.0 setup: found /usr/include/opencv-4.2.0"
    echo "To redo the setup delete: /usr/include/opencv-4.2.0 and run this script again"
fi

if [  ! -d $TARGET_FS_PATH/usr/include/onnxruntime ];then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/psdkr/onnx_1.7.0_aragoj7.tar.gz
    tar xf onnx_1.7.0_aragoj7.tar.gz
    rm onnx_1.7.0_aragoj7.tar.gz
    cp -r  onnx_1.7.0_aragoj7/libonnxruntime.so*   $TARGET_FS_PATH/usr/lib/
    cd   $TARGET_FS_PATH/usr/lib/
    ln -s libonnxruntime.so libonnxruntime.so.1.7.0
    cd  $TARGET_FS_PATH/home/root
    mv onnx_1.7.0_aragoj7/onnxruntime $TARGET_FS_PATH/usr/include/
    rm -r onnx_1.7.0_aragoj7
    cd  $TARGET_FS_PATH/home/root
else
    echo "skipping onnxruntime setup: found /usr/include/onnxruntime"
    echo "To redo the setup delete: /usr/include/onnxruntime and run this script again"
fi

if [  ! -d $TARGET_FS_PATH/usr/include/neo-ai-dlr ];then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/psdkr/dlr_1.10.0_aragoj7.tar.gz
    tar xf dlr_1.10.0_aragoj7.tar.gz 
    rm dlr_1.10.0_aragoj7.tar.gz 
    cp -r  dlr_1.10.0_aragoj7/libdlr.so* $TARGET_FS_PATH/usr/lib
    mv dlr_1.10.0_aragoj7/neo-ai-dlr $TARGET_FS_PATH/usr/include/
    rm -r dlr_1.10.0_aragoj7
    cd $TARGET_FS_PATH/home/root/
else
    echo "skipping neo-ai-dlr setup: found /usr/include/neo-ai-dlr"
    echo "To redo the setup delete: /usr/include/neo-ai-dlr and run this script again"
fi

if [  ! -f  $TARGET_FS_PATH/usr/include/itidl_rt.h ];then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/tidl_tools.tar.gz
    tar xf tidl_tools.tar.gz
    rm tidl_tools.tar.gz
    cp tidl_tools/itidl_rt.h  $TARGET_FS_PATH/usr/include/
    cp  tidl_tools/itvm_rt.h  $TARGET_FS_PATH/usr/include/
    cd $TARGET_FS_PATH/home/root/
else
    echo "skipping itidl_rt.h setup: found /usr/include/itidl_rt.h"
    echo "To redo the setup delete: /usr/include/itidl_rt.h and run this script again"
fi

if [  ! -f  $TARGET_FS_PATH/usr/dlr/libdlr.so ];then
    mkdir  $TARGET_FS_PATH/usr/dlr
    cp  $TARGET_FS_PATH/usr/lib/libdlr.so  $TARGET_FS_PATH/usr/dlr/
fi

#Cleanup
cd $TARGET_FS_PATH/home/root/
rm -rf required_libs
rm -rf tidl_tools

echo "export the following vars with correc value in target machine"
echo "export TIDL_TOOLS_PATH="
echo "export DEVICE=j7"
cd $SCRIPTDIR
