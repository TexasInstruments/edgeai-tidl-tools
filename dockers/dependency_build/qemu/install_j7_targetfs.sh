#!/bin/bash


if (( $# < 1 ));then
echo "usage ./install_j7_targetfs /path/to/targetfs/"
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
wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/psdkr/pywhl/tflite_runtime-2.4.0-py3-none-linux_aarch64.whl

ln -s /usr/bin/pip3 /usr/bin/pip3.8 
pip3 install --upgrade --force-reinstall dlr-1.10.0-py3-none-any.whl  --prefix $TARGET_FS_PATH/usr --disable-pip-version-check
pip3 install onnxruntime_tidl-1.7.0-cp38-cp38-linux_aarch64.whl  --prefix $TARGET_FS_PATH/usr --disable-pip-version-check
pip3 install --upgrade --force-reinstall tflite_runtime-2.4.0-py3-none-linux_aarch64.whl  --prefix $TARGET_FS_PATH/usr --disable-pip-version-check
cd $TARGET_FS_PATH/home/root

if [  ! -d tensorflow ];then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/psdkr/tflite_2.4_aragoj7.tar.gz
    tar xf tflite_2.4_aragoj7.tar.gz
    rm tflite_2.4_aragoj7.tar.gz
    cp tflite_2.4_aragoj7/libtidl_tfl_delegate.so* $TARGET_FS_PATH/home/root/required_libs/
    cp tflite_2.4_aragoj7/libtensorflow-lite.a  $TARGET_FS_PATH/home/root/required_libs/
    mv tflite_2.4_aragoj7/tensorflow .
    rm -r tflite_2.4_aragoj7
    cd $TARGET_FS_PATH/home/root
fi

if [  ! -d opencv-4.2.0 ];then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/psdkr/opencv_4.2.0_aragoj7.tar.gz
    tar -xf opencv_4.2.0_aragoj7.tar.gz
    rm opencv_4.2.0_aragoj7.tar.gz
    cp opencv_4.2.0_aragoj7/opencv $TARGET_FS_PATH/home/root/required_libs/
    mv opencv_4.2.0_aragoj7/opencv-4.2.0 .
    cd $TARGET_FS_PATH/home/root
    rm -r opencv_4.2.0_aragoj7
fi

if [  ! -d onnxruntime ];then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/psdkr/onnx_1.7.0_aragoj7.tar.gz
    tar xf onnx_1.7.0_aragoj7.tar.gz
    rm onnx_1.7.0_aragoj7.tar.gz
    cp -r  onnx_1.7.0_aragoj7/libonnxruntime.so*  $TARGET_FS_PATH/home/root/required_libs/
    cp -r  onnx_1.7.0_aragoj7/libtidl_onnxrt_EP.so*  $TARGET_FS_PATH/home/root/required_libs/
    cd  $TARGET_FS_PATH/home/root/required_libs/
    ln -s libonnxruntime.so libonnxruntime.so.1.7.0
    cd  $TARGET_FS_PATH/home/root
    mv onnx_1.7.0_aragoj7/onnxruntime .
    rm -r onnx_1.7.0_aragoj7
    cd  $TARGET_FS_PATH/home/root
fi

if [  ! -d neo-ai-dlr ];then
    wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/psdkr/dlr_1.10.0_aragoj7.tar.gz
    tar xf dlr_1.10.0_aragoj7.tar.gz 
    rm dlr_1.10.0_aragoj7.tar.gz 
    cp -r  dlr_1.10.0_aragoj7/libdlr.so* $TARGET_FS_PATH/home/root/required_libs/
    mv dlr_1.10.0_aragoj7/neo-ai-dlr .
    rm -r dlr_1.10.0_aragoj7
    cd $TARGET_FS_PATH/home/root/
fi

if [  ! -f  $TARGET_FS_PATH/usr/dlr/libdlr.so ];then
    mkdir  $TARGET_FS_PATH/usr/dlr
    cp  $TARGET_FS_PATH/home/root/required_libs/libdlr.so  $TARGET_FS_PATH/usr/dlr/
fi

echo "export the following vars in target machine"
cd $SCRIPTDIR/../../tidl_tools
temp=`pwd`
echo "export TIDL_TOOLS_PATH=${temp}"
cd ~/opencv-4.2.0
temp=`pwd`
echo "export OPENCV_INSTALL_DIR=${temp}"
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/required_libs"
echo "export DEVICE=j7"
cd $SCRIPTDIR