#!/bin/bash

SCRIPTDIR=`pwd`

cd $HOME
if [ ! -d required_libs ];then
    mkdir required_libs
fi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/required_libs
export DEVICE=j7
# #For libdlr.so showing error 
# export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

cp  /mnt/work/edgeaitidltools/edgeai-tidl-tools/dockers/release_tar/arago_j7/arago_j7_pywhl.tar.gz .
tar -xf  arago_j7_pywhl.tar.gz
rm  arago_j7_pywhl.tar.gz
cd arago_j7_pywhl
pip3 install --upgrade --force-reinstall dlr-1.10.0-py3-none-any.whl
pip3 install onnxruntime_tidl-1.7.0-cp38-cp38-linux_aarch64.whl
pip3 install --upgrade --force-reinstall tflite_runtime-2.4.0-py3-none-linux_aarch64.whl
cd $HOME

if [  ! -d tensorflow ];then
    cp  /mnt/work/edgeaitidltools/edgeai-tidl-tools/dockers/release_tar/arago_j7/tflite_2.4_aragoj7.tar.gz .
    tar xf tflite_2.4_aragoj7.tar.gz
    rm tflite_2.4_aragoj7.tar.gz
    cp tflite_2.4_aragoj7/libtidl_tfl_delegate.so* $HOME/required_libs/
    cp tflite_2.4_aragoj7/libtensorflow-lite.a  $HOME/required_libs/
    mv tflite_2.4_aragoj7/tensorflow .
    rm -r tflite_2.4_aragoj7
    cd $HOME
fi

if [  ! -d opencv4 ];then
    cp  /mnt/work/edgeaitidltools/edgeai-tidl-tools/dockers/release_tar/arago_j7/opencv_4.2.0_aragoj7.tar.gz .
    tar -xf opencv_4.2.0_aragoj7.tar.gz
    rm opencv_4.2.0_aragoj7.tar.gz
    cp opencv_4.2.0_aragoj7/opencv $HOME/required_libs/
    mv opencv_4.2.0_aragoj7/opencv-4.2.0 .
    cd opencv-4.2.0
    export OPENCV_INSTALL_DIR=$(pwd)
    cd $HOME
    rm -r opencv_4.2.0_aragoj7
fi

cd ~/opencv-4.2.0
export OPENCV_INSTALL_DIR=$(pwd)
cd $HOME

if [  ! -d onnxruntime ];then
    # wget the tar here
    #cp for temp to simulate wget
    cp  /mnt/work/edgeaitidltools/edgeai-tidl-tools/dockers/release_tar/arago_j7/onnx_1.7.0_aragoj7.tar.gz .
    tar xf onnx_1.7.0_aragoj7.tar.gz
    rm onnx_1.7.0_aragoj7.tar.gz
    cp -r  onnx_1.7.0_aragoj7/libonnxruntime.so* $HOME/required_libs/
    cp -r  onnx_1.7.0_aragoj7/libtidl_onnxrt_EP.so* $HOME/required_libs/
    cd $HOME/required_libs/
    ln -s libonnxruntime.so libonnxruntime.so.1.7.0
    cd $HOME
    mv onnx_1.7.0_aragoj7/onnxruntime .
    rm -r onnx_1.7.0_aragoj7
    cd $HOME
fi

if [  ! -d neo-ai-dlr ];then
    cp  /mnt/work/edgeaitidltools/edgeai-tidl-tools/dockers/release_tar/arago_j7/dlr_1.10.0_u20.tar.gz .
    tar xf dlr_1.10.0_aragoj7.tar.gz 
    rm dlr_1.10.0_aragoj7.tar.gz 
    cp -r  dlr_1.10.0_aragoj7/libdlr.so* $HOME/required_libs/
    mv dlr_1.10.0_aragoj7/neo-ai-dlr .
    rm -r dlr_1.10.0_aragoj7
    cd $HOME
fi


cd $SCRIPTDIR/../../tidl_tools
export TIDL_TOOLS_PATH=$(pwd)
cd $HOME


cd /usr/lib/aarch64-linux-gnu/
# if [  ! -L libwebp.so ];then
#     ln -s libwebp.so.6 libwebp.so
# fi
# if [  ! -L libjpeg.so ];then
#     ln -s libjpeg.so.8 libjpeg.so
# fi
# if [  ! -L libpng16.so ];then
#     ln -s libpng16.so.16 libpng16.so
# fi
# if [  ! -L libtiff.so ];then
#     ln -s libtiff.so.5 libtiff.so
# fi
if [  ! -f /usr/dlr/libdlr.so ];then
    mkdir /usr/dlr
    cp ~/required_libs/libdlr.so /usr/dlr/
fi

echo "export the following vars"
cd $SCRIPTDIR/../../tidl_tools
echo "export TIDL_TOOLS_PATH=${pwd}"
cd ~/opencv-4.2.0
echo "export OPENCV_INSTALL_DIR=${pwd}"
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/required_libs"
echo "export DEVICE=j7"
cd $SCRIPTDIR