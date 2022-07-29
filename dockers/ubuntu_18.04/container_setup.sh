#!/bin/bash

SCRIPTDIR=`pwd`


cd $HOME
if [ ! -d required_lib_18 ];then
    mkdir required_lib_18
fi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/required_lib_18
export DEVICE=j7
#For libdlr.so showing error 
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

cp  /host/mnt/work/edgeaitidltools/edgeai-tidl-tools/dockers/release_tar/u_18/u_18_pywhl.tar.gz .
tar -xf  u_18_pywhl.tar.gz
rm  u_18_pywhl.tar.gz
cd u_18_pywhl
pip3 install --upgrade --force-reinstall dlr-1.10.0-py3-none-any.whl
pip3 install protobuf==3.19
pip3 install onnxruntime_tidl-1.7.0-cp36-cp36m-linux_aarch64.whl
pip3 install --upgrade --force-reinstall tflite_runtime-2.4.0-py3-none-linux_aarch64.whl
cd $HOME
if [  ! -d tensorflow ];then
    # wget the tar here
    #cp for temp to simulate wget
    cp  /host/mnt/work/edgeaitidltools/edgeai-tidl-tools/dockers/release_tar/u_18/tflite_2.4_u18.tar.gz .
    tar xf tflite_2.4_u18.tar.gz 
    rm tflite_2.4_u18.tar.gz
    
    cp tflite_2.4_u18/libtidl_tfl_delegate.so* $HOME/required_lib_18/
    cp tflite_2.4_u18/libtensorflow-lite.a  $HOME/required_lib_18/

    mv tflite_2.4_u18/tensorflow .
    rm -r tflite_2.4_u18
    cd $HOME
fi
if [  ! -d opencv-4.2.0 ];then

    # wget the tar here
    #cp for temp to simulate wget
    cp  /host/mnt/work/edgeaitidltools/edgeai-tidl-tools/dockers/release_tar/u_18/opencv_4.2.0_u18.tar.gz .
    tar xf opencv_4.2.0_u18.tar.gz  
    rm opencv_4.2.0_u18.tar.gz
    cp -r  opencv_4.2.0_u18/opencv $HOME/required_lib_18/
    mv opencv_4.2.0_u18/opencv-4.2.0 .
    cd opencv-4.2.0
    export OPENCV_INSTALL_DIR=$(pwd)
    cd $HOME
    rm -r opencv_4.2.0_u18
    cd $HOME
fi
cd ~/opencv-4.2.0
export OPENCV_INSTALL_DIR=$(pwd)
cd $HOME
if [  ! -d onnxruntime ];then
    # wget the tar here
    #cp for temp to simulate wget
    cp  /host/mnt/work/edgeaitidltools/edgeai-tidl-tools/dockers/release_tar/u_18/onnx_1.7.0_u18.tar.gz .
    tar xf onnx_1.7.0_u18.tar.gz
    rm onnx_1.7.0_u18.tar.gz
    cp -r  onnx_1.7.0_u18/libonnxruntime.so* $HOME/required_lib_18/
    cp -r  onnx_1.7.0_u18/libtidl_onnxrt_EP.so* $HOME/required_lib_18/

    cd $HOME/required_lib_18/
    ln -s libonnxruntime.so libonnxruntime.so.1.7.0
    cd $HOME
    mv onnx_1.7.0_u18/onnxruntime .
    rm -r onnx_1.7.0_u18
    cd $HOME
fi
if [  ! -d neo-ai-dlr ];then
    cp  /host/mnt/work/edgeaitidltools/edgeai-tidl-tools/dockers/release_tar/u_18/dlr_1.10.0_u18.tar.gz .
    tar xf dlr_1.10.0_u18.tar.gz 
    rm dlr_1.10.0_u18.tar.gz 
    cp -r  dlr_1.10.0_u18/libdlr.so* $HOME/required_lib_18/
    mv dlr_1.10.0_u18/neo-ai-dlr .
    rm -r dlr_1.10.0_u18
    cd $HOME
fi


cd $SCRIPTDIR/../../tidl_tools
export TIDL_TOOLS_PATH=$(pwd)
cd $HOME


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
    cp ~/required_lib_18/libdlr.so /usr/dlr/
fi

cd $SCRIPTDIR


