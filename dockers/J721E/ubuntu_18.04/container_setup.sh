#!/bin/bash

SCRIPTDIR=`pwd`

SHORT=o:,h
LONG=opencv_install:,compile_glib2_9:,help
OPTS=$(getopt -a -n weather --options $SHORT --longoptions $LONG -- "$@")

eval set -- "$OPTS"
opencv_install=0
compile_glib2_9=0
while :
do
  case "$1" in
    -o | --opencv_install )
      opencv_install="$2"
      shift 2
      ;;
    -c | --compile_glib2_9 )
      compile_glib2_9="$2"
      shift 2
      ;;
    -h | --help)
      echo "usage:"
      echo "source container_setup.sh"
      echo "source container_setup.sh --opencv_install"
      echo "source container_setup.sh --compile_glib2_9"
      break
      ;;
    --)
      shift;
      break
      ;;
    *)
      echo "Unexpected option: $1"
      ;;
  esac
done

download_and_compile_glibc(){
    # in case of Ubunut 18.04 the default glibc is not matching with the tidl lib glibc 
    # we need to compie the glibc2.9 for usage of same     
    cd $HOME
    wget -c https://ftp.gnu.org/gnu/glibc/glibc-2.29.tar.gz
    apt install gawk bison
    tar -zxvf glibc-2.29.tar.gz
    cd glibc-2.29
    mkdir glibc-build
    cd glibc-build/
    ../configure --prefix=/home/cyberithub/glibc-2.29/glibc-build
    make
    make install
    export PATH=/$HOME/glibc-2.29/glibc-build/bin/:/usr/bin/:/bin:/sbin
    export LD_LIBRARY_PATH=$HOME/glibc-2.29/glibc-build/math:/usr/lib/:/usr/lib/aarch64-linux-gnu/:/lib:/lib/aarch64-linux-gnu://$HOME/glibc-2.29/glibc-build/lib
}
if [ $compile_glib2_9 -eq 1 ];then
    download_and_compile_glibc
fi
REL=08_06_00_00

cd $HOME
if [ ! -d required_libs ];then
    mkdir required_libs
fi

export SOC=j7
#For libdlr.so showing error 
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
export TIDL_TOOLS_PATH=

if [ ! -d u_18_pywhl ];then
    mkdir u_18_pywhl
fi
cd u_18_pywhl
STR=`pip3 list | grep dlr`
SUB='dlr'
if [[ "$STR" != *"$SUB"* ]]; then
    wget --quiet https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/UBUNTU_18_04/dlr-1.10.0-py3-none-any.whl
    pip3 install --upgrade --force-reinstall dlr-1.10.0-py3-none-any.whl
    cp /usr/local/lib/python3.6/dist-packages/dlr/libdlr.so $HOME/required_libs
fi
pip3 install protobuf==3.19

STR=`pip3 list | grep onnxruntime-tidl`
SUB='onnxruntime-tidl'
if [[ "$STR" != *"$SUB"* ]]; then
    wget --quiet https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/UBUNTU_18_04/onnxruntime_tidl-1.7.0-cp36-cp36m-linux_aarch64.whl    
    pip3 install onnxruntime_tidl-1.7.0-cp36-cp36m-linux_aarch64.whl
fi
STR=`pip3 list | grep tflite-runtime`
SUB='tflite-runtime'
if [[ "$STR" != *"$SUB"* ]]; then
    wget --quiet https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/UBUNTU_18_04/tflite_runtime-2.8.2-cp36-cp36m-linux_aarch64.whl    
    pip3 install --upgrade --force-reinstall tflite_runtime-2.8.2-cp36-cp36m-linux_aarch64.whl
fi
cd $HOME
rm -r u_18_pywhl
if [  ! -d /usr/include/tensorflow ];then
    wget --quiet https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/UBUNTU_18_04/tflite_2.8_u18.tar.gz
    tar xf tflite_2.8_u18.tar.gz
    rm tflite_2.8_u18.tar.gz
    mv tflite_2.8_u18/tensorflow /usr/include
    mv tflite_2.8_u18/tflite_2.8 /usr/lib/
    mv tflite_2.8_u18/build /usr/lib/
    cp tflite_2.8_u18/libtensorflow-lite.a $HOME/required_libs/
    rm -r tflite_2.8_u18    
    cd $HOME
else
    echo "skipping tensorflow setup: found /usr/include/tensorflow"
    echo "To redo the setup delete: /usr/include/tensorflow and run this script again"
fi

if [  ! -d /usr/include/opencv-4.2.0 ];then
    if [ $opencv_install -eq 1 ];then
        wget --quiet https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/UBUNTU_18_04/opencv_4.2.0_u18.tar.gz        
        tar xf opencv_4.2.0_u18.tar.gz  
    tar xf opencv_4.2.0_u18.tar.gz  
        tar xf opencv_4.2.0_u18.tar.gz  
        rm opencv_4.2.0_u18.tar.gz
        cp -r  opencv_4.2.0_u18/opencv $HOME/required_libs/
        mv opencv_4.2.0_u18/opencv-4.2.0 /usr/include/
        rm -r opencv_4.2.0_u18
        cd $HOME
    fi
else
    echo "skipping opencv-4.2.0 setup: found /usr/include/opencv-4.2.0"
    echo "To redo the setup delete: /usr/include/opencv-4.2.0 and run this script again"
fi

cd $HOME
if [  ! -d /usr/include/onnxruntime ];then
    wget --quiet https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/UBUNTU_18_04/onnx_1.7.0_u18.tar.gz
    tar xf onnx_1.7.0_u18.tar.gz
    rm onnx_1.7.0_u18.tar.gz
    cp -r  onnx_1.7.0_u18/libonnxruntime.so* $HOME/required_libs/    
    mv onnx_1.7.0_u18/onnxruntime /usr/include/
    rm -r onnx_1.7.0_u18
    cd $HOME
else
    echo "skipping onnxruntime setup: found /usr/include/onnxruntime"
    echo "To redo the setup delete: /usr/include/onnxruntime and run this script again"
fi

if [  ! -d /usr/include/neo-ai-dlr ];then
    wget --quiet https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/UBUNTU_18_04/dlr_1.10.0_u18.tar.gz
    tar xf dlr_1.10.0_u18.tar.gz 
    rm dlr_1.10.0_u18.tar.gz
    cd dlr_1.10.0_u18
    unzip dlr-1.10.0-py3-none-any.whl
    cp ./dlr/libdlr.so $HOME/required_libs/
    cd - 
    mv dlr_1.10.0_u18/neo-ai-dlr /usr/include/
    rm -r dlr_1.10.0_u18
    cd $HOME
else    
    echo "skipping neo-ai-dlr setup: found /usr/include/neo-ai-dlr"
    echo "To redo the setup delete: /usr/include/neo-ai-dlr and run this script again"
fi

if [  ! -f /usr/include/itidl_rt.h ];then
    wget --quiet   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/TIDL_TOOLS/AM68PA/tidl_tools.tar.gz
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
    ln -s /host/usr/lib/libti_rpmsg_char.so
    ln -s /host/usr/lib/libti_rpmsg_char.so.0
fi
if [  ! -L libvx_tidl_rt.so ];then
    ln -s /host/usr/lib/libvx_tidl_rt.so
    ln -s libvx_tidl_rt.so libvx_tidl_rt.so.1.0
fi
if [  ! -L libtidl_onnxrt_EP.so ];then
    ln -s /host/usr/lib/libtidl_onnxrt_EP.so  libtidl_onnxrt_EP.so
fi
if [  ! -L libtidl_tfl_delegate.so.1.0 ];then
    ln -s /host/usr/lib/libtidl_tfl_delegate.so.1.0
    ln -s /host/usr/lib/libtidl_tfl_delegate.so
fi
if [  ! -L libtivision_apps.so.8.5.0 ];then
    ln -s /host/usr/lib/libtivision_apps.so  libtivision_apps.so.8.5.0
fi
if [  ! -L libtivision_apps.so.8.6.0 ];then
    ln -s /host/usr/lib/libtivision_apps.so  libtivision_apps.so.8.6.0
fi

if [  ! -f /usr/dlr/libdlr.so ];then
    mkdir /usr/dlr
    if [  ! -f ~/required_libs/libdlr.so ];then
        cp ~/required_libs/libdlr.so /usr/dlr/
    fi
fi

if [   -d $HOME/required_libs ];then
    cp -r $HOME/required_libs/* /usr/lib/
    ln -s /usr/lib/libonnxruntime.so /usr/lib/libonnxruntime.so.1.7.0
    ln -s  /usr/lib/libtidl_tfl_delegate.so.1.0 /usr/lib/libtidl_tfl_delegate.so
fi

#Cleanup
cd $HOME
rm -rf required_libs
rm -rf tidl_tools


cd $SCRIPTDIR

