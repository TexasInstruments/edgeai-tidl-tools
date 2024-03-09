#!/bin/bash

# Copyright (c) 2018-2021, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

######################################################################


compile_armnn(){
    #requires tflite2.8 to be build first
    cd $HOME
    export BASEDIR=~/ArmNNDelegate
    mkdir $BASEDIR
    cd $BASEDIR
    apt-get update && apt-get install git wget unzip zip python git cmake scons

    git clone "https://review.mlplatform.org/ml/armnn" 
    cd armnn
    git checkout branches/armnn_22_02
    
    #Buikd compute lib
    cd $BASEDIR
    git clone https://review.mlplatform.org/ml/ComputeLibrary 
    cd ComputeLibrary/
    git checkout $(../armnn/scripts/get_compute_library.sh -p) # e.g. v21.11
    # The machine used for this guide only has a Neon CPU which is why I only have "neon=1" but if 
    # your machine has an arm Gpu you can enable that by adding `opencl=1 embed_kernels=1 to the command below
    sed -i 's/aarch64-linux-gnu-/aarch64-none-linux-gnu-/' SConstruct
    scons arch=arm64-v8a neon=1 extra_cxx_flags="-fPIC" benchmark_tests=0 validation_tests=0 

    
    #compile protobuff
    cd $BASEDIR
    git clone -b v3.12.0 https://github.com/google/protobuf.git protobuf
    cd protobuf
    git submodule update --init --recursive
    ./autogen.sh
    mkdir arm64_build
    cd arm64_build
    CC=aarch64-none-linux-gnu-gcc \
    CXX=aarch64-none-linux-gnu-g++ \
    ../configure --host=aarch64-linux \
    --prefix=$BASEDIR/google/arm64_pb_install \
    --with-protoc=$BASEDIR/google/x86_64_pb_install/bin/protoc
    make install -j16
    
    #build flatbuffers
    cd $BASEDIR
    wget --quiet  -O flatbuffers-1.12.0.tar.gz https://github.com/google/flatbuffers/archive/v1.12.0.tar.gz
    tar xf flatbuffers-1.12.0.tar.gz
    cd flatbuffers-1.12.0
    rm -f CMakeCache.txt
    mkdir build-arm64
    cd build-arm64
    # Add -fPIC to allow us to use the libraries in shared objects.
    CXXFLAGS="-fPIC" cmake .. -DCMAKE_C_COMPILER=aarch64-none-linux-gnu-gcc \
        -DCMAKE_CXX_COMPILER=aarch64-none-linux-gnu-g++ \
        -DFLATBUFFERS_BUILD_FLATC=1 \
        -DCMAKE_INSTALL_PREFIX:PATH=$BASEDIR/flatbuffers-arm64 \
        -DFLATBUFFERS_BUILD_TESTS=0
    make all install
    
    # build tflite 2.5
    cd $BASEDIR
    git clone https://github.com/tensorflow/tensorflow.git
    cd tensorflow/
    git checkout v2.5.0
    cd ..
    mkdir -p tflite/build
    cd tflite/build
    ARMCC_PREFIX=$ARM64_GCC_PATH/bin/aarch64-none-linux-gnu-\
    ARMCC_FLAGS="-funsafe-math-optimizations" \
    cmake -DCMAKE_C_COMPILER=${ARMCC_PREFIX}gcc \
    -DCMAKE_CXX_COMPILER=${ARMCC_PREFIX}g++ \
    -DCMAKE_C_FLAGS="${ARMCC_FLAGS}" -DCMAKE_CXX_FLAGS="${ARMCC_FLAGS}" \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON  -DCMAKE_SYSTEM_NAME=Linux \
    -DTFLITE_ENABLE_XNNPACK=ON \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    $BASEDIR/tensorflow/tensorflow/lite/ 
    
    cmake --build .

    #finally build armnn
    cd $BASEDIR
    cd armnn
    rm -rf build # Remove any previous cmake build.
    mkdir build && cd build
    # if you've got an arm Gpu add `-DARMCOMPUTECL=1` to the command below
    CXX=aarch64-none-linux-gnu-g++
    CC=aarch64-none-linux-gnu-gcc
    cmake .. -DARMCOMPUTE_ROOT=$BASEDIR/ComputeLibrary \
        -DARMCOMPUTE_BUILD_DIR=$BASEDIR/ComputeLibrary/build/ \
        -DARMCOMPUTENEON=1 \
        -DARMCOMPUTECL=1 \
        -DARMNNREF=1 \
        -DBUILD_TF_LITE_PARSER=1 \
        -DTENSORFLOW_ROOT=$BASEDIR/tensorflow/ \
        -DFLATBUFFERS_ROOT=$BASEDIR/flatbuffers-1.12.0/ \
        -DFLATC_DIR=$BASEDIR/flatbuffers-1.12.0/build \
        -DPROTOBUF_ROOT=$BASEDIR/google/x86_64_pb_install \
        -DPROTOBUF_ROOT=$BASEDIR/google/x86_64_pb_install/ \
        -DPROTOBUF_LIBRARY_DEBUG=$BASEDIR/google/arm64_pb_install/lib/libprotobuf.so.23.0.0 \
        -DPROTOBUF_LIBRARY_RELEASE=$BASEDIR/google/arm64_pb_install/lib/libprotobuf.so.23.0.0 \
        -DTF_LITE_SCHEMA_INCLUDE_PATH=$BASEDIR/tensorflow/tensorflow/lite/schema \
        -DTFLITE_LIB_ROOT=$BASEDIR/tflite/build/ \
        -DBUILD_ARMNN_TFLITE_DELEGATE=1 
    make
}

download_armnn_repo(){    
    export BASEDIR=~/ArmNNDelegate
    mkdir $BASEDIR    
    git clone --depth 1 --single-branch -b branches/armnn_22_02 https://review.mlplatform.org/ml/armnn
    cd armnn
    git checkout branches/armnn_22_02
    cd $HOME
    ln -s $BASEDIR/armnn armnn
}

download_armnn_lib(){    
    export TIDL_TARGET_LIBS=$SCRIPTDIR/tidl_target_libs
    mkdir $TIDL_TARGET_LIBS
    cd $TIDL_TARGET_LIBS
    wget --quiet  https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_03_00_19/libarmnnDelegate.so
    wget --quiet  https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_03_00_19/libarmnn.so
}

pip_install_local()
{
    if [ -f $LOCAL_PATH/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1 ];then
        echo "Local file  found. Installing $LOCAL_PATH/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1"
        pip3 install --force-reinstall $LOCAL_PATH/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1
    else
        echo "Local file not found at $LOCAL_PATH/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1. Installing default  https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1"
        pip3 install https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1
    fi
}

cp_tidl_tools()
{    
    if [ -f $LOCAL_PATH/TIDL_TOOLS/$1/tidl_tools.tar.gz ];then
        if [ $tidl_gpu_tools -eq 1 ];then
            echo "Local file  found. Copying $LOCAL_PATH/TIDL_TOOLS/$1/tidl_tools_gpu.tar.gz"
            cp $LOCAL_PATH/TIDL_TOOLS/$1/tidl_tools_gpu.tar.gz .
        else
            echo "Local file  found. Copying $LOCAL_PATH/TIDL_TOOLS/$1/tidl_tools.tar.gz"
            cp $LOCAL_PATH/TIDL_TOOLS/$1/tidl_tools.tar.gz .
        fi
    else
        if [ $tidl_gpu_tools -eq 1 ];then
            echo "Local file not found at $LOCAL_PATH/TIDL_TOOLS/$1/tidl_tools.tar.gz . Downloading  default   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/TIDL_TOOLS/$1/tidl_tools_gpu.tar.gz"
            wget --quiet  https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/TIDL_TOOLS/$1/tidl_tools_gpu.tar.gz
        else
            echo "Local file not found at $LOCAL_PATH/TIDL_TOOLS/$1/tidl_tools.tar.gz . Downloading  default   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/TIDL_TOOLS/$1/tidl_tools.tar.gz"
            wget --quiet  https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/TIDL_TOOLS/$1/tidl_tools.tar.gz
        fi
    fi
}

cp_osrt_lib()
{ 
    if [ -f $LOCAL_PATH/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1 ];then
        echo "Local file  found. Copying $LOCAL_PATH/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1"
        cp  $LOCAL_PATH/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1 .
        
    else
        echo "Local file not found at  $LOCAL_PATH/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1 . Downloading  default https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1"
        wget --quiet  https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1
    fi
}


SCRIPTDIR=`pwd`
REL=09_01_06_00
skip_cpp_deps=0
skip_arm_gcc_download=0
skip_x86_python_install=0
use_local=0
load_armnn=0


POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    --skip_cpp_deps)
    skip_cpp_deps=1
    ;;
    --skip_arm_gcc_download)
    skip_arm_gcc_download=1
    ;;
    --skip_x86_python_install)
    skip_x86_python_install=1
    ;;
    --use_local)
    use_local=1
    ;;    
    --load_armnn)
    load_armnn=1
    ;;
    -h|--help)
    echo Usage: $0 [options]
    echo
    echo Options,
    echo --skip_cpp_deps            Skip Downloading or Compiling dependencies for CPP examples
    echo --skip_arm_gcc_download            Skip Downloading or setting environment variable  for ARM64_GCC_PATH
    echo --skip_x86_python_install            Skip installing of python packages    
    echo --use_local            use OSRT packages and tidl_tools from localPath if present
    echo --load_armnn           load amrnn libs  for arm
    exit 0
    ;;
esac
shift # past argument
done
set -- "${POSITIONAL[@]}" # restore positional parameters

#Check if tools are built for
if [ $TIDL_TOOLS_TYPE == GPU ];then
    tidl_gpu_tools=1
else
    tidl_gpu_tools=0
fi

version_match=`python3 -c 'import sys;r=0 if sys.version_info >= (3,6) else 1;print(r)'`
if [ $version_match -ne 0 ]; then
    echo 'python version must be >= 3.6'
return
fi

arch=$(uname -p)
if [[ $arch == x86_64 ]]; then
    echo "X64 Architecture"
elif [[ $arch == aarch64 ]]; then
    echo "ARM Architecture"
    $skip_arm_gcc_download=1
else
    echo 'Processor Architecture must be x86_64 or aarch64'
    echo 'Processor Architecture "'$arch'" is Not Supported '
return
fi

if [[ $use_local == 1 ]];then
    if [ -z "$LOCAL_PATH" ];then
        echo "LOCAL_PATH not defined. set LOCAL_PATH to/your/path/08_XX_XX_XX/"        
        return
    else
        echo "using OSRT from LOCAL_PATH:$LOCAL_PATH"
    fi
    
fi

if [ -z "$SOC" ];then
    echo "SOC not defined. Run either of below commands"
    echo "export SOC=am62"
    echo "export SOC=am62a"
    echo "export SOC=am68a"
    echo "export SOC=am68pa"
    echo "export SOC=am69a"
    return
fi

# ######################################################################
# # Installing dependencies
if [[ $arch == x86_64 && $skip_x86_python_install -eq 0 ]]; then   
    echo 'Installing python packages...'
    pip3 install pybind11[global]
    pip3 install -r ./requirements_pc.txt
fi
if [[ $arch == x86_64 ]]; then
    pip3 install pybind11[global]
    if [[ $use_local == 1 ]];then
        echo 'Installing python osrt packages from local...'
        pip_install_local dlr-1.13.0-py3-none-any.whl
        pip_install_local tvm-0.12.0-cp310-cp310-linux_x86_64.whl
        pip_install_local onnxruntime_tidl-1.14.0-cp310-cp310-linux_x86_64.whl
        pip_install_local tflite_runtime-2.8.2-cp310-cp310-linux_x86_64.whl
    else
        echo 'Installing python osrt packages...'
        pip3 install --quiet https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/dlr-1.13.0-py3-none-any.whl
        pip3 install --quiet https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tvm-0.12.0-cp310-cp310-linux_x86_64.whl
        pip3 install --quiet https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/onnxruntime_tidl-1.14.0-cp310-cp310-linux_x86_64.whl
        pip3 install --quiet https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tflite_runtime-2.8.2-cp310-cp310-linux_x86_64.whl
    fi
fi

if [ -z "$TIDL_TOOLS_PATH" ]; then
    if [ -f tidl_tools.tar.gz ];then
        rm tidl_tools.tar.gz
    fi
    if [ -f tidl_tools_gpu.tar.gz ];then
        rm tidl_tools_gpu.tar.gz
    fi
    if [ -d tidl_tools ];then
        rm -r tidl_tools
    fi
    if  [ $SOC == am62a ];then
        if [[ $use_local == 1 ]];then
            cp_tidl_tools AM62A
        else
            if [ $tidl_gpu_tools -eq 1 ];then
                echo 'Downloading gpu tidl tools for AM62A SOC ...'
                wget --quiet   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/TIDL_TOOLS/AM62A/tidl_tools_gpu.tar.gz
            else
                echo 'Downloading tidl tools for AM62A SOC ...'
                wget --quiet   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/TIDL_TOOLS/AM62A/tidl_tools.tar.gz               
            fi
        fi
    elif  [ $SOC == am68pa ];then
        if [[ $use_local == 1 ]];then
            cp_tidl_tools AM68PA
        else
            if [ $tidl_gpu_tools -eq 1 ];then
                echo 'Downloading gpu tidl tools for AM68PA SOC ...'
                wget --quiet   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/TIDL_TOOLS/AM68PA/tidl_tools_gpu.tar.gz
            else
                echo 'Downloading tidl tools for AM68PA SOC ...'
                wget --quiet   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/TIDL_TOOLS/AM68PA/tidl_tools.tar.gz
            fi
        fi
    elif  [ $SOC == am68a ];then
        if [[ $use_local == 1 ]];then
            cp_tidl_tools AM68A
        else
            if [ $tidl_gpu_tools -eq 1 ];then
                echo 'Downloading gpu tidl tools for AM68A SOC ...'
                wget --quiet   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/TIDL_TOOLS/AM68A/tidl_tools_gpu.tar.gz
            else
                echo 'Downloading tidl tools for AM68A SOC ...'
                wget --quiet   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/TIDL_TOOLS/AM68A/tidl_tools.tar.gz
            fi
        fi
    elif  [ $SOC == am69a ];then
        if [[ $use_local == 1 ]];then
            cp_tidl_tools AM69A
        else
            if [ $tidl_gpu_tools -eq 1 ];then
                echo 'Downloading gpu tidl tools for AM69A SOC ...'
                wget --quiet   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/TIDL_TOOLS/AM69A/tidl_tools_gpu.tar.gz
            else
                echo 'Downloading tidl tools for AM69A SOC ...'
                wget --quiet   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/TIDL_TOOLS/AM69A/tidl_tools.tar.gz
            fi  
        fi
    else
        echo "SOC shell var not set correctly($SOC). Set"
        echo "export SOC=am62"
        echo "export SOC=am62a"
        echo "export SOC=am68pa"
        echo "export SOC=am68a"
        echo "export SOC=am69a"
        return 
    fi
    #Untar tidl tools & remove the tar ball
    if [ $tidl_gpu_tools -eq 1 ];then
        tar -xzf tidl_tools_gpu.tar.gz
        if [ -f tidl_tools_gpu.tar.gz ];then
            rm tidl_tools_gpu.tar.gz
        fi
    else
        tar -xzf tidl_tools.tar.gz
        if [ -f tidl_tools.tar.gz ];then
            rm tidl_tools.tar.gz
        fi
    fi
    cd tidl_tools
    if [[ ! -L libvx_tidl_rt.so.1.0 && ! -f libvx_tidl_rt.so.1.0 ]];then
         ln -s  libvx_tidl_rt.so libvx_tidl_rt.so.1.0
    fi 
    export TIDL_TOOLS_PATH=$(pwd)
    #Return to the top level
    cd ..
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TIDL_TOOLS_PATH:$TIDL_TOOLS_PATH/osrt_deps:$TIDL_TOOLS_PATH/osrt_deps/opencv/

if [[ $arch == x86_64 && $skip_arm_gcc_download -eq 0 ]]; then
    if [ ! -d gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu ];then
        wget --quiet  https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
        tar -xf gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
        export ARM64_GCC_PATH=$(pwd)/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu
    else
        echo "skipping gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu download: found $(pwd)/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu"
        export ARM64_GCC_PATH=$(pwd)/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu
    fi
fi

if [[ $arch == x86_64 ]]; then
    if [ -f $CGT7X_ROOT/bin/cl7x ]; then
        echo "CGT7X_ROOT already set to $CGT7X_ROOT, skipping download"
    else
        if [ ! -d ti-cgt-c7000_3.1.0.LTS ];then
            wget --quiet https://dr-download.ti.com/software-development/ide-configuration-compiler-or-debugger/MD-707zYe3Rik/3.1.0.LTS/ti_cgt_c7000_3.1.0.LTS_linux-x64_installer.bin
            chmod +x ti_cgt_c7000_3.1.0.LTS_linux-x64_installer.bin
            ./ti_cgt_c7000_3.1.0.LTS_linux-x64_installer.bin --mode unattended --installdir $(pwd)
            export CGT7X_ROOT=$(pwd)/ti-cgt-c7000_3.1.0.LTS
        else
            echo "skipping ti-cgt-c7000_3.1.0.LTS download: found $(pwd)/ti-cgt-c7000_3.1.0.LTS"
            export CGT7X_ROOT=$(pwd)/ti-cgt-c7000_3.1.0.LTS
        fi
    fi
fi

if [ $skip_cpp_deps -eq 0 ]; then
    if [[ $arch == x86_64 ]]; then
        if [ -d $TIDL_TOOLS_PATH/osrt_deps ];then
            rm -r $TIDL_TOOLS_PATH/osrt_deps
        fi
        mkdir -p $TIDL_TOOLS_PATH/osrt_deps
        cd  $TIDL_TOOLS_PATH/osrt_deps
        # onnx
        if [ ! -d onnx_1.7.0_x86_u22 ];then
            echo "Installing:onnxruntime"
            if [ -f onnx_1.14.0_x86_u22.tar.gz ];then
                rm onnx_1.14.0_x86_u22.tar.gz
            fi
            if [[ $use_local == 1 ]];then
                cp_osrt_lib onnx_1.14.0_x86_u22.tar.gz
            else    
                wget --quiet   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/onnx_1.14.0_x86_u22.tar.gz
            fi
            tar -xf onnx_1.14.0_x86_u22.tar.gz
            cd onnx_1.14.0_x86_u22
            if [ ! -f libonnxruntime.so ];then
                ln -s libonnxruntime.so.1.14.0 libonnxruntime.so 
            fi
            if [ ! -f libonnxruntime.so.1.14.0 ];then
                ln -s libonnxruntime.so libonnxruntime.so.1.14.0 
            fi
            cd ../
            rm onnx_1.14.0_x86_u22.tar.gz
        else
            echo "skipping onnxruntime setup: found $TIDL_TOOLS_PATH/osrt_deps/onnxruntime"
            echo "To redo the setup delete:$TIDL_TOOLS_PATH/osrt_deps/onnxruntime and run this script again"
        fi
        # tflite_2.8
        if [ ! -d tflite_2.8_x86_u22 ];then
            echo "Installing:tflite_2.8"
            if [ -f tflite_2.8_x86_u22.tar.gz ];then
                rm tflite_2.8_x86_u22.tar.gz
            fi
            if [[ $use_local == 1 ]];then
                cp_osrt_lib tflite_2.8_x86_u22.tar.gz
            else    
                wget --quiet   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tflite_2.8_x86_u22.tar.gz
            fi            
            tar -xf tflite_2.8_x86_u22.tar.gz  
            mkdir tflite_2.8_x86_u22 && tar xf tflite_2.8_x86_u22.tar.gz -C tflite_2.8_x86_u22 --strip-components 1      
            rm tflite_2.8_x86_u22.tar.gz   -r
        else
            echo "skipping tensorflow setup: found $TIDL_TOOLS_PATH/osrt_deps/tensorflow"
            echo "To redo the setup delete:$TIDL_TOOLS_PATH/osrt_deps/tensorflow and run this script again"
        fi

        #opencv
        if [ ! -d  opencv_4.2.0_x86_u22 ];then 
            echo "Installing:opencv"
            if [ -f opencv_4.2.0_x86_u22.tar.gz ];then
                rm opencv_4.2.0_x86_u22.tar.gz
            fi
            if [[ $use_local == 1 ]];then
                cp_osrt_lib opencv_4.2.0_x86_u22.tar.gz
            else    
                wget --quiet   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/opencv_4.2.0_x86_u22.tar.gz
            fi
            mkdir opencv_4.2.0_x86_u22 && tar xf opencv_4.2.0_x86_u22.tar.gz -C opencv_4.2.0_x86_u22 --strip-components 1
            rm opencv_4.2.0_x86_u22.tar.gz
        else
            echo "skipping opencv-4.2.0 setup: found $TIDL_TOOLS_PATH/osrt_deps/opencv-4.2.0_x86_u22"
            echo "To redo the setup delete:$TIDL_TOOLS_PATH/osrt_deps/opencv-4.2.0_x86_u22 and run this script again"
        fi

        #dlr
        if [ ! -d dlr_1.10.0_x86_u22 ];then
            echo "Installing:dlr"
            if [ -f dlr_1.10.0_x86_u22.tar.gz ];then
                rm dlr_1.10.0_x86_u22.tar.gz
            fi            
            if [[ $use_local == 1 ]];then
                cp_osrt_lib dlr_1.10.0_x86_u22.tar.gz
            else    
                wget --quiet   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/dlr_1.10.0_x86_u22.tar.gz
            fi   
            mkdir dlr_1.10.0_x86_u22 && tar xf dlr_1.10.0_x86_u22.tar.gz -C dlr_1.10.0_x86_u22 --strip-components 1
            rm dlr_1.10.0_x86_u22.tar.gz   -r
        else
            echo "skipping neo-ai-dlr setup: found $TIDL_TOOLS_PATH/osrt_deps/neo-ai-dlr"
            echo "To redo the setup delete:$TIDL_TOOLS_PATH/osrt_deps/neo-ai-dlr and run this script again"
        fi

dlr_loc=$(python3  << EOF
import dlr 
print(dlr.__file__)
EOF
)
        suffix="__init__.py"
        dlr_loc=${dlr_loc%"$suffix"}
        cp $dlr_loc/libdlr.so .

    fi  

fi

if [ $load_armnn -eq 1 ]; then
    if [[ $arch == x86_64 ]]; then
        if [ ! -d armnn ];then
            download_armnn_repo
        fi 
        if [ ! -f $SCRIPTDIR/tidl_target_libs/libarmnnDelegate.so ];then
            download_armnn_lib
        fi 
    fi
fi

cd $SCRIPTDIR
