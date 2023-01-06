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
    wget -O flatbuffers-1.12.0.tar.gz https://github.com/google/flatbuffers/archive/v1.12.0.tar.gz
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
    wget https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_03_00_19/libarmnnDelegate.so
    wget https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_03_00_19/libarmnn.so
}

SCRIPTDIR=`pwd`


skip_cpp_deps=0
skip_arm_gcc_download=0
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
    --load_armnn)
    load_armnn=1
    ;;
    -h|--help)
    echo Usage: $0 [options]
    echo
    echo Options,
    echo --skip_cpp_deps            Skip Downloading or Compiling dependencies for CPP examples
    echo --skip_arm_gcc_download            Skip Downloading or setting environment variable  for ARM64_GCC_PATH
    echo --load_armnn           load amrnn libs  for arm
    exit 0
    ;;
esac
shift # past argument
done
set -- "${POSITIONAL[@]}" # restore positional parameters



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

if [ -z "$DEVICE" ];then
    echo "DEVICE not defined. Run either of below commands"
    echo "export DEVICE=j7"
    echo "export DEVICE=am62"
    echo "export DEVICE=am62a"
    return
else 
    if [ $DEVICE != j7 ] && [ $DEVICE != am62 ] && [ $DEVICE != am62a ]; then
        echo "DEVICE shell var not set correctly. Set"
        echo "export DEVICE=j7"
        echo "export DEVICE=am62"
        echo "export DEVICE=am62a"
        return
    fi
fi


# ######################################################################
# # Installing dependencies
echo 'Installing python packages...'
if [[ $arch == x86_64 ]]; then
    #TODO8.5 update the link inside requirement PC
    pip3 install -r ./requirements_pc.txt
fi
if [[ -z "$TIDL_TOOLS_PATH" ]]; then
    if  [ $DEVICE == am62a ];then
        echo 'Downloading tidl tools for AM62A device ...'
        wget  https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_05_00_25/tidl_tools.tar.gz
    else
        wget  https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_05_00_00/tidl_tools.tar.gz
    fi
    # wget http://gtweb.dal.design.ti.com/nightly_builds/tidl-osrt-build/327-2022-12-07_01-29-33/artifacts/output/tidl_tools/tidl_tools.tar.gz
    tar -xzf tidl_tools.tar.gz
    rm tidl_tools.tar.gz
    cd  tidl_tools
    if [ ! -L libvx_tidl_rt.so.1.0 ];then
         ln -s  libvx_tidl_rt.so libvx_tidl_rt.so.1.0
    fi 
    export TIDL_TOOLS_PATH=$(pwd)
    cd ..
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TIDL_TOOLS_PATH:$TIDL_TOOLS_PATH/osrt_deps

if [[ $arch == x86_64 && $skip_arm_gcc_download -eq 0 ]]; then
    if [ ! -d gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu ];then
        wget https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
        tar -xf gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
        export ARM64_GCC_PATH=$(pwd)/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu
    else
        echo "skipping gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu download: found $(pwd)/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu"
    fi
fi

if [ $skip_cpp_deps -eq 0 ]; then
    if [[ $arch == x86_64 ]]; then
        mkdir  $TIDL_TOOLS_PATH/osrt_deps
        mkdir  $TIDL_TOOLS_PATH/yaml-cpp
        cd  $TIDL_TOOLS_PATH/osrt_deps
        # onnx
        if [ ! -d onnx_1.7.0_x86_u18 ];then
            rm onnx_1.7.0_x86_u18.tar.gz
            wget http://gtweb.dal.design.ti.com/nightly_builds/tidl-osrt-build/326-2022-12-06_23-27-59/artifacts/output/onnx/onnx_1.7.0_x86_u18.tar.gz
            tar -xf onnx_1.7.0_x86_u18.tar.gz
            cp onnx_1.7.0_x86_u18/libonnxruntime.so .
            cp onnx_1.7.0_x86_u18/onnxruntime . -r 
            ln -s libonnxruntime.so libonnxruntime.so.1.7.0
            rm onnx_1.7.0_x86_u18.tar.gz    -r
        else
            echo "skipping onnxruntime setup: found $TIDL_TOOLS_PATH/osrt_deps/onnxruntime"
            echo "To redo the setup delete:$TIDL_TOOLS_PATH/osrt_deps/onnxruntime and run this script again"
        fi
        # tflite_2.8
        if [ ! -d tflite_2.8_x86_u18 ];then
            rm tflite_2.8_x86_u18.tar.gz
            wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_05_00_00/ubuntu18_04_x86_64/tflite_2.8_x86_u18.tar.gz
            tar -xf tflite_2.8_x86_u18.tar.gz        
            cp tflite_2.8_x86_u18/libtensorflow-lite.a .
            cp tflite_2.8_x86_u18/tensorflow/ . -r
            cp tflite_2.8_x86_u18/tflite_2.8/ tflite_2.8_x86 -r
            rm tflite_2.8_x86_u18.tar.gz   -r
        else
            echo "skipping tensorflow setup: found $TIDL_TOOLS_PATH/osrt_deps/tensorflow"
            echo "To redo the setup delete:$TIDL_TOOLS_PATH/osrt_deps/tensorflow and run this script again"
        fi

        #opencv
        if [ ! -d  opencv_4.2.0_x86_u18 ];then
            rm opencv_4.2.0_x86_u18.tar.gz
            wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_05_00_00/ubuntu18_04_x86_64/opencv_4.2.0_x86_u18.tar.gz
            tar -xf opencv_4.2.0_x86_u18.tar.gz 
            cp opencv_4.2.0_x86_u18/opencv-4.2.0 . -r
            cp opencv_4.2.0_x86_u18/opencv .  -r 
            rm opencv_4.2.0_x86_u18.tar.gz
        else
            echo "skipping opencv-4.2.0 setup: found $TIDL_TOOLS_PATH/osrt_deps/opencv-4.2.0_x86_u18"
            echo "To redo the setup delete:$TIDL_TOOLS_PATH/osrt_deps/opencv-4.2.0_x86_u18 and run this script again"
        fi

        #dlr
        if [ ! -d dlr_1.10.0_x86_u18 ];then
            rm dlr_1.10.0_x86_u18.tar.gz
            wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_05_00_00/ubuntu18_04_x86_64/dlr_1.10.0_x86_u18.tar.gz
            tar -xf dlr_1.10.0_x86_u18.tar.gz
            cp dlr_1.10.0_x86_u18/neo-ai-dlr . -r
            rm dlr_1.10.0_x86_u18.tar.gz   -r
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
