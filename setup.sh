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
compile_tflite_2.8(){
    cd $HOME/tensorflow
    export TENSORFLOW_INSTALL_DIR=$(pwd)

    #native compialtion for x86
    mkdir tflite_build
    cd tflite_build
    cmake ../tensorflow_src/tensorflow/lite
    cmake --build . -j
    
    #cross compilation for arm
    # mkdir tflite_build_arm
    # cd tflite_build_arm
    # ARMCC_PREFIX=$ARM64_GCC_PATH/bin/aarch64-none-linux-gnu-    
    # ARMCC_FLAGS="-funsafe-math-optimizations"
    # cmake -DCMAKE_C_COMPILER=${ARMCC_PREFIX}gcc \
    # -DCMAKE_CXX_COMPILER=${ARMCC_PREFIX}g++ \
    # -DCMAKE_C_FLAGS="${ARMCC_FLAGS}" \
    # -DCMAKE_CXX_FLAGS="${ARMCC_FLAGS}" \
    # -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    # -DCMAKE_SYSTEM_NAME=Linux \
    # -DTFLITE_ENABLE_XNNPACK=ON \
    # -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    # -DBUILD_SHARED_LIBS=ON \
    # ../tensorflow_src/tensorflow/lite/

    # cmake --build . -j 
}

download_tflite_2.8(){
    cd $HOME
    mkdir tensorflow    
    cd tensorflow
    export TENSORFLOW_INSTALL_DIR=$(pwd)
    git clone --depth 1 --single-branch -b v2.8.0 https://github.com/tensorflow/tensorflow.git tensorflow_src

}

download_tflite2.4(){
    cd $HOME
    git clone --depth 1 --single-branch -b tidl-j7 https://github.com/TexasInstruments/tensorflow.git
    mkdir -p tensorflow/tensorflow/lite/tools/make/downloads
    cd tensorflow/tensorflow/lite/tools/make/downloads
    wget https://github.com/google/flatbuffers/archive/v1.12.0.tar.gz
    tar -xzf v1.12.0.tar.gz
    rm v1.12.0.tar.gz
    mv  flatbuffers-1.12.0 flatbuffers
}

compile_opencv(){
    cd $HOME
    #cp -r opencv-4.1.0/cmake opencv-4.1.0/cmake_static 
    if [[ $arch == x86_64 ]]; then
        cd opencv-4.1.0/cmake/
        cmake -DBUILD_opencv_highgui:BOOL="1" -DBUILD_opencv_videoio:BOOL="0" -DWITH_IPP:BOOL="0" -DWITH_WEBP:BOOL="1" -DWITH_OPENEXR:BOOL="1" -DWITH_IPP_A:BOOL="0" -DBUILD_WITH_DYNAMIC_IPP:BOOL="0" -DBUILD_opencv_cudacodec:BOOL="0" -DBUILD_PNG:BOOL="1" -DBUILD_opencv_cudaobjdetect:BOOL="0" -DBUILD_ZLIB:BOOL="1" -DBUILD_TESTS:BOOL="0" -DWITH_CUDA:BOOL="0" -DBUILD_opencv_cudafeatures2d:BOOL="0" -DBUILD_opencv_cudaoptflow:BOOL="0" -DBUILD_opencv_cudawarping:BOOL="0" -DINSTALL_TESTS:BOOL="0" -DBUILD_TIFF:BOOL="1" -DBUILD_JPEG:BOOL="1" -DBUILD_opencv_cudaarithm:BOOL="0" -DBUILD_PERF_TESTS:BOOL="0" -DBUILD_opencv_cudalegacy:BOOL="0" -DBUILD_opencv_cudaimgproc:BOOL="0" -DBUILD_opencv_cudastereo:BOOL="0" -DBUILD_opencv_cudafilters:BOOL="0" -DBUILD_opencv_cudabgsegm:BOOL="0" -DBUILD_SHARED_LIBS:BOOL="0" -DWITH_ITT=OFF ../
        make -j 32
        cd -
        # cd opencv-4.1.0/cmake_static
        # cmake -DWITH_OPENEXR=OFF -DOPENCV_GENERATE_PKGCONFIG=ON BUILD_JASPER=ON -DBUILD_JPEG:BOOL="1" -DBUILD_SHARED_LIBS=OFF -DCMAKE_C_COMPILER=$ARM64_GCC_PATH/bin/aarch64-none-linux-gnu-gcc -DCMAKE_CXX_COMPILER=$ARM64_GCC_PATH/bin/aarch64-none-linux-gnu-g++ -DCMAKE_TOOLCHAIN_FILE=../platforms/linux/arm-gnueabi.toolchain.cmake ../
        # make -j32
        # cd -
    fi
}
download_opencv(){
    cd $HOME
    wget https://github.com/opencv/opencv/archive/4.1.0.zip
    unzip 4.1.0.zip
    cd opencv-4.1.0
    export OPENCV_INSTALL_DIR=$(pwd)
    cd -   
    rm 4.1.0.zip
}
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
    # wget https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_02_00_01-rc1/libarmnn.so
}

SCRIPTDIR=`pwd`


skip_cpp_deps=0
skip_arm_gcc_download=0
skip_armnn=0

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
    --skip_armnn)
    skip_armnn=1
    ;;
    -h|--help)
    echo Usage: $0 [options]
    echo
    echo Options,
    echo --skip_cpp_deps            Skip Downloading or Compiling dependencies for CPP examples
    echo --skip_arm_gcc_download            Skip Downloading or setting environment variable  for ARM64_GCC_PATH
    echo --skip_armnn_compile            Skip amrnn cross compilation  for arm
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
    return
else 
    if [ $DEVICE != j7 ] && [ $DEVICE != am62 ]; then
        echo "DEVICE shell var not set correctly. Set"
        echo "export DEVICE=j7"
        echo "export DEVICE=am62"
        return
    fi
fi

######################################################################
# Installing dependencies
echo 'Installing python packages...'
if [[ $arch == x86_64 ]]; then
    pip3 install -r ./requirements_pc.txt
elif [[ $arch == aarch64 ]]; then
    if [[ $DEVICE == j7 ]]; then
        pip3 install -r ./requirements_j7.txt
    elif [[ $DEVICE == am62 ]]; then
        echo "using python packages from filesytem "
    fi
fi
if [[ -z "$TIDL_TOOLS_PATH" ]]; then
    wget https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_02_00_01-rc1/tidl_tools.tar.gz
    tar -xzf tidl_tools.tar.gz
    rm tidl_tools.tar.gz
    cd  tidl_tools
    export TIDL_TOOLS_PATH=$(pwd)
    cd ..
fi


if [[ $arch == x86_64 ]]; then
    if [ $skip_arm_gcc_download -eq 0 ]; then
        wget https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
        tar -xf gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
        export ARM64_GCC_PATH=$(pwd)/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu
    fi
fi

if [ $skip_cpp_deps -eq 0 ]; then
    if [[ $arch == x86_64 ]]; then
        cd  $TIDL_TOOLS_PATH
        wget https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_02_00_01-rc1/libonnxruntime.so.1.7.0
        wget https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc2/libtensorflow-lite.a
        ln -s libonnxruntime.so.1.7.0 libonnxruntime.so
        export LD_LIBRARY_PATH=$TIDL_TOOLS_PATH
        cd -
    fi
    cd $HOME
    git clone --depth 1 --single-branch -b tidl-j7 https://github.com/TexasInstruments/neo-ai-dlr
    cd neo-ai-dlr
    git submodule init
    git submodule update --init --recursive
    cd ..
    git clone --depth 1 --single-branch -b tidl-j7 https://github.com/TexasInstruments/onnxruntime.git
    if [[ $DEVICE == j7 ]]; then
        echo "Device j7: Downloading tflite 2.4"
        download_tflite2.4
    elif [[  $DEVICE == am62 ]]; then
        echo "Device am62: Downloading tflite 2.8"
        download_tflite_2.8
        if [[ $arch == x86_64 ]]; then
            #to update the cmake
            cd $SCRIPTDIR
            wget https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-linux-x86_64.sh
            chmod u+x cmake-3.22.1-linux-x86_64.sh 
            mkdir $SCRIPTDIR/cmake
            ./cmake-3.22.1-linux-x86_64.sh --skip-license --prefix=$SCRIPTDIR/cmake
            export PATH=$SCRIPTDIR/cmake/bin:$PATH

            compile_tflite_2.8
        fi
    else
        echo 'unknown DEVICE: "'$DEVICE'"'
        return
    fi

    download_opencv
    if  [[$arch == x86_64 ]] ; then
        cd $HOME        
        compile_opencv
    fi
fi

if [ $skip_armnn -eq 0 ]; then
    if [[ $arch == x86_64 ]]; then
        download_armnn_repo
        download_armnn_lib
    fi
fi

cd $SCRIPTDIR






