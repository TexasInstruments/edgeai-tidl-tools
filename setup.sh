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
SCRIPTDIR=`pwd`


skip_cpp_deps=0
skip_arm_gcc_download=0

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
    -h|--help)
    echo Usage: $0 [options]
    echo
    echo Options,
    echo --skip_cpp_deps            Skip Downloading or Compiling dependencies for CPP examples
    echo --skip_arm_gcc_download            Skip Downloading or setting environment variable  for ARM64_GCC_PATH
    
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

######################################################################
# Installing dependencies
echo 'Installing python packages...'
if [[ $arch == x86_64 ]]; then
pip3 install -r ./requirements_pc.txt
elif [[ $arch == aarch64 ]]; then
pip3 install -r ./requirements_j7.txt
fi

if [[ -z "$TIDL_TOOLS_PATH" ]]
then
wget https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_01_00_09-rc1/tidl_tools.tar.gz
tar -xzf tidl_tools.tar.gz
rm tidl_tools.tar.gz
cd  tidl_tools
export TIDL_TOOLS_PATH=$(pwd)
cd ..
fi

if [ $skip_cpp_deps -eq 0 ]
then
if [[ $arch == x86_64 ]]; then
    cd  $TIDL_TOOLS_PATH
    wget https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_01_00_09-rc1/libonnxruntime.so.1.7.0
    wget https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc2/libtensorflow-lite.a
    ln -s libonnxruntime.so.1.7.0 libonnxruntime.so
    cd -
fi
fi


if [[ $arch == x86_64 ]]; then
export LD_LIBRARY_PATH=$TIDL_TOOLS_PATH
fi


if [ $skip_arm_gcc_download -eq 0 ]
then
wget https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
tar -xf gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
export ARM64_GCC_PATH=$(pwd)/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu
fi

if [ $skip_cpp_deps -eq 0 ]
then
cd $HOME
git clone --depth 1 --single-branch -b tidl-j7 https://github.com/TexasInstruments/neo-ai-dlr
git clone --depth 1 --single-branch -b tidl-j7 https://github.com/TexasInstruments/onnxruntime.git
git clone --depth 1 --single-branch -b tidl-j7 https://github.com/TexasInstruments/tensorflow.git
mkdir -p tensorflow/tensorflow/lite/tools/make/downloads
cd tensorflow/tensorflow/lite/tools/make/downloads
wget https://github.com/google/flatbuffers/archive/v1.12.0.tar.gz
tar -xzf v1.12.0.tar.gz
rm v1.12.0.tar.gz
mv  flatbuffers-1.12.0 flatbuffers
cd -
wget https://github.com/opencv/opencv/archive/4.1.0.zip
unzip 4.1.0.zip
rm 4.1.0.zip
if [[ $arch == x86_64 ]]; then
cd opencv-4.1.0/cmake/
cmake -DBUILD_opencv_highgui:BOOL="1" -DBUILD_opencv_videoio:BOOL="0" -DWITH_IPP:BOOL="0" -DWITH_WEBP:BOOL="1" -DWITH_OPENEXR:BOOL="1" -DWITH_IPP_A:BOOL="0" -DBUILD_WITH_DYNAMIC_IPP:BOOL="0" -DBUILD_opencv_cudacodec:BOOL="0" -DBUILD_PNG:BOOL="1" -DBUILD_opencv_cudaobjdetect:BOOL="0" -DBUILD_ZLIB:BOOL="1" -DBUILD_TESTS:BOOL="0" -DWITH_CUDA:BOOL="0" -DBUILD_opencv_cudafeatures2d:BOOL="0" -DBUILD_opencv_cudaoptflow:BOOL="0" -DBUILD_opencv_cudawarping:BOOL="0" -DINSTALL_TESTS:BOOL="0" -DBUILD_TIFF:BOOL="1" -DBUILD_JPEG:BOOL="1" -DBUILD_opencv_cudaarithm:BOOL="0" -DBUILD_PERF_TESTS:BOOL="0" -DBUILD_opencv_cudalegacy:BOOL="0" -DBUILD_opencv_cudaimgproc:BOOL="0" -DBUILD_opencv_cudastereo:BOOL="0" -DBUILD_opencv_cudafilters:BOOL="0" -DBUILD_opencv_cudabgsegm:BOOL="0" -DBUILD_SHARED_LIBS:BOOL="0" -DWITH_ITT=OFF ../
make -j 32
cd -
fi
fi

cd $SCRIPTDIR






