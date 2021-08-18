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

version_match=`python3 -c 'import sys;r=0 if sys.version_info >= (3,6) else 1;print(r)'`
if [ $version_match -ne 0 ]; then
echo 'python version must be >= 3.6'
exit 1
fi

######################################################################
# Installing dependencies
echo 'Installing python packages...'
pip3 install -r ./requirements_pc.txt
pip3 install https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/dlr-1.8.0-py3-none-any.whl
pip3 install https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/tvm-0.8.dev0-cp36-cp36m-linux_x86_64.whl
pip3 install https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/onnxruntime_tidl-1.7.0-cp36-cp36m-linux_x86_64.whl
pip3 install https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/tflite_runtime-2.4.0-py3-none-any.whl
wget https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/tidl_tools.tar.gz
tar -xzf tidl_tools.tar.gz
rm tidl_tools.tar.gz
cd  tidl_tools
wget https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc2/libonnxruntime.so.1.7.0
wget https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc2/libtensorflow-lite.a
ln -s libonnxruntime.so.1.7.0 libonnxruntime.so
export TIDL_TOOLS_PATH=$(pwd)
export LD_LIBRARY_PATH=$TIDL_TOOLS_PATH
cd ..
cd ..
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
cd opencv-4.1.0/cmake/
cmake -DBUILD_opencv_highgui:BOOL="1" -DBUILD_opencv_videoio:BOOL="0" -DWITH_IPP:BOOL="0" -DWITH_WEBP:BOOL="1" -DWITH_OPENEXR:BOOL="1" -DWITH_IPP_A:BOOL="0" -DBUILD_WITH_DYNAMIC_IPP:BOOL="0" -DBUILD_opencv_cudacodec:BOOL="0" -DBUILD_PNG:BOOL="1" -DBUILD_opencv_cudaobjdetect:BOOL="0" -DBUILD_ZLIB:BOOL="1" -DBUILD_TESTS:BOOL="0" -DWITH_CUDA:BOOL="0" -DBUILD_opencv_cudafeatures2d:BOOL="0" -DBUILD_opencv_cudaoptflow:BOOL="0" -DBUILD_opencv_cudawarping:BOOL="0" -DINSTALL_TESTS:BOOL="0" -DBUILD_TIFF:BOOL="1" -DBUILD_JPEG:BOOL="1" -DBUILD_opencv_cudaarithm:BOOL="0" -DBUILD_PERF_TESTS:BOOL="0" -DBUILD_opencv_cudalegacy:BOOL="0" -DBUILD_opencv_cudaimgproc:BOOL="0" -DBUILD_opencv_cudastereo:BOOL="0" -DBUILD_opencv_cudafilters:BOOL="0" -DBUILD_opencv_cudabgsegm:BOOL="0" -DBUILD_SHARED_LIBS:BOOL="0" -DWITH_ITT=OFF ../
make -j 32
cd -
cd $SCRIPTDIR






