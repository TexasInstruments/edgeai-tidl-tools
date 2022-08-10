#!/bin/bash
# This script should be run inside the CONTAINER
# Outputs:
# - onnxruntime/build_aarch64/Release/libonnxruntime.so
# - onnxruntime/build_aarch64/Release/dist/onnxruntime_tidl-1.7.0-cp38-cp38-linux_x86_64.whl

cd $HOME
cp ~/dlrt-build/onnx/miniconda.sh .
bash ~/miniconda.sh -b -p $HOME/miniconda 
source /root/miniconda/bin/activate 
conda init 
source /root/.bashrc 
conda create -n py38 -y python=3.8 
conda activate py38 
conda install -y numpy 

cp dlrt-build/onnx/cmake-3.22.1-linux-x86_64.sh .
chmod +x cmake-3.22.1-linux-x86_64.sh 
mkdir /usr/bin/cmake
./cmake-3.22.1-linux-x86_64.sh --skip-license --prefix=/usr/bin/cmake
export PATH=$PATH:/usr/bin/cmake/bin

chown root:root -R /root/dlrt-build/onnx/onnxruntime
cd ~/dlrt-build/onnx/onnxruntime

cat << EOF >  tools.cmake
SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_PROCESSOR aarch64)
SET(tools /root/dlrt-build/onnx/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu)
SET(CMAKE_SYSTEM_VERSION 1)
SET(CMAKE_C_COMPILER /root/dlrt-build/onnx/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc)
SET(CMAKE_CXX_COMPILER /root/dlrt-build/onnx/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++)
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
SET(CMAKE_FIND_ROOT_PATH /root/dlrt-build/onnx/targetfs/)
EOF

python3 tools/ci_build/build.py --build_dir build_aarch64 --config Release --build_shared_lib --parallel 32 --skip_tests --skip_onnx_tests --use_tidl --build_wheel --path_to_protoc_exe /root/dlrt-build/onnx/protobuf-3.11.3/src/protoc --cmake_extra_defines "CMAKE_TOOLCHAIN_FILE=/root/dlrt-build/onnx/onnxruntime/tools.cmake" "NUMPY_INCLUDE_DIR=/root/dlrt-build/onnx/targetfs/usr/lib/python3.8/site-packages/numpy/core/include" "PYTHON_INCLUDE_DIR=/root/dlrt-build/onnx//targetfs/usr/include;/root/dlrt-build/onnx//targetfs/usr/include/python3.8;/root/dlrt-build/onnx//targetfs/usr/lib/python3.8/site-packages/numpy/core/include" "PYTHON_LIBRARY=/root/dlrt-build/onnx//targetfs/usr/lib/python3.8" "CMAKE_VERBOSE_MAKEFILE:BOOL=ON"

cd ~/dlrt-build
