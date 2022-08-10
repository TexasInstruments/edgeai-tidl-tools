#!/bin/bash
# This script should be run inside the CONTAINER
# Outputs:
# - neo-ai-dlr/python/dist/dlr-1.10.0-py3-none-any.whl
# - neo-ai-dlr/python/build/lib/dlr/libdlr.so

cd $HOME
cp ~/dlrt-build/dlr/miniconda.sh .
bash ~/miniconda.sh -b -p $HOME/miniconda 
source /root/miniconda/bin/activate 
conda init 
source /root/.bashrc 
conda create -n py38 -y python=3.8 
conda activate py38 
conda install -y numpy 

cp dlrt-build/dlr/cmake-3.22.1-linux-x86_64.sh .
chmod +x cmake-3.22.1-linux-x86_64.sh 
mkdir /usr/bin/cmake
./cmake-3.22.1-linux-x86_64.sh --skip-license --prefix=/usr/bin/cmake
export PATH=$PATH:/usr/bin/cmake/bin

chown root:root -R /root/dlrt-build/dlr/
cd ~/dlrt-build/dlr/neo-ai-dlr
cat << EOF >  tools.cmake
SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_PROCESSOR aarch64)
SET(CMAKE_SYSTEM_VERSION 1)
SET(tools /root/dlrt-build/dlr/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu)
SET(CMAKE_C_COMPILER /root/dlrt-build/dlr/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc)
SET(CMAKE_CXX_COMPILER /root/dlrt-build/dlr/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++)
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
EOF

rm -rf build
mkdir build
cd build
cmake -DUSE_TIDL=ON -DUSE_TIDL_RT_PATH=/root/dlrt-build/dlr/dlr_tidl_include/ -DCMAKE_CXX_FLAGS=-isystem\ /root/dlrt-build/dlr/dlr_tidl_include/ -DDLR_BUILD_TESTS=OFF -DCMAKE_TOOLCHAIN_FILE=../tools.cmake ..
make -j 32
cd ../python
python3 setup.py bdist_wheel

cd ~/dlrt-build
