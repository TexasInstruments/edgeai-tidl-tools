#!/bin/bash
# This script should be run inside the CONTAINER
# Outputs:
# - neo-ai-dlr/python/dist/dlr-1.10.0-py3-none-any.whl
# - neo-ai-dlr/python/build/lib/dlr/libdlr.so
# - tvm/python/dist/tvm-0.9.dev0-cp39-cp39-linux_x86_64.whl (.so are inside the pip whl)
# - tvm/build_x86/libtvm.so*

cd $HOME
chown root:root -R /root/dlrt-build/dlr/
cp ~/dlrt-build/dlr/miniconda.sh .
bash ~/miniconda.sh -b -p $HOME/miniconda 
source /root/miniconda/bin/activate 
conda init 
source /root/.bashrc 

cp dlrt-build/dlr/cmake-3.22.1-linux-x86_64.sh .
chmod +x cmake-3.22.1-linux-x86_64.sh 
mkdir /usr/bin/cmake
./cmake-3.22.1-linux-x86_64.sh --skip-license --prefix=/usr/bin/cmake
export PATH=$PATH:/usr/bin/cmake/bin

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/x86_64-linux-gnu/
ln -s /lib/x86_64-linux-gnu/libtinfo.so.5 /lib/x86_64-linux-gnu/libtinfo.so
export PSDKR_PATH=/root/dlrt-build/dlr/ti-processor-sdk-rtos-j721e-evm-08_04_00_02/

#tvm x86
cd ~/dlrt-build/dlr/tvm
mkdir build_x86
cd build_x86
cmake -DUSE_MICRO=ON -DUSE_SORT=ON -DUSE_TIDL=ON -DUSE_LLVM="/root/dlrt-build/dlr/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/bin/llvm-config --link-static" -DHIDE_PRIVATE_SYMBOLS=ON -DUSE_TIDL_RT_PATH=$(ls -d ${PSDKR_PATH}/tidl_j7*/ti_dl/rt) -DUSE_TIDL_PSDKR_PATH=${PSDKR_PATH} ..
make clean; make -j$(nproc)

# build python package in $TVM_HOME/python/dist
cd ..; rm -fr build; ln -s build_x86 build
cd python; python3 ./setup.py bdist_wheel; ls dist

#tvm aarch
cd ~/dlrt-build/dlr/tvm
export ARM64_GCC_PATH=/root/dlrt-build/dlr/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu
mkdir build_aarch64; cd build_aarch64
cmake -DUSE_SORT=ON -DUSE_TIDL=ON -DUSE_TIDL_RT_PATH=$(ls -d ${PSDKR_PATH}/tidl_j7*/ti_dl/rt) -DUSE_TIDL_PSDKR_PATH=${PSDKR_PATH} -DCMAKE_TOOLCHAIN_FILE=../cmake/modules/contrib/ti-aarch64-linux-gcc-toolchain.cmake ..
make clean; make -j$(nproc) runtime


#dlr x86
cd ~/dlrt-build/dlr/neo-ai-dlr
mkdir build_x86; cd build_x86
cmake -DUSE_TIDL=ON -DUSE_TIDL_RT_PATH=$(ls -d ${PSDKR_PATH}/tidl_j7*/ti_dl/rt) -DDLR_BUILD_TESTS=OFF ..
make clean; make -j$(nproc)

# build python package in $DLR_HOME/python/dist
cd ..; rm -f build; ln -s build_x86 build
cd python; python3 ./setup.py bdist_wheel; ls dist

#dlr aarch
cd ~/dlrt-build/dlr/neo-ai-dlr
mkdir build_aarch64; cd build_aarch64
cmake -DUSE_TIDL=ON -DUSE_TIDL_RT_PATH=$(ls -d ${PSDKR_PATH}/tidl_j7*/ti_dl/rt) -DDLR_BUILD_TESTS=OFF -DCMAKE_TOOLCHAIN_FILE=../cmake/ti-aarch64-linux-gcc-toolchain.cmake ..
make clean; make -j$(nproc)
# build python package in $DLR_HOME/python/dist
cd ..; rm -f build; ln -s build_aarch64 build
cd python; python3 ./setup.py bdist_wheel; ls dist

cd ~/dlrt-build
