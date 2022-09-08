#! /bin/bash
# This script should be run inside the CONTAINER
# Outputs:
# - dlr/neo-ai-dlr/build/lib/libdlr.so
# - dlr/neo-ai-dlr/python/dist/dlr-1.10.0-py3-none-any.whl


export PSDKR_PATH=/root/dlrt-build/dlr/ti-processor-sdk-rtos-j721e-evm-08_04_00_02/
chown root:root -R dlr
cd dlr/neo-ai-dlr
mkdir build_$1_aarch; cd build_$1_aarch
cmake -DUSE_TIDL=ON -DUSE_TIDL_RT_PATH=$(ls -d ${PSDKR_PATH}/tidl_j7*/ti_dl/rt)   -DDLR_BUILD_TESTS=OFF ..
ls  -d ${PSDKR_PATH}/tidl_j721e_08_04_00_12/ti_dl/rt
make clean; make -j$(nproc)

# build python package in $DLR_HOME/python/dist
cd ..; rm -f build; ln -s build_$1_aarch build
cd python; python3 ./setup.py bdist_wheel; ls dist
mkdir -p /root/dlrt-build/dlr/neo-ai-dlr/python/dist/$1/
mv /root/dlrt-build/dlr/neo-ai-dlr/python/dist/dlr-1.10.0-py3-none-any.whl /root/dlrt-build/dlr/neo-ai-dlr/python/dist/$1/dlr-1.10.0-py3-none-any.whl 

