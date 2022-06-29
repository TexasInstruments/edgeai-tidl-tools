#! /bin/bash
# This script should be run inside the CONTAINER
# Outputs:
# - onnxruntime/build/Linux/Release/libonnxruntime.so.1.7.0
# - onnxruntime/build/Linux/Release/dist/onnxruntime_tidl-1.7.0-cp36-cp36m-linux_aarch64.whl

chown root:root -R dlr
cd dlr/neo-ai-dlr/build
cmake -DUSE_TIDL=ON -DUSE_TIDL_RT_PATH=/root/dlrt-build/dlr/meta-psdkla/recipes-core/packagegroups/neo-ai-tvm/ -DCMAKE_CXX_FLAGS=-isystem\ /root/dlrt-build/dlr/meta-psdkla/recipes-core/packagegroups/neo-ai-dlr/inc  -DDLR_BUILD_TESTS=OFF ..
make -j32
cd -
