#! /bin/bash
# This script should be run inside the CONTAINER
# Outputs:
# - dlr/neo-ai-dlr/build/lib/libdlr.so
# - dlr/neo-ai-dlr/python/dist/dlr-1.10.0-py3-none-any.whl

chown root:root -R dlr
cd dlr/neo-ai-dlr/build
cmake -DUSE_TIDL=ON -DUSE_TIDL_RT_PATH=/root/dlrt-build/dlr/meta-psdkla/recipes-core/packagegroups/neo-ai-tvm/ -DCMAKE_CXX_FLAGS=-isystem\ /root/dlrt-build/dlr/dlr_tidl_include/  -DDLR_BUILD_TESTS=OFF ..
make -j32
cd ../python
python3 setup.py bdist_wheel
cd -
