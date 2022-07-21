#! /bin/bash
# This script should be run inside the CONTAINER
# Outputs:
# - onnxruntime/build_aarch64/Release/libonnxruntime.so
# - onnxruntime/build_aarch64/Release/dist/onnxruntime_tidl-1.7.0-cp36-cp36m-linux_aarch64.whl


chown root:root -R onnx
cd onnx/onnxruntime
ret=`python3 -c 'import sys; print("%i" % (sys.hexversion>0x03060000))'`
if [ $ret -eq 1 ]; then
    # python version > 3.6 assuming 3.8
    python3 tools/ci_build/build.py --build_dir build_aarch64 --config Release --build_shared_lib --parallel 32 --skip_tests --skip_onnx_tests --use_tidl --build_wheel --cmake_extra_defines "CMAKE_TOOLCHAIN_FILE=/root/dlrt-build/onnx/tools.cmake" "PYTHON_INCLUDE_DIR=/usr/include;/usr/include/python3.8" "PYTHON_LIBRARY=/usr/lib/python3.8" 

else
    python3 tools/ci_build/build.py --build_dir build_aarch64 --config Release --build_shared_lib --parallel 32 --skip_tests --skip_onnx_tests --use_tidl --build_wheel --cmake_extra_defines "CMAKE_TOOLCHAIN_FILE=/root/dlrt-build/onnx/tools.cmake" "PYTHON_INCLUDE_DIR=/usr/include;/usr/include/python3.6" "PYTHON_LIBRARY=/usr/lib/python3.6" 
fi

cd -
