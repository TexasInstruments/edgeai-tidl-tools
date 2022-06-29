#! /bin/bash
# This script should be run inside the CONTAINER
# Outputs:
# - onnxruntime/build_aarch64/Release/libonnxruntime.so
# - onnxruntime/build_aarch64/Release/dist/onnxruntime_tidl-1.7.0-cp36-cp36m-linux_aarch64.whl


chown root:root -R onnx
cd onnx/onnxruntime

python3 tools/ci_build/build.py --build_dir build_aarch64 --config Release --build_shared_lib --parallel 32 --skip_tests --skip_onnx_tests --use_tidl --build_wheel --cmake_extra_defines "CMAKE_TOOLCHAIN_FILE=/root/dlrt-build/onnx/tools.cmake" "PYTHON_INCLUDE_DIR=/usr/include;/usr/include/python3.6" "PYTHON_LIBRARY=/usr/lib/python3.6" 
cd -
