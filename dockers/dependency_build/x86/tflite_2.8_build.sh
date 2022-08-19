#!/bin/bash
# This script should be run inside the CONTAINER
# Outputs:TODO
# - tflite_2.8/tflite_build_arm/libtensorflow-lite.a
# - (not build from this script build from build_tflite_2.8.sh)tflite_2.8/tensorflow_src/tensorflow/tensorflow/lite/tools/pip_package/gen/tflite_pip/python3/dist/tflite_runtime-2.8.0-cp36-cp36m-linux_aarch64.whl

cd $HOME

# Setup cmake > 3.16 required for tflite2.8 build
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add -
apt install -y software-properties-common
apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
apt-get update
apt install -y cmake

cd ~
cp ~/dlrt-build/tflite_2.8/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz ~
tar -xf gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
ARMCC_PREFIX=~/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-
ARMCC_FLAGS="-funsafe-math-optimizations"
cd ~/dlrt-build/tflite_2.8/
mkdir tflite_build_arm
cd tflite_build_arm
rm -r *
cmake -DCMAKE_C_COMPILER=${ARMCC_PREFIX}gcc \
      -DCMAKE_CXX_COMPILER=${ARMCC_PREFIX}g++ \
      -DCMAKE_C_FLAGS="${ARMCC_FLAGS}" \
      -DCMAKE_CXX_FLAGS="${ARMCC_FLAGS}" \
      -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
      -DCMAKE_SYSTEM_NAME=Linux \
      -DTFLITE_ENABLE_XNNPACK=ON \
      -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
     ../tensorflow_src/tensorflow/tensorflow/lite/
cmake --build . -j
cd ~

cd ~/dlrt-build
