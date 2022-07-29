#!/bin/bash
# This script should be run inside the CONTAINER
# Outputs:
# - tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a
# - tensorflow/lite/tools/pip_package/gen/tflite_pip/python3/dist/tflite_runtime-2.4.0-py3-none-linux_aarch64.whl
tensorflow/lite/tools/pip_package/gen/tflite_pip/python3/dist/tflite_runtime-2.4.0-py3-none-linux_aarch64.whl
cd $HOME
cp ~/dlrt-build/tflite_2.4/miniconda.sh .
bash ~/miniconda.sh -b -p $HOME/miniconda 
source /root/miniconda/bin/activate 
conda init 
source /root/.bashrc 
source /root/.bashrc 
conda create -n py38 -y python=3.8 
conda activate py38 
conda install -y numpy 

cp dlrt-build/tflite_2.4/bazel-3.1.0-installer-linux-x86_64.sh .
chmod +x bazel-3.1.0-installer-linux-x86_64.sh 
./bazel-3.1.0-installer-linux-x86_64.sh --user 
export PATH=$PATH:/root/bin

cp ~/dlrt-build/tflite_2.4/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz ~
tar -xf gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
cd ~/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/
for f in aarch64-none-linux*; do mv "$f" "$(echo "$f" | sed s/aarch64-none/aarch64/)"; done
ls -l
export PATH=$PATH:$PWD

cd ~
source ~/.bazel/bin/bazel-complete.bash 
cd ~/dlrt-build/tflite_2.4/tensorflow 
tensorflow/lite/tools/make/download_dependencies.sh 
tensorflow/lite/tools/pip_package/build_pip_package_with_bazel.sh aarch64
./tensorflow/lite/tools/make/build_aarch64_lib.sh
cd ~/dlrt-build
