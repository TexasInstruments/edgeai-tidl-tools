#! /bin/bash
# This script should be run inside the CONTAINER
# Outputs:
# - This will generate the tflite2.8 wheel based on the container image version py38 and py36 .whl file will be generated 
# - For ubuntu20 and ubunut 18 , py38 and py36 wheels respectevely  will be generated 


chown root:root -R tensorflow_src
cd tensorflow_src

PYTHON=python3 tensorflow/lite/tools/pip_package/build_pip_package_with_cmake.sh native
cd -
