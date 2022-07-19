#! /bin/bash
# This script should be run inside the docker host machine 
# Outputs:
# - This script will clone required files for corresponding xxx_build.sh file 

git clone --depth 1 --single-branch -b v2.8.0 https://github.com/tensorflow/tensorflow.git tensorflow_src



