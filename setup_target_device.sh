#!/bin/bash

# Copyright (c) 2024, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

######################################################################

REL=10_00_01_00
SCRIPTDIR=`pwd`
TARGET_FS_PATH=/


if [ `arch` == "aarch64" ]; then
    echo "Installing dependedcies at $TARGET_FS_PATH"

    cd $TARGET_FS_PATH/$HOME

    if [ ! -d arago_j7_pywhl ];then
        mkdir arago_j7_pywhl
    fi
    if [ ! -d required_libs ];then
        mkdir required_libs
    fi

    cleanup() {
        rm -rf $TARGET_FS_PATH/$HOME/arago_j7_pywhl
        rm -rf $TARGET_FS_PATH/$HOME/required_libs
    }

    update_dlr() {
        cd $TARGET_FS_PATH/$HOME/arago_j7_pywhl
        wget --proxy off https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/ARAGO/dlr-1.13.0-py3-none-any.whl
        pip3 install --upgrade --force-reinstall dlr-1.13.0-py3-none-any.whl --disable-pip-version-check

        cd $TARGET_FS_PATH/$HOME/required_libs
        git clone -b master git://git.ti.com/processor-sdk-vision/arm-tidl.git
        cp arm-tidl/rt/inc/itidl_rt.h  $TARGET_FS_PATH/usr/include/
        cp arm-tidl/rt/inc/itvm_rt.h $TARGET_FS_PATH/usr/include/

        cd $TARGET_FS_PATH/$HOME/
    }

    update_onnx() {
        cd $TARGET_FS_PATH/$HOME/arago_j7_pywhl
        wget --proxy off https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/ARAGO/onnxruntime_tidl-1.14.0+10000005-cp310-cp310-linux_aarch64.whl
        pip3 install onnxruntime_tidl-1.14.0+10000005-cp310-cp310-linux_aarch64.whl --disable-pip-version-check
        
        cd $TARGET_FS_PATH/$HOME/required_libs

        if [ -d $TARGET_FS_PATH/usr/include/onnxruntime ];then
            echo "***** Skipping Onnxruntime setup: found /usr/include/onnxruntime *****"
            echo "***** To redo the setup delete: /usr/include/onnxruntime and run this script again *****"
        else
            wget --proxy off https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/ARAGO/onnx_1.14.0_aragoj7.tar.gz
            tar xf onnx_1.14.0_aragoj7.tar.gz
            rm onnx_1.14.0_aragoj7.tar.gz
            mv onnx_1.14.0_aragoj7/onnxruntime $TARGET_FS_PATH/usr/include/
            cp -r  onnx_1.14.0_aragoj7/libonnxruntime.so.1.14.0+10000005   $TARGET_FS_PATH/usr/lib/
            cd   $TARGET_FS_PATH/usr/lib/
            ln -sf libonnxruntime.so.1.14.0+10000005 libonnxruntime.so
        fi

        cd $TARGET_FS_PATH/$HOME/
    }

    update_tensorflow() {
        cd $TARGET_FS_PATH/$HOME/arago_j7_pywhl
        wget --proxy off https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/ARAGO/tflite_runtime-2.12.0-cp310-cp310-linux_aarch64.whl
        pip3 install --upgrade --force-reinstall tflite_runtime-2.12.0-cp310-cp310-linux_aarch64.whl --disable-pip-version-check
        
        cd $TARGET_FS_PATH/$HOME/required_libs

        if [ -d $TARGET_FS_PATH/usr/include/tensorflow ];then
            echo "***** Skipping Tensorflow setup: Found /usr/include/tensorflow *****"
            echo "***** To redo the setup delete: /usr/include/tensorflow and run this script again *****"
        elif [ -d $TARGET_FS_PATH/usr/lib/tflite_2.12 ];then
            echo "***** Skipping Tensorflow setup: Found /usr/lib/tflite_2.12 *****"
            echo "***** To redo the setup delete: /usr/lib/tflite_2.12 and run this script again *****"
        else
            wget --proxy off https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/ARAGO/tflite_2.12_aragoj7.tar.gz
            tar xf tflite_2.12_aragoj7.tar.gz
            rm tflite_2.12_aragoj7.tar.gz
            mv tflite_2.12_aragoj7/tensorflow  $TARGET_FS_PATH/usr/include
            mv tflite_2.12_aragoj7/tflite_2.12  $TARGET_FS_PATH/usr/lib/
            cp tflite_2.12_aragoj7/libtensorflow-lite.a $TARGET_FS_PATH/usr/lib/
        fi

        cd $TARGET_FS_PATH/$HOME/
    }

    update_numpy() {
        pip3 install --upgrade --force-reinstall --no-cache-dir numpy==1.23.0 --disable-pip-version-check
    }

    update_dlr
    update_onnx
    update_tensorflow
    update_numpy
    cleanup
    
    cd $SCRIPTDIR
fi