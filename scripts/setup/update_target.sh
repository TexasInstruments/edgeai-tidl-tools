#!/bin/bash

# Copyright (c) 2026, Texas Instruments
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
if [ `arch` != "aarch64" ]; then
    echo "The script must be invoked on aarch64 system"
    exit -1
fi

SCRIPTDIR=`pwd`
TARGET_FS_PATH=/

REL="11_02_05_00"

SOC=${SOC:-'null'}
TISDK_IMAGE=${TISDK_IMAGE:-'null'}
SDK_VERSION=${SDK_VERSION:-'null'}
UPDATE_OSRT_COMPONENTS=${UPDATE_OSRT_COMPONENTS:-1}
UPDATE_FIRMWARE_AND_LIB=${UPDATE_FIRMWARE_AND_LIB:-1}

SOC=${SOC^^}

echo "========================================================================="
echo "REL: ${REL}"
echo "SOC: ${SOC}"
echo "TISDK_IMAGE: ${TISDK_IMAGE}"
echo "SDK_VERSION: ${SDK_VERSION}"
echo "UPDATE_OSRT_COMPONENTS: ${UPDATE_OSRT_COMPONENTS}"
echo "UPDATE_FIRMWARE_AND_LIB: ${UPDATE_FIRMWARE_AND_LIB}"
echo "========================================================================="


verify_env() {
    if [ "$REL" != "11_02_05_00" ]; then
        echo "Cannot invoke this script with version $REL. This is not a backward compatible release."
        return 1
    fi

    case "$SOC" in
      AM62|AM62A|J721E|J721S2|J784S4|J722S)
        SOC=$SOC
        ;;
      AM68PA|TDA4VM)
        SOC="J721E"
        ;;
      AM68A|TDA4VL)
        SOC="J721S2"
        ;; 
      AM69A|TDA4VH)
        SOC="J784S4"
        ;;
      AM67A|TDA4AEN)
        SOC="J722S"
        ;;
      *)
        echo "Invalid SOC $SOC defined. Allowed values are:"
        echo "AM62, AM62A, (J721E or TDA4VM), (J721S2 or TDA4VL or AM68A), (J784S4 or TDA4VH or AM69A) and (J722S or TDA4AEN or AM67A)"
        return
        ;;
    esac


    if [ "$TISDK_IMAGE" != "adas" ] && [ "$TISDK_IMAGE" != "edgeai" ]; then
        echo
        echo "Incorrect TISDK_IMAGE defined: $TISDK_IMAGE"
        echo "Run either of below commands"
        echo "export TISDK_IMAGE=edgeai"
        echo "export TISDK_IMAGE=adas"
        return 1
    fi

    if [ "$SDK_VERSION" != "11_1" ] && [ "$SDK_VERSION" != "11_0" ]; then
        echo
        echo "Incorrect SDK_VERSION defined: $SDK_VERSION"
        echo "Allowed values for SDK_VERSION is 11_1 or 11_0"
        return 1
    fi

    if [ "$SOC" == "AM62A" ]; then
        if [ "$SDK_VERSION" == "11_0" ]; then
            echo
            echo "SDK_VERSION 11_0 does not exist for AM62A"
            return 1
        fi
        if [ "$TISDK_IMAGE" == "adas" ]; then
            echo
            echo "AM62A does not have ADAS SDK. Use EDGEAI"
            return 1
        fi
    fi

    if [ "$TISDK_IMAGE" == "edgeai" ] && [ "$SDK_VERSION" == "11_1" ]; then
        if [ "$SOC" == "J721S2" ] || [ "$SOC" == "J784S4" ] || [ "$SOC" == "J722S" ] || [ "$SOC" == "J721E" ]; then
            echo
            echo "$SOC does not have 11_1 EDGEAI SDK"
            return 1
        fi
    fi

    return 0
}

update_arm_tidl_headers() {
    echo
    echo "==================== Updating Headers ===================="
    cd $TARGET_FS_PATH/$HOME/required_libs
    git clone -b master git://git.ti.com/processor-sdk-vision/arm-tidl.git
    if [ "$?" -eq "0" ]; then
        # Backup old files
        if [ ! -f "$TARGET_FS_PATH/usr/include/itidl_rt.h.bkp" ]; then
            mv $TARGET_FS_PATH/usr/include/itidl_rt.h $TARGET_FS_PATH/usr/include/itidl_rt.h.bkp
        fi
        if [ ! -f "$TARGET_FS_PATH/usr/include/itidl_io.h.bkp" ]; then
            mv $TARGET_FS_PATH/usr/include/itidl_io.h $TARGET_FS_PATH/usr/include/itidl_io.h.bkp
        fi
        if [ ! -f "$TARGET_FS_PATH/usr/include/itvm_rt.h.bkp" ]; then
            mv $TARGET_FS_PATH/usr/include/itvm_rt.h $TARGET_FS_PATH/usr/include/itvm_rt.h.bkp
        fi
        cp arm-tidl/rt/inc/itidl_rt.h  $TARGET_FS_PATH/usr/include/
        cp arm-tidl/rt/inc/itidl_io.h $TARGET_FS_PATH/usr/include/
        cp arm-tidl/rt/inc/itvm_rt.h $TARGET_FS_PATH/usr/include/

        cp -r arm-tidl/rt/inc/*  $TARGET_FS_PATH/usr/include/processor_sdk/tidl_j7/arm-tidl/rt/inc/
        cp -r arm-tidl/tiovx_kernels/include/*  $TARGET_FS_PATH/usr/include/processor_sdk/tidl_j7/arm-tidl/tiovx_kernels/include/
    else
        echo "[WARN] Failed to clone arm-tidl" 
    fi
}

update_cnpy() {
    if [ -f "$TARGET_FS_PATH/usr/lib/libcnpy.so" ]; then
        echo
        echo "[WARN] CNPY already present (/usr/lib/libcnpy.so). Skipping it."
    else
        echo
        echo "==================== Installing CNPY ===================="
        cd $TARGET_FS_PATH/$HOME/required_libs
        git clone -b master https://github.com/rogersce/cnpy.git
        if [ "$?" -eq "0" ]; then
            cd cnpy
            mkdir build && cd build
            cmake .. -DCMAKE_INSTALL_PREFIX=$TARGET_FS_PATH/usr/
            make
            make install
        else
            echo "[WARN] Failed to clone cnpy" 
        fi
    fi
}

update_osrt_components() {
    echo
    echo "==================== Updating OSRT Components ===================="

    echo
    echo "==================== Updating onnxruntime wheel ===================="
    onnx_wheel=onnxruntime_tidl-1.15.0-cp312-cp312-linux_aarch64.whl
    cd $TARGET_FS_PATH/$HOME/arago_j7_pywhl
    wget --proxy off https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/ARAGO/$SDK_VERSION/$onnx_wheel
    pip3 install $onnx_wheel --disable-pip-version-check

    echo
    echo "==================== Updating onnxruntime library ===================="
    cd $TARGET_FS_PATH/$HOME/required_libs
    onnx_tar=onnx_1.15.0_aragoj7
    wget --proxy off https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/ARAGO/$SDK_VERSION/$onnx_tar.tar.gz
    if [ "$?" -eq "0" ]; then
        tar xf $onnx_tar.tar.gz && rm $onnx_tar.tar.gz

        if [ ! -d "$TARGET_FS_PATH/usr/include/onnxruntime.bkp" ]; then
            mv $TARGET_FS_PATH/usr/include/onnxruntime $TARGET_FS_PATH/usr/include/onnxruntime.bkp
        fi
        rm -rf $TARGET_FS_PATH/usr/include/onnxruntime
        mv $onnx_tar/onnxruntime $TARGET_FS_PATH/usr/include/

        if [ ! -f "$TARGET_FS_PATH/usr/lib/libonnxruntime.so.1.15.0.bkp" ]; then
            mv $TARGET_FS_PATH/usr/lib/libonnxruntime.so.1.15.0 $TARGET_FS_PATH/usr/lib/libonnxruntime.so.1.15.0.bkp
        fi
        cp -r  $onnx_tar/libonnxruntime.so.1.15.0   $TARGET_FS_PATH/usr/lib/
        cd   $TARGET_FS_PATH/usr/lib/
        ln -sf libonnxruntime.so.1.15.0 libonnxruntime.so
    fi

    echo
    echo "==================== Updating tflite wheel ===================="
    cd $TARGET_FS_PATH/$HOME/arago_j7_pywhl
    tfl_wheel=tflite_runtime-2.12.0-cp312-cp312-linux_aarch64.whl
    wget --proxy off https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/ARAGO/$SDK_VERSION/$tfl_wheel
    pip3 install --upgrade --force-reinstall $tfl_wheel --disable-pip-version-check

    echo
    echo "==================== Updating tflite library ===================="
    cd $TARGET_FS_PATH/$HOME/required_libs
    tfl_tar=tflite_2.12_aragoj7
    wget --proxy off https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/ARAGO/$SDK_VERSION/$tfl_tar.tar.gz
    if [ "$?" -eq "0" ]; then
        tar xf $tfl_tar.tar.gz && rm $tfl_tar.tar.gz
        
        if [ ! -d "$TARGET_FS_PATH/usr/include/tensorflow.bkp" ]; then
            mv $TARGET_FS_PATH/usr/include/tensorflow $TARGET_FS_PATH/usr/include/tensorflow.bkp
        fi
        rm -rf $TARGET_FS_PATH/usr/include/tensorflow
        mv $tfl_tar/tensorflow  $TARGET_FS_PATH/usr/include

        if [ ! -d "$TARGET_FS_PATH/usr/lib/tflite_2.12.bkp" ]; then
            mv $TARGET_FS_PATH/usr/lib/tflite_2.12 $TARGET_FS_PATH/usr/lib/tflite_2.12.bkp
        fi
        rm -rf $TARGET_FS_PATH/usr/lib/tflite_2.12
        mv $tfl_tar/tflite_2.12  $TARGET_FS_PATH/usr/lib/

        if [ ! -f "$TARGET_FS_PATH/usr/lib/libtensorflow-lite.a.bkp" ]; then
            mv $TARGET_FS_PATH/usr/lib/libtensorflow-lite.a $TARGET_FS_PATH/usr/lib/libtensorflow-lite.a.bkp
        fi
        cp $tfl_tar/libtensorflow-lite.a $TARGET_FS_PATH/usr/lib/
    fi

    echo
    echo "==================== Updating tvm wheel ===================="
    cd $TARGET_FS_PATH/$HOME/arago_j7_pywhl
    tvm_wheel=tvm-0.18.0-cp312-cp312-linux_aarch64.whl
    wget --proxy off https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/ARAGO/$SDK_VERSION/$tvm_wheel
    pip3 install --upgrade --force-reinstall $tvm_wheel --disable-pip-version-check

    echo
    echo "==================== Updating tvm library ===================="
    cd $TARGET_FS_PATH/$HOME/required_libs
    if [ ! -d "$TARGET_FS_PATH/usr/include/tvm.bkp" ]; then
        mv $TARGET_FS_PATH/usr/include/tvm $TARGET_FS_PATH/usr/include/tvm.bkp
    fi
    rm -rf $TARGET_FS_PATH/usr/include/tvm
    mkdir -p $TARGET_FS_PATH/usr/include/tvm/tvm
    cd $TARGET_FS_PATH/usr/lib/python3.12/site-packages/tvm
    cp --parents $(find . -name "*.h*") $TARGET_FS_PATH/usr/include/tvm/tvm/
    cd -
    ln -sf $TARGET_FS_PATH/usr/lib/python3.12/site-packages/tvm/libtvm.so $TARGET_FS_PATH/usr/lib/libtvm.so
    ln -sf $TARGET_FS_PATH/usr/lib/python3.12/site-packages/tvm/libtvm_runtime.so $TARGET_FS_PATH/usr/lib/libtvm_runtime.so

    echo
    echo "==================== Updating tidlruntime wheel ===================="
    cd $TARGET_FS_PATH/$HOME/arago_j7_pywhl
    tidlrt_wheel=tidlruntime-0.1.0-cp312-cp312-linux_aarch64.whl
    wget --proxy off https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/ARM_LINUX/ARAGO/$SDK_VERSION/$tidlrt_wheel
    pip3 install --upgrade --force-reinstall $tidlrt_wheel --disable-pip-version-check

    echo
    echo "==================== Updating tidlruntime library ===================="
    cd $TARGET_FS_PATH/$HOME/required_libs
    ln -sf $TARGET_FS_PATH/usr/lib/python3.12/site-packages/tidlruntime/include $TARGET_FS_PATH/usr/include/tidlruntime
    ln -sf $TARGET_FS_PATH/usr/lib/python3.12/site-packages/tidlruntime/lib/libtidlruntime.a $TARGET_FS_PATH/usr/lib/libtidlruntime.a

    echo
    echo "==================== Reinstalling Numpy(1.26.4) ===================="
    pip3 install --upgrade --force-reinstall --no-cache-dir numpy==1.26.4 --disable-pip-version-check

    cd $TARGET_FS_PATH/$HOME/
}

update_firmware_and_lib() {
    FIRMWARE_TARBALL=https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/FIRMWARES/${SOC^^}/$TISDK_IMAGE/$SDK_VERSION/firmware.tar.gz
    TIDL_LIB_TARBALL=https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/FIRMWARES/${SOC^^}/$TISDK_IMAGE/$SDK_VERSION/tidl_lib.tar.gz

    echo
    echo "==================== Updating C7X firmware ===================="
    cd $TARGET_FS_PATH/$HOME/updated_firmware_and_lib

    if [ "${SOC}" == "AM62A" ]; then
        FIRMWARE_PATH=$TARGET_FS_PATH/lib/firmware/ti-ipc/am62axx
    else
        if [ "${TISDK_IMAGE}" == "edgeai" ]; then
            FIRMWARE_PATH=$TARGET_FS_PATH/lib/firmware/vision_apps_eaik
        else
            FIRMWARE_PATH=$TARGET_FS_PATH/lib/firmware/vision_apps_evm
        fi
    fi

    echo "FIRMWARE_PATH: ${FIRMWARE_PATH}"
    wget --proxy off $FIRMWARE_TARBALL
    if [ "$?" -ne "0" ]; then
        echo "Downloading firmware failed. Please check if $FIRMWARE_TARBALL is valid"
    fi
    tar -xf firmware.tar.gz && rm firmware.tar.gz

    if [ "${SOC}" == "AM62A" ]; then
        cd firmware/ti-ipc/am62axx
    else
        if [ "${TISDK_IMAGE}" == "edgeai" ]; then
            cd firmware/vision_apps_eaik
        else
            cd firmware/vision_apps_evm
        fi
    fi

    for file in `find ./ -name "*c7*.out*"`; do
        echo "Replacing ${file}"
        if [ -f $FIRMWARE_PATH/$file ]; then
            if [ ! -f "$FIRMWARE_PATH/$file.bkp" ]; then
                mv $FIRMWARE_PATH/$file $FIRMWARE_PATH/$file.bkp
            fi
        else
            echo "WARNING: $file not used in $FIRMWARE_PATH. Still copying"
        fi
        cp $file $FIRMWARE_PATH/
    done


    echo
    echo "==================== Updating TIDL libraries ===================="
    cd $TARGET_FS_PATH/$HOME/updated_firmware_and_lib
    wget --proxy off $TIDL_LIB_TARBALL
    if [ "$?" -ne "0" ]; then
        echo "Downloading tidl_lib failed. Please check if $TIDL_LIB_TARBALL is valid"
    fi
    tar -xf tidl_lib.tar.gz && rm tidl_lib.tar.gz
    cd tidl_lib
    for file in *; do
        echo "Replacing ${file}"
        if [ -f $TARGET_FS_PATH/usr/lib/$file ]; then
            if [ ! -f "$TARGET_FS_PATH/usr/lib/$file.bkp" ]; then
                mv $TARGET_FS_PATH/usr/lib/$file $TARGET_FS_PATH/usr/lib/$file.bkp
            fi
        else
            echo "WARNING: $file not used. Still copying"
        fi
        cp $file $TARGET_FS_PATH/usr/lib
    done
}

verify_env
if [ "$?" -eq "0" ]; then
    rm -rf $TARGET_FS_PATH/$HOME/arago_j7_pywhl
    rm -rf $TARGET_FS_PATH/$HOME/required_libs
    rm -rf $TARGET_FS_PATH/$HOME/updated_firmware_and_lib
    mkdir -p $TARGET_FS_PATH/$HOME/arago_j7_pywhl
    mkdir -p $TARGET_FS_PATH/$HOME/required_libs
    mkdir -p $TARGET_FS_PATH/$HOME/updated_firmware_and_lib

    # Update arm-tidl headers
    update_arm_tidl_headers

    # Update cnpy headers
    update_cnpy

    # Update osrt wheels and libraries
    if [ "${UPDATE_OSRT_COMPONENTS}" -eq "1" ]; then
        update_osrt_components
    fi

    # Update c7x firmware and libraries
    if [ "${UPDATE_FIRMWARE_AND_LIB}" -eq "1" ]; then
        if [ "$SOC" != "AM62" ]; then
            update_firmware_and_lib
            echo "[INFO] C7x Firmware and Libs are updated. Please reeboot the device." 
        else
            echo
            echo "[WARN] C7x Firmware and Libs are not applicable for am62" 
        fi
    fi

    rm -rf $TARGET_FS_PATH/$HOME/arago_j7_pywhl
    rm -rf $TARGET_FS_PATH/$HOME/required_libs
    rm -rf $TARGET_FS_PATH/$HOME/updated_firmware_and_lib
fi

cd $SCRIPTDIR
