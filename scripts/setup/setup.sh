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
arch=$(uname -m)
if [[ $arch == x86_64 ]]; then
    echo "X86_64 Architecture"
else
    echo 'Processor Architecture must be x86_64'
    echo 'Processor Architecture "'$arch'" is not supported'
return
fi

REL=11_02_12_00
echo "Version $REL"

CURRDIR=`pwd`
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TIDL_TOOLS_BASE_PATH=$SCRIPTDIR/../../tools
mkdir -p ${TIDL_TOOLS_BASE_PATH}

# Check command execution status
check_status()
{
    if [ $? -ne 0 ]; then
        cd ${CURRDIR}
        echo "ERROR: $1"
        echo "Exiting with code $?"
        exit $?
    fi
}

skip_model_optimizer=0
skip_cpp_deps=0
skip_data=0

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    --skip_model_optimizer)
    skip_model_optimizer=1
    ;;
    --skip_cpp_deps)
    skip_cpp_deps=1
    ;;
    --skip_data)
    skip_data=1
    ;;
    -h|--help)
    echo Usage: $0 [options]
    echo
    echo Options,
    echo --skip_model_optimizer     Skip installing model optimizer python package
    echo --skip_cpp_deps            Skip downloading dependencies for CPP examples
    echo --skip_data                Skip downloading out-of-box models and inputs
    exit 0
    ;;
esac
shift
done
set -- "${POSITIONAL[@]}"

# Check python version
version_match=`python3 -c 'import sys;r=0 if sys.version_info >= (3,6) else 1;print(r)'`
if [ $version_match -ne 0 ]; then
    echo 'python version must be >= 3.6'
return
fi

# Check if CPU or GPU tools
if [ -z "$TIDL_TOOLS_TYPE" ];then
    echo "Defaulting to CPU tools"
    tidl_gpu_tools=0
else
    echo "TIDL_TOOLS_TYPE set to :$TIDL_TOOLS_TYPE"
    if [ $TIDL_TOOLS_TYPE == GPU ];then
        tidl_gpu_tools=1
    else
        tidl_gpu_tools=0
    fi
fi

# This release (11_02_12_00) supports J722S only
SOC=${SOC^^}
SOC=${SOC:-"J722S"}
case "$SOC" in
  J722S)
    ALL_SOCS=("J722S")
    ;;
  AM67A|TDA4AEN)
    SOC="J722S"
    ALL_SOCS=("J722S")
    ;;
  *)
    echo "Invalid SOC $SOC defined. Release $REL only supports J722S | TDA4AEN | AM67A."
    return
    ;;
esac
echo
echo "Using SOC: ${SOC}"

# Python packages setup
cd ${SCRIPTDIR}
echo
echo '******************* INSTALLING REQUIRED PYTHON PACKAGES ******************'

pip3 install pybind11[global]
pip3 install -r ./requirements_pc.txt
check_status "Failed to install packages from requirements_pc.txt"

echo '******************* REQUIRED PYTHON PACKAGES INSTALLED *******************'

echo
echo '******************** INSTALLING OSRT PYTHON PACKAGES ********************'
echo "Installing: onnxruntime python wheel"
pip3 install --quiet https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/onnxruntime_tidl-1.15.0-cp310-cp310-linux_x86_64.whl
check_status "Failed to install onnxruntime_tidl wheel"

echo "Installing: tflite python wheel"
pip3 install --quiet https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tflite_runtime-2.12.0-cp310-cp310-linux_x86_64.whl
check_status "Failed to install tflite_runtime wheel"

echo "Installing: tidlruntime python wheel"
pip3 install --quiet https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tidlruntime-0.1.0-cp310-cp310-linux_x86_64.whl
check_status "Failed to install tidlruntime wheel"

echo "Installing: tvm python wheel"
pip3 install --quiet https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tvm-0.18.0-cp310-cp310-linux_x86_64.whl
check_status "Failed to install tvm wheel"

echo '******************** OSRT PYTHON PACKAGES INSTALLED ********************'
cd ${SCRIPTDIR}

# Graph optimizer tool setup
cd ${SCRIPTDIR}/../../osrt-model-tools
if [[ $skip_model_optimizer -eq 0 ]]; then
    echo
    echo '*********************** DOWNLOADING MODEL OPTIMIZER **********************'
    source ./setup.sh
    
    echo '*********************** MODEL OPTIMIZER DOWNLOADED ***********************'
fi
cd ${SCRIPTDIR}

# TIDL TOOLS setup 
cd ${TIDL_TOOLS_BASE_PATH}
echo
echo '************************* DOWNLOADING TIDL_TOOLS *************************'
# Loop over all SOCs in the ALL_SOCS array
for current_soc in "${ALL_SOCS[@]}"; do
    
    if [ "${current_soc}" == "AM62" ]; then
        continue
    fi

    echo
    echo "Processing SOC: ${current_soc}"
    
    TIDL_TOOLS_PATH=${TIDL_TOOLS_BASE_PATH}/${current_soc^^}/
    mkdir -p ${TIDL_TOOLS_PATH}

    if [ -d ${TIDL_TOOLS_PATH}/tidl_tools ]; then
        echo "[WARNING] ${TIDL_TOOLS_PATH} already has tidl_tools present. Skip downloading."
        echo "          To download again, please remove ${TIDL_TOOLS_PATH}/tidl_tools"
        continue
    fi

    cd ${TIDL_TOOLS_PATH}
    rm -rf tidl_tools.tar.gz tidl_tools_gpu.tar.gz tidl_tools 2>/dev/null

    if [ $tidl_gpu_tools -eq 1 ]; then
        TOOLS_LINK=https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/TIDL_TOOLS/${current_soc^^}/tidl_tools_gpu.tar.gz
        echo "Downloading GPU TIDL TOOLS for ${current_soc^^} in ${TIDL_TOOLS_PATH}..."
        echo "Download link : ${TOOLS_LINK}"
        wget --quiet $TOOLS_LINK
        check_status "Failed to download GPU TIDL TOOLS for ${current_soc^^}"

        tar -xzf tidl_tools_gpu.tar.gz
        rm tidl_tools_gpu.tar.gz
    else
        TOOLS_LINK=https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/TIDL_TOOLS/${current_soc^^}/tidl_tools.tar.gz
        echo "Downloading CPU TIDL TOOLS for ${current_soc^^} in ${TIDL_TOOLS_PATH}..."
        echo "Download link : ${TOOLS_LINK}"
        wget --quiet $TOOLS_LINK
        check_status "Failed to download CPU TIDL TOOLS for ${current_soc^^}"

        tar -xzf tidl_tools.tar.gz        
        rm tidl_tools.tar.gz
    fi

    cd tidl_tools
    if [[ ! -L libvx_tidl_rt.so.1.0 && ! -f libvx_tidl_rt.so.1.0 ]]; then
        ln -s libvx_tidl_rt.so libvx_tidl_rt.so.1.0
    fi
    cd ${TIDL_TOOLS_BASE_PATH}
done
echo '************************* TIDL_TOOLS DOWNLOADED *************************'
cd ${SCRIPTDIR}

# CPP OSRT DEPS
cd ${TIDL_TOOLS_BASE_PATH}
if [ $skip_cpp_deps -eq 0 ]; then
    echo
    echo '*************************** DOWNLOADING OSRT CPP DEPS *************************'

    cd ${TIDL_TOOLS_BASE_PATH}
    rm -rf osrt_deps 2>/dev/null
    mkdir -p osrt_deps    

    cd osrt_deps

    # ONNXRUNTIME
    OSRT_CPP_DEP_LINK=https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/onnx_1.15.0_x86_u22.tar.gz
    echo "Installing: onnxruntime cpp deps"
    echo "Download link : ${OSRT_CPP_DEP_LINK}"
    rm -rf onnx_1.15.0_x86_u22.tar.gz onnx_1.15.0_x86_u22 2>/dev/null
    wget --quiet ${OSRT_CPP_DEP_LINK}
    check_status "Failed to download onnxruntime cpp deps"
    tar -xf onnx_1.15.0_x86_u22.tar.gz    
    cd onnx_1.15.0_x86_u22
    if [ ! -f libonnxruntime.so ];then
        ln -s libonnxruntime.so.1.15.0 libonnxruntime.so
    fi
    if [ ! -f libonnxruntime.so.1.15.0 ];then
        ln -s libonnxruntime.so libonnxruntime.so.1.15.0
    fi
    cd ../
    rm -rf onnx_1.15.0_x86_u22.tar.gz

    # TFLITE
    TFLITE_CPP_DEP_LINK=https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tflite_2.12_x86_u22.tar.gz
    echo "Downloading: tflite cpp deps"
    echo "Download link : ${TFLITE_CPP_DEP_LINK}"
    rm -rf tflite_2.12_x86_u22.tar.gz tflite_2.12_x86_u22 2>/dev/null
    wget --quiet ${TFLITE_CPP_DEP_LINK}
    check_status "Failed to download tflite cpp deps"
    tar -xf tflite_2.12_x86_u22.tar.gz    
    rm -rf tflite_2.12_x86_u22.tar.gz 

    # TVM
    echo "Setting up TVM cpp deps"
    tvm_python_module_dir=$(python3  << EOF
import tvm
import os
print(os.path.dirname(tvm.__file__))
EOF
)
    check_status "Failed to get TVM python module directory"

    ln -sf "$tvm_python_module_dir" tvm_0.18.0_x86_u22
    check_status "Failed to create symbolic link for TVM"

    echo '*************************** CPP DEPS DOWNLOADED **************************'

    echo
    echo '*************************** CLONING AND BUILDING CNPY *************************'
    # Install and build cnpy for cpp numpy dependency
    cd ${TIDL_TOOLS_BASE_PATH}

    CNPY_GIT=https://github.com/rogersce/cnpy.git
    echo "Cloning and building: cnpy"
    echo "Clone link : ${CNPY_GIT}"
    rm -rf cnpy 2>/dev/null
    git clone ${CNPY_GIT}
    cd cnpy
    mkdir build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=${TIDL_TOOLS_BASE_PATH}/cnpy
    make
    make install

    echo '*************************** CNPY BUILD DONE **************************'
fi
cd ${SCRIPTDIR}

echo
echo '************************* INSTALLING ARM GCC COMPILER *************************'
cd ${TIDL_TOOLS_BASE_PATH}
if [ ! -d arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu ];then
    wget --quiet https://developer.arm.com/-/media/Files/downloads/gnu/13.2.rel1/binrel/arm-gnu-toolchain-13.2.rel1-x86_64-aarch64-none-linux-gnu.tar.xz
    check_status "Failed to download ARM GNU TOOLCHAIN"
    tar -xf arm-gnu-toolchain-13.2.rel1-x86_64-aarch64-none-linux-gnu.tar.xz
    rm -rf arm-gnu-toolchain-13.2.rel1-x86_64-aarch64-none-linux-gnu.tar.xz
else
    echo "Skipping arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu download: found at $(pwd)/arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu"
fi
cd ${SCRIPTDIR}

echo
echo '************************* INSTALLING C7X COMPILER *************************'

cd ${TIDL_TOOLS_BASE_PATH}
if [ ! -d ti-cgt-c7000_5.0.0.LTS ];then
    wget --quiet https://dr-download.ti.com/software-development/ide-configuration-compiler-or-debugger/MD-707zYe3Rik/5.0.0.LTS/ti_cgt_c7000_5.0.0.LTS_linux-x64_installer.bin
    check_status "Failed to download C7X compiler installer"
    chmod +x ti_cgt_c7000_5.0.0.LTS_linux-x64_installer.bin
    ./ti_cgt_c7000_5.0.0.LTS_linux-x64_installer.bin --mode unattended --installdir $(pwd)
    rm -rf ti_cgt_c7000_5.0.0.LTS_linux-x64_installer.bin
else
    echo "Skipping ti-cgt-c7000_5.0.0.LTS download: found at $(pwd)/ti-cgt-c7000_5.0.0.LTS"
fi

# Download out-of-box data
if [ $skip_data -eq 0 ]; then
    echo
    echo '************************* DOWNLOADING OUT-OF-BOX MODELS AND INPUTS *************************'

    DATA_DIR="${SCRIPTDIR}/../../runtimes/examples"
    if [ -d "$DATA_DIR/data" ]; then
        echo "  ${DATA_DIR}/data already exists, skipping download"
    else
        cd ${DATA_DIR}
        DATA_LINK=https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/Data/data.tar.gz
        echo "Downloading: out-of-box data"
        echo "Download link : ${DATA_LINK}"
        wget --quiet ${DATA_LINK}
        check_status "Failed to download out-of-box data"
        tar -xf data.tar.gz    
        rm -rf data.tar.gz 
        echo '************************* MODELS DOWNLOADED *************************'
    fi
fi

cd ${SCRIPTDIR}

echo
echo "Setup Done"
