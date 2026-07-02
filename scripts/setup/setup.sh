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

REL=11_02_14_00
echo "Version $REL"

CURRDIR=`pwd`
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TIDL_TOOLS_BASE_PATH=$SCRIPTDIR/../../tools
mkdir -p ${TIDL_TOOLS_BASE_PATH}

# Check command execution status
check_status()
{
    local code=$?
    if [ $code -ne 0 ]; then
        cd ${CURRDIR}
        echo "ERROR: $1"
        echo "Exiting with code $code"
        exit $code
    fi
}

skip_model_optimizer=0
skip_cpp_deps=0
skip_data=0
skip_onnxruntime=0
skip_tflite=0
skip_tvm=0
skip_tidlruntime=0

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
    --skip_onnxruntime)
    skip_onnxruntime=1
    ;;
    --skip_tflite)
    skip_tflite=1
    ;;
    --skip_tvm)
    skip_tvm=1
    ;;
    --skip_tidlruntime)
    skip_tidlruntime=1
    ;;
    -h|--help)
    echo Usage: $0 [options]
    echo
    echo Options,
    echo --skip_model_optimizer     Skip installing model optimizer python package
    echo --skip_cpp_deps            Skip downloading dependencies for CPP examples
    echo --skip_data                Skip downloading out-of-box models and inputs
    echo --skip_onnxruntime         Skip installing onnxruntime python wheel and CPP deps
    echo --skip_tflite              Skip installing tflite python wheel and CPP deps
    echo --skip_tvm                 Skip installing tvm python wheel and CPP deps
    echo --skip_tidlruntime         Skip installing tidlruntime python wheel
    exit 0
    ;;
esac
shift
done
set -- "${POSITIONAL[@]}"

# Check python version
version_match=`python3 -c 'import sys;r=0 if (sys.version_info >= (3,10) and sys.version_info < (3,11))  else 1;print(r)'`
if [ $version_match -ne 0 ]; then
    echo 'python version must be 3.10'
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

# Define all supported SOCs
ALL_SOCS=("AM62" "AM62A" "J721E" "J721S2" "J784S4" "J722S")
SOC=${SOC^^}
if [ ! -z "$SOC" ];then
    case "$SOC" in
      AM62|AM62A|J721E|J721S2|J784S4|J722S)
        ALL_SOCS=("$SOC")
        ;;
      AM68PA|TDA4VM)
        ALL_SOCS=("J721E")
        ;;
      AM68A|TDA4VL)
        ALL_SOCS=("J721S2")
        ;; 
      AM69A|TDA4VH)
        ALL_SOCS=("J784S4")
        ;;
      AM67A|TDA4AEN)
        ALL_SOCS=("J722S")
        ;;
      *)
        echo "Invalid SOC $SOC defined. Allowed values are:"
        echo "AM62, AM62A, (J721E or TDA4VM), (J721S2 or TDA4VL or AM68A), (J784S4 or TDA4VH or AM69A) and (J722S or TDA4AEN or AM67A)"
        return
        ;;
    esac
    echo
    echo "Using specified SOC=${SOC}"
fi

# Basic Python packages setup
cd ${SCRIPTDIR}
echo
echo '******************* INSTALLING BASIC PYTHON PACKAGES ******************'

pip3 install pybind11[global]
pip3 install -r ./requirements_pc.txt
check_status "Failed to install packages from requirements_pc.txt"
pip3 install -r ../../test/tidl_unit/requirements.txt
check_status "Failed to install packages from ${SCRIPTDIR}/../../test/tidl_unit/requirements.txt"

echo '******************* BASIC PYTHON PACKAGES INSTALLED *******************'

# Graph optimizer tool setup
if [[ $skip_model_optimizer -eq 0 ]]; then
    cd ${SCRIPTDIR}/../../model-tools/tidl-onnx-model-optimizer
    echo
    echo '*********************** INSTALLING TIDL-ONNX-MODEL-OPTIMIZER **********************'
    source ./setup.sh
    echo '*********************** TIDL-ONNX-MODEL-OPTIMIZER INSTALLED ***********************'
fi

if [[ $skip_model_optimizer -eq 0 ]]; then
    cd ${SCRIPTDIR}/../../model-tools/osrt-model-tools
    echo
    echo '*********************** INSTALLING OSRT-MODEL-TOOLS **********************'
    source ./setup.sh
    echo '*********************** OSRT-MODEL-TOOLS INSTALLED ***********************'
fi

cd ${SCRIPTDIR}

# CNPY
if [ $skip_cpp_deps -eq 0 ]; then
    echo
    echo '*************************** BUILDING CNPY ****************************'
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
    cd ${SCRIPTDIR}
fi

# Download out-of-box data
if [ $skip_data -eq 0 ]; then
    echo
    echo '************************* DOWNLOADING OUT-OF-BOX MODELS AND INPUTS *************************'

    DATA_DIR="${SCRIPTDIR}/../../runtimes/examples"
    if [ -d "$DATA_DIR/data" ]; then
        echo " ${DATA_DIR}/data already exists, skipping download"
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

# OSRT Installation
echo
echo '******************** INSTALLING OSRT PACKAGES ********************'

if [ $skip_cpp_deps -eq 0 ]; then
    rm -rf ${TIDL_TOOLS_BASE_PATH}/osrt_deps 2>/dev/null
fi

# ONNXRUNTIME
if [ $skip_onnxruntime -eq 0 ]; then

    # Python
    echo
    echo "[ONNXRUNTIME PYTHON]"
    OSRT_PYTHON_WHL_LINK=https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/onnxruntime_tidl-1.23.0-cp310-cp310-linux_x86_64.whl
    echo "Downloading and Installing: ${OSRT_PYTHON_WHL_LINK}"
    pip3 uninstall -y onnxruntime 2>/dev/null
    pip3 uninstall -y onnxruntime_tidl 2>/dev/null
    pip3 install --quiet ${OSRT_PYTHON_WHL_LINK}
    check_status "Failed to install ${OSRT_PYTHON_WHL_LINK}"

    # CPP
    if [ $skip_cpp_deps -eq 0 ]; then
        echo
        echo "[ONNXRUNTIME CPP]"
        mkdir -p ${TIDL_TOOLS_BASE_PATH}/osrt_deps
        cd ${TIDL_TOOLS_BASE_PATH}/osrt_deps
        OSRT_CPP_DEP_LINK=https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/onnx_1.23.0_x86_u22.tar.gz
        echo "Downloading and Setting : ${OSRT_CPP_DEP_LINK}"
        rm -rf onnx_1.23.0_x86_u22.tar.gz onnx_1.23.0_x86_u22 2>/dev/null
        wget --quiet ${OSRT_CPP_DEP_LINK}
        check_status "Failed to download onnxruntime cpp deps"
        tar -xf onnx_1.23.0_x86_u22.tar.gz
        cd onnx_1.23.0_x86_u22
        if [ ! -f libonnxruntime.so ];then
            ln -s libonnxruntime.so.1.23.0 libonnxruntime.so
        fi
        if [ ! -f libonnxruntime.so.1.23.0 ];then
            ln -s libonnxruntime.so libonnxruntime.so.1.23.0
        fi
        cd ../
        rm -rf onnx_1.23.0_x86_u22.tar.gz
        cd ${SCRIPTDIR}
    fi
else
    echo "Skipping: onnxruntime (--skip_onnxruntime)"
fi

# TFLITE
if [ $skip_tflite -eq 0 ]; then

    # Python
    echo
    echo "[TFLITE PYTHON]"
    TFLITE_PYTHON_WHL_LINK=https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tflite_runtime-2.12.0-cp310-cp310-linux_x86_64.whl
    echo "Downloading and Installing: ${TFLITE_PYTHON_WHL_LINK}"
    pip3 uninstall -y tflite_runtime 2>/dev/null
    pip3 install --quiet ${TFLITE_PYTHON_WHL_LINK}
    check_status "Failed to install ${TFLITE_PYTHON_WHL_LINK}"

    # CPP
    if [ $skip_cpp_deps -eq 0 ]; then
        echo
        echo "[TFLITE CPP]"
        mkdir -p ${TIDL_TOOLS_BASE_PATH}/osrt_deps
        cd ${TIDL_TOOLS_BASE_PATH}/osrt_deps
        TFLITE_CPP_DEP_LINK=https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tflite_2.12_x86_u22.tar.gz
        echo "Downloading and Setting : ${TFLITE_CPP_DEP_LINK}"
        rm -rf tflite_2.12_x86_u22.tar.gz tflite_2.12_x86_u22 2>/dev/null
        wget --quiet ${TFLITE_CPP_DEP_LINK}
        check_status "Failed to download tflite cpp deps"
        tar -xf tflite_2.12_x86_u22.tar.gz
        rm -rf tflite_2.12_x86_u22.tar.gz
        cd ${SCRIPTDIR}
    fi
else
    echo "Skipping: tflite (--skip_tflite)"
fi

# TVM
if [ $skip_tvm -eq 0 ]; then
    # Python
    echo
    echo "[TVM PYTHON]"
    TVM_PYTHON_WHL_LINK=https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tvm-0.18.0-cp310-cp310-linux_x86_64.whl
    echo "Downloading and Installing: ${TVM_PYTHON_WHL_LINK}"
    pip3 uninstall -y tvm 2>/dev/null
    pip3 install --quiet ${TVM_PYTHON_WHL_LINK}
    check_status "Failed to install ${TVM_PYTHON_WHL_LINK}"

    # CPP
    if [ $skip_cpp_deps -eq 0 ]; then
        echo
        echo "[TVM CPP]"
        if python3 -c "import tvm" 2>/dev/null; then
            tvm_python_module_dir=$(python3 << EOF
import tvm
import os
print(os.path.dirname(tvm.__file__))
EOF
)
            check_status "Failed to get TVM python module directory"
            mkdir -p ${TIDL_TOOLS_BASE_PATH}/osrt_deps
            cd ${TIDL_TOOLS_BASE_PATH}/osrt_deps
            ln -sf "$tvm_python_module_dir" tvm_0.18.0_x86_u22
            check_status "Failed to create symbolic link for TVM"
        else
            echo "WARNING: TVM Python module not found. Skipping TVM cpp deps setup."
        fi
    fi

    # ARM GCC COMPILER
    echo
    echo "[TVM ARM GCC COMPILER]"
    cd ${TIDL_TOOLS_BASE_PATH}
    if [ ! -d arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu ];then
        ARM_GCC_LINK=https://developer.arm.com/-/media/Files/downloads/gnu/13.2.rel1/binrel/arm-gnu-toolchain-13.2.rel1-x86_64-aarch64-none-linux-gnu.tar.xz
        echo "Downloading and Setting : ${ARM_GCC_LINK}"
        wget --quiet ${ARM_GCC_LINK}
        check_status "Failed to download ARM GNU TOOLCHAIN"
        tar -xf arm-gnu-toolchain-13.2.rel1-x86_64-aarch64-none-linux-gnu.tar.xz
        rm -rf arm-gnu-toolchain-13.2.rel1-x86_64-aarch64-none-linux-gnu.tar.xz
    else
        echo "Skipping arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu download: found at $(pwd)/arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu"
    fi

    # C7X COMPILER
    echo
    echo "[TVM C7X COMPILER]"
    if [ ! -d ti-cgt-c7000_5.0.0.LTS ];then
        C7X_INSTALLER_LINK=https://dr-download.ti.com/software-development/ide-configuration-compiler-or-debugger/MD-707zYe3Rik/5.0.0.LTS/ti_cgt_c7000_5.0.0.LTS_linux-x64_installer.bin
        echo "Downloading and Setting : ${C7X_INSTALLER_LINK}"
        wget --quiet ${C7X_INSTALLER_LINK}
        check_status "Failed to download C7X compiler installer"
        chmod +x ${C7X_INSTALLER_LINK##*/}
        ./${C7X_INSTALLER_LINK##*/} --mode unattended --installdir $(pwd)
        rm -rf ${C7X_INSTALLER_LINK##*/}
    else
        echo "Skipping ti-cgt-c7000_5.0.0.LTS download: found at $(pwd)/ti-cgt-c7000_5.0.0.LTS"
    fi

    cd ${SCRIPTDIR}
else
    echo "Skipping: tvm (--skip_tvm)"
fi

# TIDLRUNTIME
if [ $skip_tidlruntime -eq 0 ]; then
    echo
    echo "[TIDLRUNTIME PYTHON]"
    TIDLRUNTIME_PYTHON_WHL_LINK=https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tidlruntime-0.1.0-cp310-cp310-linux_x86_64.whl
    echo "Downloading and Installing: ${TIDLRUNTIME_PYTHON_WHL_LINK}"
    pip3 uninstall -y tidlruntime 2>/dev/null
    pip3 install --quiet ${TIDLRUNTIME_PYTHON_WHL_LINK}
    check_status "Failed to install ${TIDLRUNTIME_PYTHON_WHL_LINK}"
else
    echo "Skipping: tidlruntime (--skip_tidlruntime)"
fi

echo '******************** OSRT PACKAGES INSTALLED ********************'

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

    cd ${TIDL_TOOLS_PATH}
    if [ -d ${TIDL_TOOLS_PATH}/tidl_tools ]; then
        echo "[INFO] Existing tidl_tools found at ${TIDL_TOOLS_PATH}/tidl_tools. Re-downloading..."
        rm -rf tidl_tools
    fi
    rm -rf tidl_tools.tar.gz tidl_tools_gpu.tar.gz 2>/dev/null

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

echo
echo "Setup Done"
