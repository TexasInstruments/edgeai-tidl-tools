#!/bin/bash

# Copyright (c) 2025, Texas Instruments
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
pip_install_local()
{
    if [ -f $LOCAL_PATH/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1 ];then
        echo "Local file  found. Installing $LOCAL_PATH/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1"
        pip3 install --force-reinstall $LOCAL_PATH/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1
    else
        echo "Local file not found at $LOCAL_PATH/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1. Installing default  https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1"
        pip3 install https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1
    fi
}

cp_tidl_tools()
{
    if [ -f $LOCAL_PATH/TIDL_TOOLS/$1/tidl_tools.tar.gz ];then
        if [ $tidl_gpu_tools -eq 1 ];then
            echo "Local file  found. Copying $LOCAL_PATH/TIDL_TOOLS/$1/tidl_tools_gpu.tar.gz"
            cp $LOCAL_PATH/TIDL_TOOLS/$1/tidl_tools_gpu.tar.gz .
        else
            echo "Local file  found. Copying $LOCAL_PATH/TIDL_TOOLS/$1/tidl_tools.tar.gz"
            cp $LOCAL_PATH/TIDL_TOOLS/$1/tidl_tools.tar.gz .
        fi
    else
        if [ $tidl_gpu_tools -eq 1 ];then
            echo "Local file not found at $LOCAL_PATH/TIDL_TOOLS/$1/tidl_tools.tar.gz . Downloading  default   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/TIDL_TOOLS/$1/tidl_tools_gpu.tar.gz"
            wget --quiet  https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/TIDL_TOOLS/$1/tidl_tools_gpu.tar.gz
        else
            echo "Local file not found at $LOCAL_PATH/TIDL_TOOLS/$1/tidl_tools.tar.gz . Downloading  default   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/TIDL_TOOLS/$1/tidl_tools.tar.gz"
            wget --quiet  https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/TIDL_TOOLS/$1/tidl_tools.tar.gz
        fi
    fi
}

cp_osrt_lib()
{
    if [ -f $LOCAL_PATH/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1 ];then
        echo "Local file  found. Copying $LOCAL_PATH/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1"
        cp  $LOCAL_PATH/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1 .

    else
        echo "Local file not found at  $LOCAL_PATH/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1 . Downloading  default https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1"
        wget --quiet  https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/$1
    fi
}


SCRIPTDIR=`pwd`
REL=10_01_03_00
skip_cpp_deps=0
skip_arm_gcc_download=0
skip_x86_python_install=0
use_local=0
load_armnn=0
skip_model_optimizer=0


POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    --skip_cpp_deps)
    skip_cpp_deps=1
    ;;
    --skip_arm_gcc_download)
    skip_arm_gcc_download=1
    ;;
    --skip_x86_python_install)
    skip_x86_python_install=1
    ;;
    --use_local)
    use_local=1
    ;;
    --load_armnn)
    load_armnn=1
    ;;
    --skip_model_optimizer)
    skip_model_optimizer=1
    ;;
    -h|--help)
    echo Usage: $0 [options]
    echo
    echo Options,
    echo --skip_cpp_deps            skip Downloading or Compiling dependencies for CPP examples
    echo --skip_arm_gcc_download    skip Downloading or setting environment variable  for ARM64_GCC_PATH
    echo --skip_x86_python_install  skip installing of python packages
    echo --use_local                use OSRT packages and tidl_tools from localPath if present
    echo --skip_model_optimizer     skip installing model optimizer python package
    exit 0
    ;;
esac
shift # past argument
done
set -- "${POSITIONAL[@]}" # restore positional parameters

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


version_match=`python3 -c 'import sys;r=0 if sys.version_info >= (3,6) else 1;print(r)'`
if [ $version_match -ne 0 ]; then
    echo 'python version must be >= 3.6'
return
fi

arch=$(uname -m)
if [[ $arch == x86_64 ]]; then
    echo "X64 Architecture"
elif [[ $arch == aarch64 ]]; then
    echo "ARM Architecture"
    $skip_arm_gcc_download=1
else
    echo 'Processor Architecture must be x86_64 or aarch64'
    echo 'Processor Architecture "'$arch'" is Not Supported '
return
fi

if [[ $use_local == 1 ]];then
    if [ -z "$LOCAL_PATH" ];then
        echo "LOCAL_PATH not defined. set LOCAL_PATH to/your/path/08_XX_XX_XX/"
        return
    else
        echo "using OSRT from LOCAL_PATH:$LOCAL_PATH"
    fi

fi

if [ -z "$SOC" ];then
    echo "SOC not defined. Run either of below commands"
    echo "export SOC=am62"
    echo "export SOC=am62a"
    echo "export SOC=am68pa | j721e"
    echo "export SOC=am68a  | j721s2"
    echo "export SOC=am69a  | j784s4"
    echo "export SOC=am67a  | j722s"
    return
fi

case "$SOC" in
  am62|am62a|am68a|am68pa|am69a|am67a)
    ;;
  j721e)
    SOC=am68pa
    ;;
  j721s2)
    SOC=am68a
    ;; 
  j784s4)
    SOC=am69a
    ;;
  j722s)
    SOC=am67a
    ;;
  *)
    echo "Invalid SOC $SOC defined. Allowed values are"
    echo "export SOC=am62"
    echo "export SOC=am62a"
    echo "export SOC=am68pa | j721e"
    echo "export SOC=am68a  | j721s2"
    echo "export SOC=am69a  | j784s4"
    echo "export SOC=am67a  | j722s"
    return
    ;;
esac

echo "SOC=${SOC}"

# ######################################################################
# # Installing dependencies
if [[ $arch == x86_64 && $skip_x86_python_install -eq 0 ]]; then
    echo 'Installing python packages...'
    pip3 install pybind11[global]
    pip3 install -r ./requirements_pc.txt
fi
if [[ $arch == x86_64 ]]; then
    pip3 install pybind11[global]
    if [[ $use_local == 1 ]];then
        echo 'Installing python osrt packages from local...'
        pip_install_local dlr-1.13.0-py3-none-any.whl
        pip_install_local tvm-0.12.0-cp310-cp310-linux_x86_64.whl
        pip_install_local onnxruntime_tidl-1.15.0-cp310-cp310-linux_x86_64.whl
        pip_install_local tflite_runtime-2.12.0-cp310-cp310-linux_x86_64.whl
    else
        echo 'Installing python osrt packages...'
        pip3 install --quiet https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/dlr-1.13.0-py3-none-any.whl
        pip3 install --quiet https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tvm-0.12.0-cp310-cp310-linux_x86_64.whl
        pip3 install --quiet https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/onnxruntime_tidl-1.15.0-cp310-cp310-linux_x86_64.whl
        pip3 install --quiet https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tflite_runtime-2.12.0-cp310-cp310-linux_x86_64.whl
    fi
fi

# Create tools directory
mkdir -p $SCRIPTDIR/tools

if [ -z "$TIDL_TOOLS_PATH" ]; then

    mkdir -p $SCRIPTDIR/tools/${SOC^^}/
    cd tools/${SOC^^}/

    if [ -f tidl_tools.tar.gz ];then
        rm tidl_tools.tar.gz
    fi
    if [ -f tidl_tools_gpu.tar.gz ];then
        rm tidl_tools_gpu.tar.gz
    fi
    if [ -d tidl_tools ];then
        rm -r tidl_tools
    fi

    if [[ $use_local == 1 ]];then
        cp_tidl_tools ${SOC^^}
    else
        if [ $tidl_gpu_tools -eq 1 ];then
            echo "Downloading GPU TIDL TOOLS for ${SOC^^} ..."
            wget --quiet   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/TIDL_TOOLS/${SOC^^}/tidl_tools_gpu.tar.gz
        else
            echo "Downloading CPU TIDL TOOLS ${SOC^^} ..."
            wget --quiet   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/TIDL_TOOLS/${SOC^^}/tidl_tools.tar.gz
        fi
    fi

    # Untar tidl tools & remove the tar ball
    if [ $tidl_gpu_tools -eq 1 ];then
        tar -xzf tidl_tools_gpu.tar.gz
        if [ -f tidl_tools_gpu.tar.gz ];then
            rm tidl_tools_gpu.tar.gz
        fi
    else
        tar -xzf tidl_tools.tar.gz
        if [ -f tidl_tools.tar.gz ];then
            rm tidl_tools.tar.gz
        fi
    fi
    cd tidl_tools
    if [[ ! -L libvx_tidl_rt.so.1.0 && ! -f libvx_tidl_rt.so.1.0 ]];then
         ln -s  libvx_tidl_rt.so libvx_tidl_rt.so.1.0
    fi
    export TIDL_TOOLS_PATH=$(pwd)
    cd $SCRIPTDIR
else
    echo "TIDL_TOOLS_PATH already set to ${TIDL_TOOLS_PATH}. Skipping..."
fi

# graph optimizer tool setup
if [[ $arch == x86_64 && $skip_model_optimizer -eq 0 ]]; then
    cd $SCRIPTDIR/osrt-model-tools
    source ./setup.sh
    cd $SCRIPTDIR
fi

if [[ $arch == x86_64 && $skip_arm_gcc_download -eq 0 ]]; then
    cd $SCRIPTDIR/tools/
    if [ ! -d gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu ];then
        wget --quiet  https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
        tar -xf gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
        export ARM64_GCC_PATH=$(pwd)/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu
    else
        echo "skipping gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu download: found $(pwd)/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu"
        export ARM64_GCC_PATH=$(pwd)/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu
    fi
    cd $SCRIPTDIR
fi

if [[ $arch == x86_64 ]]; then
    cd $SCRIPTDIR/tools/
    if [ -f $CGT7X_ROOT/bin/cl7x ]; then
        echo "CGT7X_ROOT already set to $CGT7X_ROOT, skipping download"
    else
        if [ ! -d ti-cgt-c7000_3.1.0.LTS ];then
            wget --quiet https://dr-download.ti.com/software-development/ide-configuration-compiler-or-debugger/MD-707zYe3Rik/3.1.0.LTS/ti_cgt_c7000_3.1.0.LTS_linux-x64_installer.bin
            chmod +x ti_cgt_c7000_3.1.0.LTS_linux-x64_installer.bin
            ./ti_cgt_c7000_3.1.0.LTS_linux-x64_installer.bin --mode unattended --installdir $(pwd)
            export CGT7X_ROOT=$(pwd)/ti-cgt-c7000_3.1.0.LTS
        else
            echo "skipping ti-cgt-c7000_3.1.0.LTS download: found $(pwd)/ti-cgt-c7000_3.1.0.LTS"
            export CGT7X_ROOT=$(pwd)/ti-cgt-c7000_3.1.0.LTS
        fi
    fi
    cd $SCRIPTDIR
fi

if [ $skip_cpp_deps -eq 0 ]; then
    if [[ $arch == x86_64 ]]; then
        cd $SCRIPTDIR/tools/
        if [ -d osrt_deps ];then
            rm -r osrt_deps
        fi
        mkdir -p osrt_deps
        cd osrt_deps
        # onnxruntime
        echo "Installing:onnxruntime"
        if [ -f onnx_1.15.0_x86_u22.tar.gz ];then
            rm onnx_1.15.0_x86_u22.tar.gz
        fi
        if [[ $use_local == 1 ]];then
            cp_osrt_lib onnx_1.15.0_x86_u22.tar.gz
        else
            wget --quiet   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/onnx_1.15.0_x86_u22.tar.gz
        fi
        tar -xf onnx_1.15.0_x86_u22.tar.gz
        cd onnx_1.15.0_x86_u22
        if [ ! -f libonnxruntime.so ];then
            ln -s libonnxruntime.so.1.15.0 libonnxruntime.so
        fi
        if [ ! -f libonnxruntime.so.1.15.0 ];then
            ln -s libonnxruntime.so libonnxruntime.so.1.15.0
        fi
        cd ../
        rm onnx_1.15.0_x86_u22.tar.gz

        # tflite
        echo "Installing:tflite_2.12"
        if [ -f tflite_2.12_x86_u22.tar.gz ];then
            rm tflite_2.12_x86_u22.tar.gz
        fi
        if [[ $use_local == 1 ]];then
            cp_osrt_lib tflite_2.12_x86_u22.tar.gz
        else
            wget --quiet   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/tflite_2.12_x86_u22.tar.gz
        fi
        tar -xf tflite_2.12_x86_u22.tar.gz
        rm tflite_2.12_x86_u22.tar.gz   -r

        # opencv
        echo "Installing:opencv"
        if [ -f opencv_4.2.0_x86_u22.tar.gz ];then
            rm opencv_4.2.0_x86_u22.tar.gz
        fi
        if [[ $use_local == 1 ]];then
            cp_osrt_lib opencv_4.2.0_x86_u22.tar.gz
        else
            wget --quiet   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/opencv_4.2.0_x86_u22.tar.gz
        fi
        mkdir opencv_4.2.0_x86_u22 && tar xf opencv_4.2.0_x86_u22.tar.gz -C opencv_4.2.0_x86_u22 --strip-components 1
        rm opencv_4.2.0_x86_u22.tar.gz

        # dlr
        echo "Installing:dlr"
        if [ -f dlr_1.10.0_x86_u22.tar.gz ];then
            rm dlr_1.10.0_x86_u22.tar.gz
        fi
        if [[ $use_local == 1 ]];then
            cp_osrt_lib dlr_1.10.0_x86_u22.tar.gz
        else
            wget --quiet   https://software-dl.ti.com/jacinto7/esd/tidl-tools/$REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/dlr_1.10.0_x86_u22.tar.gz
        fi
        mkdir dlr_1.10.0_x86_u22 && tar xf dlr_1.10.0_x86_u22.tar.gz -C dlr_1.10.0_x86_u22 --strip-components 1
        rm dlr_1.10.0_x86_u22.tar.gz   -r

dlr_loc=$(python3  << EOF
import dlr
print(dlr.__file__)
EOF
)
        suffix="__init__.py"
        dlr_loc=${dlr_loc%"$suffix"}
        cp $dlr_loc/libdlr.so .

    fi

fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TIDL_TOOLS_PATH:$SCRIPTDIR/tools/osrt_deps:$SCRIPTDIR/tools/osrt_deps/opencv_4.2.0_x86_u22/opencv/

cd $TIDL_TOOLS_PATH
ln -s -r $SCRIPTDIR/tools/osrt_deps/ &> /dev/null
cd $SCRIPTDIR

echo "========================================================================="
echo "SOC=$SOC"
echo "TIDL_TOOLS_PATH=$TIDL_TOOLS_PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "CGT7X_ROOT=$CGT7X_ROOT"
echo "ARM64_GCC_PATH=$ARM64_GCC_PATH"
echo "========================================================================="
