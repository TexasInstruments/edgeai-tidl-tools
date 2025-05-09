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

SCRIPTDIR=`pwd`
SOC=${1,,}

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

export SOC=$SOC
export TIDL_TOOLS_PATH=$SCRIPTDIR/tools/${SOC^^}/tidl_tools
if [ -z "$( ls -A $TIDL_TOOLS_PATH )" ]; then
   echo "[ERROR] $TIDL_TOOLS_PATH does not exist or is empty. Please run the setup.sh"
   return
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TIDL_TOOLS_PATH:$SCRIPTDIR/tools/osrt_deps:$SCRIPTDIR/tools/osrt_deps/opencv_4.2.0_x86_u22/opencv/

arch=$(uname -m)
if [[ $arch == x86_64 ]]; then
    if [ -d $SCRIPTDIR/tools/ti-cgt-c7000_5.0.0.LTS ];then
        export CGT7X_ROOT=$SCRIPTDIR/tools/ti-cgt-c7000_5.0.0.LTS
    fi
fi
if [[ $arch == x86_64 ]]; then
    if [ -d $SCRIPTDIR/tools/arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu ];then
        export ARM64_GCC_PATH=$SCRIPTDIR/tools/arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu
    fi
fi

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
