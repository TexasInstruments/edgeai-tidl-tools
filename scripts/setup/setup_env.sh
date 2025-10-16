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

CURRDIR=`pwd`
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TOOLSDIR=`realpath ${SCRIPTDIR}/../../tools`

SOC=$1

if [ -z "$SOC" ];then
    echo "[ERROR] SOC not defined"
    return
fi

SOC=${SOC^^}
case "$SOC" in
  AM62|AM62A|J721S2|J721E|J784S4|J722S)
    ;;
  AM68PA|TDA4VM)
    SOC=J721E
    ;;
  AM68A|TDA4VL)
    SOC=J721S2
    ;; 
  AM69A|TDA4VH)
    SOC=J784S4
    ;;
  AM67A|TDA4AEN)
    SOC=J722S
    ;;
  *)
    echo "[ERROR] Invalid SOC $SOC defined"
    echo "AM62, AM62A, (J721E or TDA4VM), (J721S2 or TDA4VL or AM68A), (J784S4 or TDA4VH or AM69A) and (J722S or TDA4AEN or AM67A)"
    return
    ;;
esac

export SOC=$SOC
export TIDL_TOOLS_PATH=$TOOLSDIR/$SOC/tidl_tools
if [ -z "$( ls -A $TIDL_TOOLS_PATH )" ]; then
   echo "[ERROR] $TIDL_TOOLS_PATH does not exist or is empty. Please run the setup.sh"
   return
fi

# Remove any paths ending with tidl_tools or osrt_deps from LD_LIBRARY_PATH to avoid appending duplicate
if [ ! -z "$LD_LIBRARY_PATH" ]; then
    NEW_LD_LIBRARY_PATH=""
    IFS=':' read -ra PATHS <<< "$LD_LIBRARY_PATH"
    for path in "${PATHS[@]}"; do
        if [[ "$path" != *"tidl_tools" && "$path" != *"osrt_deps" ]]; then
            if [ -z "$NEW_LD_LIBRARY_PATH" ]; then
                NEW_LD_LIBRARY_PATH="$path"
            else
                NEW_LD_LIBRARY_PATH="$NEW_LD_LIBRARY_PATH:$path"
            fi
        fi
    done
    LD_LIBRARY_PATH="$NEW_LD_LIBRARY_PATH"
fi

# Add the paths to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TIDL_TOOLS_PATH:$TOOLSDIR/osrt_deps

if [ -f $TOOLSDIR/ti-cgt-c7000_5.0.0.LTS/bin/cl7x ]; then
    export CGT7X_ROOT=$TOOLSDIR/ti-cgt-c7000_5.0.0.LTS
else
    echo "[WARN] ti-cgt-c7000_5.0.0.LTS does not contain the cl7x executable. Please run the setup.sh to install"
fi

if [ -d $TOOLSDIR/arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu ];then
    export ARM64_GCC_PATH=$TOOLSDIR/arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu
else
    echo "[WARN] ARM GNU toolchain not found. Please run the setup.sh to install"
fi  

echo "========================================================================="
echo "SOC=$SOC"
echo "TIDL_TOOLS_PATH=$TIDL_TOOLS_PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "CGT7X_ROOT=$CGT7X_ROOT"
echo "ARM64_GCC_PATH=$ARM64_GCC_PATH"
echo "========================================================================="
