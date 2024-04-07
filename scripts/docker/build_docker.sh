#!/bin/bash

# Copyright (c) 2018-2023, Texas Instruments
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
script_dir=$(dirname -- ${BASH_SOURCE[0]})

#Check if CPU or GPU tools
if [ -z "$TIDL_TOOLS_TYPE" ];then
    echo "TIDL_TOOLS_TYPE unset, defaulting to CPU tools"
    tidl_gpu_tools=0
else
    if [ $TIDL_TOOLS_TYPE == GPU ];then
        tidl_gpu_tools=1
    else
        tidl_gpu_tools=0
    fi
fi

#Check for TI Proxy:
use_ti_proxy=0
if [ -z "$USE_PROXY" ];then
    echo "TIDL_TOOLS_TYPE unset, defaulting to CPU tools"
    use_ti_proxy=0
else
    if [ $USE_PROXY == ti ];then
        echo "Using TI Proxy"
        use_ti_proxy=1
    else
        use_ti_proxy=0
    fi
fi

if [ $tidl_gpu_tools -eq 1 ];then
    sudo docker build -f $script_dir/Dockerfile_GPU -t x86_ubuntu_22 .
else
    if [ $use_ti_proxy -eq 1 ];then
        sudo docker build --build-arg REPO_LOCATION=artifactory.itg.ti.com/docker-public/library/ --build-arg USE_PROXY=ti  -f $script_dir/Dockerfile -t x86_ubuntu_22 .
    else
        sudo docker build -f $script_dir/Dockerfile -t x86_ubuntu_22 .
    fi
fi