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


if [ -z "$REPO_LOCATION" ];then
    echo "No REPO_LOCATION specified, using default"
else
    echo "Using REPO_LOCATION: $REPO_LOCATION"
fi

if [ -z "$PROXY" ];then
    echo "No PROXY specified"
    PROXY=none
else
    echo "Using PROXY: $PROXY"
fi

if [ $tidl_gpu_tools -eq 1 ];then
    sudo docker build --build-arg REPO_LOCATION=$REPO_LOCATION --build-arg PROXY=$PROXY  -f $script_dir/Dockerfile_GPU -t edgeai_tidl_tools_x86_ubuntu_22_gpu .

else
    sudo docker build --build-arg REPO_LOCATION=$REPO_LOCATION --build-arg PROXY=$PROXY  -f $script_dir/Dockerfile -t edgeai_tidl_tools_x86_ubuntu_22 .
fi