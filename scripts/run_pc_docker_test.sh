#!/bin/bash -e
# Copyright (c) {2015 - 2021} Texas Instruments Incorporated
#
# All rights reserved not granted herein.
#
# Limited License.
#
# Texas Instruments Incorporated grants a world-wide, royalty-free, non-exclusive
# license under copyrights and patents it now or hereafter owns or controls to make,
# have made, use, import, offer to sell and sell ("Utilize") this software subject to the
# terms herein.  With respect to the foregoing patent license, such license is granted
# solely to the extent that any such patent is necessary to Utilize the software alone.
# The patent license shall not apply to any combinations which include this software,
# other than combinations with devices manufactured by or for TI ("TI Devices").
# No hardware patent is licensed hereunder.
#
# Redistributions must preserve existing copyright notices and reproduce this license
# (including the above copyright notice and the disclaimer and (if applicable) source
# code license limitations below) in the documentation and/or other materials provided
# with the distribution
#
# Redistribution and use in binary form, without modification, are permitted provided
# that the following conditions are met:
#
# *       No reverse engineering, decompilation, or disassembly of this software is
# permitted with respect to any software provided in binary form.
#
# *       any redistribution and use are licensed by TI for use only with TI Devices.
#
# *       Nothing shall obligate TI to provide you with source code for the software
# licensed and provided to you in object code.
#
# If software source code is provided to you, modification and redistribution of the
# source code are permitted provided that the following conditions are met:
#
# *       any redistribution and use of the source code, including any resulting derivative
# works, are licensed by TI for use only with TI Devices.
#
# *       any redistribution and use of any object code compiled from the source code
# and any resulting derivative works, are licensed by TI for use only with TI Devices.
#
# Neither the name of Texas Instruments Incorporated nor the names of its suppliers
#
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# DISCLAIMER.
#
# THIS SOFTWARE IS PROVIDED BY TI AND TI'S LICENSORS "AS IS" AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL TI AND TI'S LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

use_local=0
am62a=0
am68a=0
am68pa=0
am69a=0
am62=0

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    --use_local)
    use_local=1
    ;;  
    --am62a)
    am62a=1
    ;;    
    --am62)
    am62=1
    ;; 
    --am68a)
    am68a=1
    ;; 
    --am69a)
    am69a=1
    ;; 
    --am68pa)
    am68pa=1
    ;;                 
    -h|--help)
    echo Usage: $0 [device] [options]
    echo
    echo Options,
    echo --use_local            use OSRT packages and tidl_tools from localPath if present
    echo --am62a           test for am62a
    exit 0
    ;;
esac
shift # past argument
done
set -- "${POSITIONAL[@]}" # restore positional parameters

if [[ $use_local == 1 ]];then
    if [ -z "$LOCAL_PATH" ];then
        echo "LOCAL_PATH not defined. set LOCAL_PATH to/your/path/08_XX_XX_XX/"        
        return 0
    else
        echo "using OSRT from LOCAL_PATH:$LOCAL_PATH"
    fi
    
fi

DOCKER_TAG=ubuntu18-test
if [[ "$(docker images -q $DOCKER_TAG 2> /dev/null)" == "" ]]; then
  echo "building docker image"
  sudo docker build --build-arg REPO_LOCATION=artifactory.itg.ti.com/docker-public/library/ --build-arg USE_PROXY=ti -t $DOCKER_TAG  -f Dockerfile .
else
    echo "Using existing docker image"
fi
run_test()
{
  DOCKER_TAG=ubuntu18-test
  if [[ $use_local == 1 ]];then
      echo "Running test for $1 with --use_local"
      docker run --rm  -it -v $(pwd)/:/root/edgeai-tidl-tools -v /:/host --env SOC=$1 --env LOCAL_PATH=/host/$LOCAL_PATH --network host --name $DOCKER_TAG  --shm-size=6gb  $DOCKER_TAG  /bin/bash
    else
      echo "Running test for $1"
      docker run --rm  -it -v $(pwd)/:/root/edgeai-tidl-tools -v /:/host --env SOC=$1  --network host --name $DOCKER_TAG  --shm-size=6gb  $DOCKER_TAG  /bin/bash
    fi
}
if [ $am62a == 1 ];then    
  run_test am62a
fi
if [ $am62 == 1 ];then
  run_test am62
fi
if [ $am69a == 1 ];then
  run_test am69a
fi 
if [ $am68a == 1 ];then
  run_test am68a
fi
if [ $am68pa == 1 ];then
  run_test am68pa
fi

