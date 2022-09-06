#!/bin/bash

#  Copyright (C) 2021 Texas Instruments Incorporated - http://www.ti.com/
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#    Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
#    Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the
#    distribution.
#
#    Neither the name of Texas Instruments Incorporated nor the names of
#    its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# ./tensorflow_src/tensorflow/tensorflow/lite/tools/pip_package/gen/tflite_pip/python3.6/dist/tflite_runtime-2.8.2-cp36-cp36m-linux_aarch64.whl
# ./tensorflow_src/tensorflow/tensorflow/lite/tools/pip_package/gen/tflite_pip/python3/dist/tflite_runtime-2.8.2-cp36-cp36m-linux_x86_64.whl
# ./tensorflow_src/tensorflow/tensorflow/lite/tools/pip_package/gen/tflite_pip/python3.8/dist/tflite_runtime-2.8.2-cp38-cp38-linux_aarch64.whl

ping bitbucket.itg.ti.com -c 1 > /dev/null 2>&1
if [ "$?" -eq "0" ]; then
    USE_PROXY=1
else
    USE_PROXY=0
fi
#To dwld the src files
./tflite_2.8_prepare.sh

if [ $# -lt 1 ];then
    echo "usage ./build_tflite_2.8.sh ubuntu18"
    exit
else
    echo "cross compiling tflite_2.8 for arago linux and Ubuntu 18.04 an Ubuntu 20.04 docker containers"
fi

DOCKERTAG=$1

docker run -it --rm \
    -v $(pwd)/:/root/dlrt-build \
    -v /:/host \
    --network host \
    --env USE_PROXY=$USE_PROXY \
    $DOCKERTAG \
    /bin/bash -c "~/dlrt-build/tflite_2.8_build.sh"

# keeping the py whl cretion in docker host coz requires docker
if [ $USE_PROXY -eq "1" ];then
    echo "inside ti network:using proxy"
    cd tflite_2.8/tensorflow_src/tensorflow/
    git checkout tensorflow/lite/tools/pip_package/Dockerfile.py3
    cat << EOF >  add.txt
ENV http_proxy http://webproxy.ext.ti.com:80
ENV https_proxy http://webproxy.ext.ti.com:80
EOF
    sed -i '/FROM ${IMAGE}/r add.txt' tensorflow/lite/tools/pip_package/Dockerfile.py3
    rm add.txt
    make -C tensorflow/lite/tools/pip_package docker-build   TENSORFLOW_TARGET=aarch64 PYTHON_VERSION=3.8 BASE_IMAGE=artifactory.itg.ti.com/docker-public/library/ubuntu:18.04
    sed -i 's/RUN python3 get-pip.py/RUN apt-get install -y python3-pip python-dev/g' tensorflow/lite/tools/pip_package/Dockerfile.py3
    sed -i 's/RUN rm get-pip.py//g ' tensorflow/lite/tools/pip_package/Dockerfile.py3
    make -C tensorflow/lite/tools/pip_package docker-build   TENSORFLOW_TARGET=aarch64 PYTHON_VERSION=3.6 BASE_IMAGE=artifactory.itg.ti.com/docker-public/library/ubuntu:18.04
    
else
    cd tflite_2.8/tensorflow_src/tensorflow/
    make -C tensorflow/lite/tools/pip_package docker-build   TENSORFLOW_TARGET=aarch64 PYTHON_VERSION=3.8
    sed -i 's/RUN python3 get-pip.py/RUN apt-get install -y python3-pip python-dev/g' tensorflow/lite/tools/pip_package/Dockerfile.py3
    sed -i 's/RUN rm get-pip.py//g ' tensorflow/lite/tools/pip_package/Dockerfile.py3 
    make -C tensorflow/lite/tools/pip_package docker-build   TENSORFLOW_TARGET=aarch64 PYTHON_VERSION=3.6
    
fi
