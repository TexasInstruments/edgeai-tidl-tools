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


ARG REPO_LOCATION='arm64v8'
FROM ${REPO_LOCATION}ubuntu:20.04 as edgeai_tidl_tools_u20
ARG DEBIAN_FRONTEND=noninteractive
ARG NPROC=1
ARG PROJECT_NAME=edgeai_tidl_tools_u20
ENV USE_PROXY=ti

# Setup proxy settings
ADD entrypoint.sh setup_proxy.sh  /usr/bin/
RUN /usr/bin/setup_proxy.sh

RUN apt-get update -y && apt-get dist-upgrade -y && apt-get install -y \
    glib-2.0-dev \
    ncurses-dev \
    libopencv-core-dev \
    libopencv-imgproc-dev \
    libyaml-cpp-dev 

RUN apt-get install -y \
    build-essential \
    cmake \    
    git \
    wget \
    unzip \
    pkg-config \    
    openexr

# Install python and pip
RUN apt-get update -y && apt-get install -y \
    python3 \
    python3-dev \
    python3-opencv \
    python3-pip 
# Install miscellaneous packages
RUN apt-get install -y \
    vim \
    gdb \
    net-tools \
    chrony \
    corkscrew \
    libglib2.0-dev

RUN pip3 install Cython 
    
RUN pip3 install \
        jupyterlab \
        PyYAML \
        numpy \
        pillow 

#setup entrypoint to run the demo application
ENTRYPOINT ["/usr/bin/entrypoint.sh"]

