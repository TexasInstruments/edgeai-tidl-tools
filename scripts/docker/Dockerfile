ARG REPO_LOCATION=''
FROM ${REPO_LOCATION}ubuntu:22.04
ARG PROXY=none
RUN bash -c 'if [ ${PROXY} != none ]; then echo "Acquire::http::proxy \"${PROXY}\";" > /etc/apt/apt.conf; fi'

# baseline
RUN apt-get update
RUN apt-get install -y python3 python3-pip python3-setuptools

# ubuntu package dependencies
# libsm6 libxext6 libxrender1 : needed by opencv
# cmake protobuf-compiler libprotoc-dev : needed by onnx
# graphviz : needed by tvm
# swig : needed by model selection tool
# curl vim git wget gdb : needeed by baseline dev
RUN bash -c 'if [ ${PROXY} != none ];then \
                export ftp_proxy=${PROXY};\
                export http_proxy=${PROXY};\
                export https_proxy=${PROXY};\
                export HTTP_PROXY=${PROXY}\
                export HTTPS_PROXY=${PROXY}\
                export no_proxy=ti.com;\
                apt-get update;\
                apt-get install -y cmake libprotobuf-dev protobuf-compiler libprotoc-dev graphviz swig curl vim git wget gdb nano zip pkg-config libgtk-3-dev libyaml-cpp-dev;\
            else \
                apt-get update;\
                apt-get install -y cmake libprotobuf-dev protobuf-compiler libprotoc-dev graphviz swig curl vim git wget gdb nano zip pkg-config libgtk-3-dev libyaml-cpp-dev;\
            fi'
COPY requirements_pc.txt /requirements_pc.txt
RUN bash -c 'if [ ${PROXY} != none ];then \
                export ftp_proxy=${PROXY};\
                export http_proxy=${PROXY};\
                export https_proxy=${PROXY};\
                export HTTP_PROXY=${PROXY}\
                export HTTPS_PROXY=${PROXY}\
                export no_proxy=ti.com;\
                pip3 install pybind11[global];\
                pip3 install -r /requirements_pc.txt;\
            else \
                pip3 install pybind11[global];\
                pip3 install -r /requirements_pc.txt;\
            fi'

RUN bash -c 'if [ ${PROXY} != none ];then echo -e "export ftp_proxy=${PROXY}\nexport http_proxy=${PROXY}\nexport https_proxy=${PROXY} " > ~/.bashrc;fi'