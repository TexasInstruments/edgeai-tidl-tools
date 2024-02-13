ARG REPO_LOCATION=''
FROM ${REPO_LOCATION}ubuntu:22.04
ARG USE_PROXY=none
RUN bash -c 'if [ $USE_PROXY = "ti" ]; then echo "Acquire::http::proxy \"http://webproxy.ext.ti.com:80\";" > /etc/apt/apt.conf; fi'
 
# baseline
RUN apt-get update
RUN apt-get install -y python3 python3-pip python3-setuptools

# ubuntu package dependencies
# libsm6 libxext6 libxrender1 : needed by opencv
# cmake protobuf-compiler libprotoc-dev : needed by onnx
# graphviz : needed by tvm
# swig : needed by model selection tool
# curl vim git wget gdb : needeed by baseline dev
RUN bash -c 'if [ $USE_PROXY = "ti" ];then \
                export ftp_proxy=http://webproxy.ext.ti.com:80;\
                export http_proxy=http://webproxy.ext.ti.com:80;\
                export https_proxy=http://webproxy.ext.ti.com:80;\
                export no_proxy=ti.com;\
                apt-get update;\
                apt-get install -y cmake libprotobuf-dev protobuf-compiler libprotoc-dev graphviz swig curl vim git wget gdb nano zip pkg-config libgtk-3-dev libyaml-cpp-dev;\
            else \
                apt-get update;\
                apt-get install -y cmake libprotobuf-dev protobuf-compiler libprotoc-dev graphviz swig curl vim git wget gdb nano zip pkg-config libgtk-3-dev libyaml-cpp-dev;\
            fi'
COPY requirements_pc.txt /requirements_pc.txt
RUN bash -c 'if [ $USE_PROXY = "ti" ];then \
                export ftp_proxy=http://webproxy.ext.ti.com:80;\
                export http_proxy=http://webproxy.ext.ti.com:80;\
                export https_proxy=http://webproxy.ext.ti.com:80;\
                export HTTP_PROXY=http://webproxy.ext.ti.com:80\
                export HTTPS_PROXY=http://webproxy.ext.ti.com:80\
                export no_proxy=ti.com;\
                pip3 install pybind11[global];\
                pip3 install -r /requirements_pc.txt;\
            else \
                pip3 install pybind11[global];\
                pip3 install -r /requirements_pc.txt;\
            fi'

RUN bash -c 'if [ $USE_PROXY = "ti" ];then echo -e "export ftp_proxy=http://webproxy.ext.ti.com:80\nexport http_proxy=http://webproxy.ext.ti.com:80\nexport https_proxy=http://webproxy.ext.ti.com:80\nexport no_proxy=ti.com " > ~/.bashrc;fi'
# Code file to execute when the docker container starts up (`entrypoint.sh`)
# ENTRYPOINT ["/root/edgeai-tidl-tools/entrypoint.sh"]
