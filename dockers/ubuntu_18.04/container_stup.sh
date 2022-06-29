# echo source /mnt/work/edge_ai_apps/docker/container_stup.sh >> ~/.bashrc
# echo cd /mnt/work/github/edgeai-tidl-tools/ >> ~/.bashrc

    
cd $HOME
if [  ! -d tensorflow ];then
    cp -r /mnt/tensorflow tensorflow 
fi
if [  ! -d onnxruntime ];then
    cp -r /mnt/onnxruntime onnxruntime
fi
if [  ! -d neo-ai-dlr ];then
    cp -r /mnt/neo-ai-dlr neo-ai-dlr
fi

if [  ! -d includes ];then
    mkdir includes
    cp /mnt/work/github/tidl_tools/itidl_rt.h includes/
    export CPLUS_INCLUDE_PATH=~/includes
    export C_INCLUDE_PATH=~/includes
fi
export CPLUS_INCLUDE_PATH=~/includes
export C_INCLUDE_PATH=~/includes


cp -r /mnt/work/psdkra_new/targetfs/usr/include/opencv4/* /usr/include/opencv4/
cp -r /mnt/work/psdkra_new/targetfs/usr/include/opencv4/* /usr/include/opencv4/

if [  ! -d required_lib_18 ];then
    cp -r /mnt/temp/container/required_lib_18  required_lib_18
    cd ~/required_lib_18 
    cp -r /mnt/work/qemu/dlrt-build/opencv/opencv-4.2.0/build/lib opencv
fi
cd ~/required_lib_18 
export LD_LIBRARY_PATH=$(pwd)
cd /usr/lib/aarch64-linux-gnu/
if [  ! -L libwebp.so ];then
    ln -s libwebp.so.6 libwebp.so
fi
if [  ! -L libjpeg.so ];then
    ln -s libjpeg.so.8 libjpeg.so
fi
if [  ! -L libpng16.so ];then
    ln -s libpng16.so.16 libpng16.so
fi
if [  ! -L libtiff.so ];then
    ln -s libtiff.so.5 libtiff.so
fi
cd /mnt/work/github/edgeai-tidl-tools/
export TIDL_TOOLS_PATH=
export DEVICE=j7
pip3 install \
    cython \
    PyYAML  \
    Dlr 
pip3 install /mnt/onnx_artifacts/arm/tfl_3.6/tflite_runtime-2.4.0-py3-none-linux_aarch64.whl
apt install -y libopencv-core-dev     libopencv-imgproc-dev  python3-opencv
if [  ! -d /usr/dlr ];then
    mkdir /usr/dlr
    cp ~/required_lib_18/libdlr.so /usr/dlr/
fi


