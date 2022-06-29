#! /bin/bash
# This script should be run inside the CONTAINER
# Outputs:
# - # - opencv/opencv-4.2.0/build/lib/*

# update how many CPUs to use
apt update -y
apt-get  install -y  libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev
apt-get install -y   libjpeg-dev libtiff5-dev libpng-dev
apt-get install -y  build-essential cmake
apt-get install -y  libv4l-dev v4l-utils
apt-get install -y  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev 
apt-get install -y  libgtk2.0-dev
apt-get install -y  mesa-utils libgl1-mesa-dri libgtkgl2.0-dev libgtkglext1-dev  
apt-get install -y  libatlas-base-dev gfortran libeigen3-dev
apt-get install -y  python2.7-dev python3-dev python-numpy python3-numpy

cd opencv/opencv-4.2.0/build/
mkdir /root/dlrt-build/opencv/opencv-4.2.0/bin
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/root/dlrt-build/opencv/opencv-4.2.0/bin -D WITH_TBB=OFF -D WITH_IPP=OFF -D WITH_1394=OFF -D BUILD_WITH_DEBUG_INFO=OFF -D BUILD_DOCS=OFF -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D BUILD_EXAMPLES=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D WITH_QT=OFF -D WITH_GTK=ON -D WITH_OPENGL=ON  -D WITH_V4L=ON  -D WITH_FFMPEG=ON -D WITH_XINE=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D OPENCV_GENERATE_PKGCONFIG=ON ../
make 



