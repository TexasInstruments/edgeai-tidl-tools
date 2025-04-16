cmake_minimum_required(VERSION 3.0.2)

set(PROJ_NAME tvm_main)

project(${PROJ_NAME})

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/common.cmake)

set(TVM_APP_SRCS
    tvm_main.cpp)

build_app(${PROJ_NAME} TVM_APP_SRCS)
