cmake_minimum_required(VERSION 3.0.2)

set(PROJ_NAME ort_main)

project(${PROJ_NAME})

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/common.cmake)

set(ORT_APP_SRCS
    onnx_main.cpp onnx_main.h)

build_app(${PROJ_NAME} ORT_APP_SRCS)
