cmake_minimum_required(VERSION 3.0.2)

set(PROJ_NAME ort_clasification)

project(${PROJ_NAME})

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/common.cmake)

set(ORT_CLS_APP_SRCS
    validator.cpp
    onnx_main.cpp)

build_app(${PROJ_NAME} ORT_CLS_APP_SRCS)
