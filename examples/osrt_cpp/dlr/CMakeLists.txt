cmake_minimum_required(VERSION 3.0.2)

set(PROJ_NAME dlr_main)

project(${PROJ_NAME})

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/common.cmake)

set(DLR_CLS_APP_SRCS
    dlr_main.cpp)

build_app(${PROJ_NAME} DLR_CLS_APP_SRCS)
