cmake_minimum_required(VERSION 3.0.2)

set(PROJ_NAME tidlrt_clasification)

project(${PROJ_NAME})

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/common.cmake)

set(TIDL_RT_APP_SRCS
    classification.cpp)

build_app(${PROJ_NAME} TIDL_RT_APP_SRCS)
