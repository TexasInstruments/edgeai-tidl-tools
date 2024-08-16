cmake_minimum_required(VERSION 3.0.2)

set(PROJ_NAME tidlrt_priority_scheduling)

project(${PROJ_NAME})

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/common.cmake)

set(TIDL_RT_APP_SRCS
    tidlrt_priority_scheduling.cpp)

build_app(${PROJ_NAME} TIDL_RT_APP_SRCS)
