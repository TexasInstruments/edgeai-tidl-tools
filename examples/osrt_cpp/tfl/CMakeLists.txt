cmake_minimum_required(VERSION 3.0.2)

set(PROJ_NAME tfl_main)

project(${PROJ_NAME})
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/common.cmake)


set(TFL_APP_SRCS
    tfl_main.cpp  tfl_main.h) 

add_subdirectory(post_process)


build_app(${PROJ_NAME} TFL_APP_SRCS)


