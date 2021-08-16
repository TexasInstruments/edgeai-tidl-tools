cmake_minimum_required(VERSION 3.0.2)

set(PROJ_NAME tfl_clasification)

project(${PROJ_NAME})

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/common.cmake)

set(TFL_CLS_APP_SRCS
    classification.cpp)

build_app(${PROJ_NAME} TFL_CLS_APP_SRCS)
