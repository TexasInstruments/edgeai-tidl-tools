cmake_minimum_required(VERSION 3.0.2)

set(PROJ_NAME tfl_clasification)
set(PROJ_NAME_OD tfl_od)

project(${PROJ_NAME})
project(${PROJ_NAME_OD})

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/common.cmake)

set(TFL_CLS_APP_SRCS
    classification.cpp ) 

set(TFL_OD_APP_SRCS
    od.cpp ) 

build_app(${PROJ_NAME} TFL_CLS_APP_SRCS)
build_app(${PROJ_NAME_OD} TFL_OD_APP_SRCS)
