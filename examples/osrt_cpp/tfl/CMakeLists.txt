cmake_minimum_required(VERSION 3.0.2)

set(PROJ_NAME tfl_clasification)
set(PROJ_NAME_OD tfl_od)
set(PROJ_NAME_SEG tfl_seg)

project(${PROJ_NAME})
project(${PROJ_NAME_OD})
project(${PROJ_NAME_SEG})

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/common.cmake)

set(TFL_CLS_APP_SRCS
    classification.cpp ) 

set(TFL_OD_APP_SRCS
    od.cpp ) 

set(TFL_SEG_APP_SRCS
    segmentation.cpp ) 

build_app(${PROJ_NAME} TFL_CLS_APP_SRCS)
build_app(${PROJ_NAME_OD} TFL_OD_APP_SRCS)
build_app(${PROJ_NAME_SEG} TFL_SEG_APP_SRCS)
