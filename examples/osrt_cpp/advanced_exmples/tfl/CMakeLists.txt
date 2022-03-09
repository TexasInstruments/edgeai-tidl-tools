cmake_minimum_required(VERSION 3.0.2)

set(PROJ_NAME tfl_main_adv)

project(${PROJ_NAME})
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/common.cmake)


set(TFL_APP_SRCS
tfl_main.cpp  tfl_main.h ) 
build_app(${PROJ_NAME} TFL_APP_SRCS)

set(PROJ_NAME tfl_main_adv_fps)
set(TFL_APP_SRCS
tfl_main_fps_calc.cpp  tfl_main.h ) 
build_app(${PROJ_NAME} TFL_APP_SRCS)

set(PROJ_NAME tfl_main_adv_op_val)
set(TFL_APP_SRCS
tfl_main_op_tensor_val.cpp  tfl_main.h ) 
build_app(${PROJ_NAME} TFL_APP_SRCS)




