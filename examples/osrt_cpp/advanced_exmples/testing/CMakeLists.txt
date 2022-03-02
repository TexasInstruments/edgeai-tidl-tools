cmake_minimum_required(VERSION 3.0.2)

set(PROJ_NAME print_debugs)

project(${PROJ_NAME})
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/common.cmake)


set(DBG_APP_SRCS
dbg_main.cpp   ) 



build_app(${PROJ_NAME} DBG_APP_SRCS)



