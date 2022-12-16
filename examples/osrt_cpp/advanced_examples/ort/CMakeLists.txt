cmake_minimum_required(VERSION 3.0.2)

set(PROJ_NAME ort_priority_scheduling)

project(${PROJ_NAME})

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/common.cmake)

set(ORT_APP_SRCS
ort_priority_scheduling.cpp ort_priority_scheduling.h)

build_app(${PROJ_NAME} ORT_APP_SRCS)
