cmake_minimum_required(VERSION 3.10.0)

project(tidl_wrapper)

include(${CMAKE_SOURCE_DIR}/../../cmake/common.cmake)

include_directories(${CMAKE_SOURCE_DIR}
                    ${CMAKE_SOURCE_DIR}/common
                    ${CMAKE_SOURCE_DIR}/onnxrt
                    ${CMAKE_SOURCE_DIR}/tflitert
                    ${CMAKE_SOURCE_DIR}/tidlrt
                    ${CMAKE_SOURCE_DIR}/tvmrt)

set(WRAPPER_SRCS
    ${WRAPPER_SRCS}
    onnxrt/onnxrt_wrapper.cpp
    tflitert/tflitert_wrapper.cpp
    tidlrt/tidlrt_wrapper.cpp
    tvmrt/tvmrt_wrapper.cpp)

build_lib(${PROJECT_NAME}   # Named argument: library name
          STATIC            # Named argument: library type
          1.0.0             # Named argument: library version
          ${WRAPPER_SRCS})