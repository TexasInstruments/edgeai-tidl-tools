cmake_minimum_required(VERSION 3.0.2)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/common.cmake)

set(DL_INFERER_SRCS
    src/ti_dl_inferer.cpp
    # src/ti_dlr_inferer.cpp
    src/ti_onnx_inferer.cpp
    src/ti_tflite_inferer.cpp)

add_library(ti_dl_inferer 
        ${DL_INFERER_SRCS} )

install(TARGETS
    ti_dl_inferer
    DESTINATION lib)

install(FILES 
${CMAKE_CURRENT_SOURCE_DIR}/include/ti_dl_inferer.h
# ${CMAKE_CURRENT_SOURCE_DIR}/include/ti_dlr_inferer.h
${CMAKE_CURRENT_SOURCE_DIR}/include/ti_onnx_inferer.h
${CMAKE_CURRENT_SOURCE_DIR}/include/ti_tflite_inferer.h
DESTINATION include)
