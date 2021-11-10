cmake_minimum_required(VERSION 3.0.2)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/common.cmake)
add_library(arg_parsing ${CMAKE_CURRENT_SOURCE_DIR}/src/arg_parsing.cpp ${CMAKE_CURRENT_SOURCE_DIR}/include/arg_parsing.h )
add_library(utility_functs ${CMAKE_CURRENT_SOURCE_DIR}/src/utility_functs.cpp ${CMAKE_CURRENT_SOURCE_DIR}/include/utility_functs.h )
add_library(model_info ${CMAKE_CURRENT_SOURCE_DIR}/src/model_info.cpp ${CMAKE_CURRENT_SOURCE_DIR}/include/model_info.h )
add_library(edgeai_classnames ${CMAKE_CURRENT_SOURCE_DIR}/src/edgeai_classnames.cpp ${CMAKE_CURRENT_SOURCE_DIR}/include/edgeai_classnames.h )

install(TARGETS
         arg_parsing
         utility_functs
         model_info
         edgeai_classnames
         DESTINATION lib)

install(FILES 
${CMAKE_CURRENT_SOURCE_DIR}/include/arg_parsing.h
${CMAKE_CURRENT_SOURCE_DIR}/include/utility_functs.h
${CMAKE_CURRENT_SOURCE_DIR}/include/model_info.h
${CMAKE_CURRENT_SOURCE_DIR}/include/edgeai_classnames.h
DESTINATION include)
