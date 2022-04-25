cmake_minimum_required(VERSION 3.0.2)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/common.cmake)
add_library(utils ${CMAKE_CURRENT_SOURCE_DIR}/src/utility_functs.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/include/utility_functs.h 
                            ${CMAKE_CURRENT_SOURCE_DIR}/src/model_info.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/include/model_info.h
                            ${CMAKE_CURRENT_SOURCE_DIR}/src/edgeai_classnames.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/include/edgeai_classnames.h
                            ${CMAKE_CURRENT_SOURCE_DIR}/src/ti_logger.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/include/ti_logger.h
                            ${CMAKE_CURRENT_SOURCE_DIR}/src/arg_parsing.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/include/arg_parsing.h
                            ${CMAKE_CURRENT_SOURCE_DIR}/src/pbPlots.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/include/pbPlots.hpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/src/supportLib.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/include/supportLib.hpp
                            )

install(TARGETS
         utils
         DESTINATION lib)

install(FILES 
${CMAKE_CURRENT_SOURCE_DIR}/include/utility_functs.h
${CMAKE_CURRENT_SOURCE_DIR}/include/model_info.h
${CMAKE_CURRENT_SOURCE_DIR}/include/edgeai_classnames.h
${CMAKE_CURRENT_SOURCE_DIR}/include/ti_logger.h
DESTINATION include)
