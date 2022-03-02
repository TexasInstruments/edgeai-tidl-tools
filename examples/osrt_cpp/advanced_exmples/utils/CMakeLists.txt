cmake_minimum_required(VERSION 3.0.2)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/common.cmake)
add_library(utils_adv ${CMAKE_CURRENT_SOURCE_DIR}/src/arg_parsing.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/include/arg_parsing.h
                            ${CMAKE_CURRENT_SOURCE_DIR}/src/print_utils.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/include/print_utils.h
                            )

install(TARGETS
         utils_adv
         DESTINATION lib)

install(FILES 
${CMAKE_CURRENT_SOURCE_DIR}/include/arg_parsing.h
${CMAKE_CURRENT_SOURCE_DIR}/include/print_utils.h

DESTINATION include)
