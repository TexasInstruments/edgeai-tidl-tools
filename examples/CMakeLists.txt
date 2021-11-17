cmake_minimum_required(VERSION 3.0.2)

project(edgeai_tidl_examples)

exec_program(pkg-config ARGS --cflags glib-2.0 OUTPUT_VARIABLE GLIB_FLAGS)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GLIB_FLAGS}")

add_subdirectory(osrt_cpp/post_process)
add_subdirectory(osrt_cpp/utils)
add_subdirectory(osrt_cpp/tfl)
# add_subdirectory(osrt_cpp/ort)
# add_subdirectory(osrt_cpp/dlr)
# add_subdirectory(tidlrt_cpp)
