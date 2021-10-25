cmake_minimum_required(VERSION 3.0.2)
add_library(tfl_post_process tfl_post_process.cpp tfl_post_process.h )
install(TARGETS tfl_post_process DESTINATION lib)
install(FILES tfl_post_process.h DESTINATION include)