cmake_minimum_required(VERSION 3.0.2)
add_library(tfl_pre_process tfl_pre_process.cpp tfl_pre_process.h )
install(TARGETS tfl_pre_process DESTINATION lib)
install(FILES tfl_pre_process.h DESTINATION include)