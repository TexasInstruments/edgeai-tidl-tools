include(GNUInstallDirs)

add_compile_options(-std=c++17 -fPIC)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE} PROJECT_NAME = ${PROJECT_NAME}")

SET(CMAKE_FIND_LIBRARY_PREFIXES "" "lib")
SET(CMAKE_FIND_LIBRARY_SUFFIXES ".a" ".lib" ".so")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/${CMAKE_BUILD_TYPE})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})

if(CMAKE_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*")
  set(HOST_CPU x86)
  if(NOT DEFINED ENV{TARGET_CPU})
    set(TARGET_CPU x86)
  else()
    set(TARGET_CPU $ENV{TARGET_CPU})
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64.*|AARCH64.*|arm64.*|ARM64.*)")
  set(HOST_CPU aarch64)
  if(NOT DEFINED ENV{TARGET_CPU})
    set(TARGET_CPU aarch64)
  else()
    set(TARGET_CPU $ENV{TARGET_CPU})
  endif()
else()
  message(FATAL_ERROR "${CMAKE_SYSTEM_PROCESSOR} is not suppported")
endif()

if((NOT ${TARGET_CPU} STREQUAL  "x86") AND (NOT ${TARGET_CPU} STREQUAL  "aarch64"))
  message(FATAL_ERROR "TARGET_CPU=${TARGET_CPU} is not suppported. Only supported values are x86 or aarch64.")
endif() 

if(NOT DEFINED ENV{SOC} OR ENV{SOC} STREQUAL "" )
    message(FATAL_ERROR "SOC not specicfied, please export SOC variable")
endif()

string(TOLOWER $ENV{SOC} SOC)
if(${SOC} STREQUAL "AM62" )
  message(FATAL_ERROR "edgeai-tidl-tools currently does not support C++ wrapper and examples for AM62")
endif()

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
add_compile_options(-DXNN_ENABLE=1)

# Add -w to supress all warning to gurad warnings coming from tvm
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -Wno-unused-result")

set(TFLITE_2_12_LIBS 
    absl_base
    absl_log_severity
    absl_malloc_internal
    absl_raw_logging_internal
    absl_spinlock_wait
    absl_strerror
    absl_throw_delegate
    absl_hashtablez_sampler
    absl_raw_hash_set
    absl_debugging_internal
    absl_demangle_internal
    absl_stacktrace
    absl_symbolize
    absl_flags
    absl_flags_commandlineflag
    absl_flags_commandlineflag_internal
    absl_flags_config
    absl_flags_internal
    absl_flags_marshalling
    absl_flags_private_handle_accessor
    absl_flags_program_name
    absl_flags_reflection
    absl_city
    absl_hash
    absl_low_level_hash
    absl_int128
    absl_exponential_biased
    absl_status
    absl_cord
    absl_cord_internal
    absl_cordz_functions
    absl_cordz_handle
    absl_cordz_info
    absl_str_format_internal
    absl_strings
    absl_strings_internal
    absl_graphcycles_internal
    absl_synchronization
    absl_civil_time
    absl_time
    absl_time_zone
    absl_bad_optional_access
    absl_bad_variant_access
    flatbuffers
    fft2d_fftsg2d
    fft2d_fftsg
    cpuinfo
    farmhash
    ruy_allocator
    ruy_apply_multiplier
    ruy_blocking_counter
    ruy_block_map
    ruy_context
    ruy_context_get_ctx
    ruy_cpuinfo
    ruy_ctx
    ruy_denormal
    ruy_frontend
    ruy_have_built_path_for_avx2_fma
    ruy_have_built_path_for_avx512
    ruy_have_built_path_for_avx
    ruy_kernel_arm
    ruy_kernel_avx2_fma
    ruy_kernel_avx512
    ruy_kernel_avx
    ruy_pack_arm
    ruy_pack_avx2_fma
    ruy_pack_avx512
    ruy_pack_avx
    ruy_prepacked_cache
    ruy_prepare_packed_matrices
    ruy_system_aligned_alloc
    ruy_thread_pool
    ruy_trmul
    ruy_tune
    ruy_wait
    ruy_profiler_instrumentation
    pthreadpool
    XNNPACK
)

if((${HOST_CPU} STREQUAL  "x86") AND (${TARGET_CPU}  STREQUAL  "x86"))
  # x86 native buildsystem
  message(STATUS "Compiling for x86...")

  if(NOT EXISTS $ENV{TIDL_TOOLS_PATH}/)
    message (FATAL_ERROR  "TIDL_TOOLS_PATH: $ENV{TIDL_TOOLS_PATH} is not found")
  endif()

  message(STATUS "Directory of ONNX_DEPS: ${CMAKE_CURRENT_LIST_DIR}/../../tools/osrt_deps/onnx_1.23.0_x86_u22/")
  if (EXISTS ${CMAKE_CURRENT_LIST_DIR}/../../tools/osrt_deps/onnx_1.23.0_x86_u22/)
    set(ONNXRT_INSTALL_DIR ${CMAKE_CURRENT_LIST_DIR}/../../tools/osrt_deps/onnx_1.23.0_x86_u22/)
  else()
      message (FATAL_ERROR  "ONNX_DEPS not found")
  endif()

  message(STATUS "Directory of TFLITE_DEPS: ${CMAKE_CURRENT_LIST_DIR}/../../tools/osrt_deps/tflite_2.12_x86_u22/")
  if (EXISTS ${CMAKE_CURRENT_LIST_DIR}/../../tools/osrt_deps/tflite_2.12_x86_u22/)
    set(TFLITE_INSTALL_DIR ${CMAKE_CURRENT_LIST_DIR}/../../tools/osrt_deps/tflite_2.12_x86_u22/)
  else()
      message (FATAL_ERROR  "TFLITE_DEPS not found")
  endif()

  message(STATUS "Directory of TVM_DEPS: ${CMAKE_CURRENT_LIST_DIR}/../../tools/osrt_deps/tvm_0.18.0_x86_u22")
  if (EXISTS ${CMAKE_CURRENT_LIST_DIR}/../../tools/osrt_deps/tvm_0.18.0_x86_u22/)
    set(TVM_INSTALL_DIR ${CMAKE_CURRENT_LIST_DIR}/../../tools/osrt_deps/tvm_0.18.0_x86_u22/)
  else()
    message (FATAL_ERROR  "TVM_DEPS not found")
  endif()

  set(X86_INCLUDE_DIRS 
      ${PROJECT_SOURCE_DIR}

      /usr/local/include
      /usr/include/glib-2.0/
      
      ${ONNXRT_INSTALL_DIR}/onnxruntime/core/session
      ${ONNXRT_INSTALL_DIR}/onnxruntime/include
      ${ONNXRT_INSTALL_DIR}/onnxruntime/include/onnxruntime
      ${ONNXRT_INSTALL_DIR}/onnxruntime/include/onnxruntime/core/session

      ${TFLITE_INSTALL_DIR}/tensorflow    
      ${TFLITE_INSTALL_DIR}/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include
      ${TFLITE_INSTALL_DIR}/tensorflow/tensorflow/lite/tools/pip_package/gen/tflite_pip/python3/cmake_build/flatbuffers/include/

      ${TVM_INSTALL_DIR}/include
      ${TVM_INSTALL_DIR}/3rdparty/dlpack/include
      ${TVM_INSTALL_DIR}/3rdparty/dmlc-core/include

      $ENV{TIDL_TOOLS_PATH}
  )

  set(SYSTEM_LINK_LIBS
      stdc++fs
      onnxruntime
      tensorflow-lite
      tvm_runtime
      ${TFLITE_2_12_LIBS}
      )

  include_directories(${X86_INCLUDE_DIRS})

  link_directories(/usr/lib
                  /usr/local/lib
                  ${ONNXRT_INSTALL_DIR}

                  ${TFLITE_INSTALL_DIR}/
                  ${TFLITE_INSTALL_DIR}/tflite_2.12
                  ${TFLITE_INSTALL_DIR}/tflite_2.12/ruy-build
                  ${TFLITE_INSTALL_DIR}/tflite_2.12/fft2d-build
                  ${TFLITE_INSTALL_DIR}/tflite_2.12/cpuinfo-build
                  ${TFLITE_INSTALL_DIR}/tflite_2.12/flatbuffers-build
                  ${TFLITE_INSTALL_DIR}/tflite_2.12/farmhash-build
                  ${TFLITE_INSTALL_DIR}/tflite_2.12/xnnpack-build
                  ${TFLITE_INSTALL_DIR}/tflite_2.12/abseil-cpp-build
                  ${TFLITE_INSTALL_DIR}/tflite_2.12/pthreadpool

                  ${TVM_INSTALL_DIR}

                  $ENV{TIDL_TOOLS_PATH}/osrt_deps
                  $ENV{TIDL_TOOLS_PATH}
                  )
elseif((${HOST_CPU} STREQUAL  "aarch64") AND (${TARGET_CPU}  STREQUAL  "aarch64"))
  # aarch64 native buildsystem
  message(STATUS "Compiling for aarch64...")

  if(ENABLE_SDK_9_2_COMPATIBILITY)
    set(TARGET_DEVICE_PYTHON python3.10)
  else()
    set(TARGET_DEVICE_PYTHON python3.12)
  endif()

  set(AARCH64_INCLUDE_DIRS
      ${PROJECT_SOURCE_DIR}

      /usr/include
      /usr/local/include
      /usr/include/processor_sdk/c7x-mma-tidl/arm-tidl/rt/inc/

      /usr/include/onnxruntime/include
      /usr/include/onnxruntime/include/onnxruntime/core/session

      /usr/include/tensorflow
      /usr/include/tensorflow/lite/tools/pip_package/gen/tflite_pip/python3/cmake_build/flatbuffers/include/
      /usr/include/tensorflow/tensorflow/lite/tools/pip_package/gen/tflite_pip/python3/cmake_build/flatbuffers/include/

      /usr/include/tvm/tvm/include
      /usr/include/tvm/tvm/3rdparty/dmlc-core/include/
      /usr/include/tvm/tvm/3rdparty/dlpack/include
     )

  set(SYSTEM_LINK_LIBS
      onnxruntime
      tensorflow-lite
      tvm_runtime
      ${TFLITE_2_12_LIBS}
      )

  include_directories(${AARCH64_INCLUDE_DIRS})

  link_directories(/usr/lib
                  /usr/lib/aarch64-linux-gnu
                  
                  /usr/lib/tflite_2.12/abseil-cpp-build
                  /usr/lib/tflite_2.12/ruy-build
                  /usr/lib/tflite_2.12/xnnpack-build
                  /usr/lib/tflite_2.12/fft2d-build
                  /usr/lib/tflite_2.12/cpuinfo-build
                  /usr/lib/tflite_2.12/flatbuffers-build
                  /usr/lib/tflite_2.12/farmhash-build
                  /usr/lib/tflite_2.12/pthreadpool
                  )
elseif((${HOST_CPU} STREQUAL  "x86") AND (${TARGET_CPU}  STREQUAL  "aarch64"))
    # aarch64 cross-compilation buildsystem
    message(STATUS "Cross-Compiling for aarch64...")

    if (NOT DEFINED ENV{SDK_PATH})
      message(FATAL_ERROR "Environment variable SDK_PATH needs to be set for cross-compilation")
    endif()

    set(TARGET_FS_PATH $ENV{SDK_PATH}/targetfs)
    set(TOOLCHAIN_PATH $ENV{SDK_PATH}/toolchain/sysroots/x86_64-arago-linux/usr/bin/aarch64-oe-linux/)

    message(STATUS "TARGET_FS_PATH: ${TARGET_FS_PATH}")
    if (NOT EXISTS ${TARGET_FS_PATH})
        message (FATAL_ERROR  "TARGET_FS_PATH=${TARGET_FS_PATH} does not exist")
    endif()

    message(STATUS "TOOLCHAIN_PATH: ${TOOLCHAIN_PATH}")
    if (NOT EXISTS ${TOOLCHAIN_PATH})
        message (FATAL_ERROR  "TOOLCHAIN_PATH=${TOOLCHAIN_PATH} does not exist")
    endif()

    set(CMAKE_C_COMPILER ${TOOLCHAIN_PATH}/aarch64-oe-linux-gcc)
    set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PATH}/aarch64-oe-linux-g++)
    
    # Add sysroot to compiler and linker flags to find system libraries in target filesystem
    set(CMAKE_SYSROOT ${TARGET_FS_PATH})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --sysroot=${TARGET_FS_PATH}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --sysroot=${TARGET_FS_PATH}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --sysroot=${TARGET_FS_PATH}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --sysroot=${TARGET_FS_PATH}")

    if(ENABLE_SDK_9_2_COMPATIBILITY)
      set(TARGET_DEVICE_PYTHON python3.10)
    else()
      set(TARGET_DEVICE_PYTHON python3.12)
    endif()
  
    set(AARCH64_INCLUDE_DIRS
        ${PROJECT_SOURCE_DIR}

        ${TARGET_FS_PATH}/usr/include
        ${TARGET_FS_PATH}/usr/local/include
        ${TARGET_FS_PATH}/usr/include/processor_sdk/c7x-mma-tidl/arm-tidl/rt/inc/

        ${TARGET_FS_PATH}/usr/include/onnxruntime/include
        ${TARGET_FS_PATH}/usr/include/onnxruntime/include/onnxruntime/core/session

        ${TARGET_FS_PATH}/usr/include/tensorflow
        ${TARGET_FS_PATH}/usr/include/tensorflow/lite/tools/pip_package/gen/tflite_pip/python3/cmake_build/flatbuffers/include/
        ${TARGET_FS_PATH}/usr/include/tensorflow/tensorflow/lite/tools/pip_package/gen/tflite_pip/python3/cmake_build/flatbuffers/include/

        ${TARGET_FS_PATH}/usr/include/tvm/tvm/include
        ${TARGET_FS_PATH}/usr/include/tvm/tvm/3rdparty/dmlc-core/include
        ${TARGET_FS_PATH}/usr/include/tvm/tvm/3rdparty/dlpack/include
       )
  
    set(SYSTEM_LINK_LIBS
        onnxruntime
        tensorflow-lite
        tvm_runtime
        ${TFLITE_2_12_LIBS}
        )
  
    include_directories(${AARCH64_INCLUDE_DIRS})
  
    link_directories(${TARGET_FS_PATH}/usr/lib
                    ${TARGET_FS_PATH}/usr/lib/aarch64-linux-gnu
                    
                    ${TARGET_FS_PATH}/usr/lib/tflite_2.12/abseil-cpp-build
                    ${TARGET_FS_PATH}/usr/lib/tflite_2.12/ruy-build
                    ${TARGET_FS_PATH}/usr/lib/tflite_2.12/xnnpack-build
                    ${TARGET_FS_PATH}/usr/lib/tflite_2.12/fft2d-build
                    ${TARGET_FS_PATH}/usr/lib/tflite_2.12/cpuinfo-build
                    ${TARGET_FS_PATH}/usr/lib/tflite_2.12/flatbuffers-build
                    ${TARGET_FS_PATH}/usr/lib/tflite_2.12/farmhash-build
                    ${TARGET_FS_PATH}/usr/lib/tflite_2.12/pthreadpool
  
                    ${TARGET_FS_PATH}/usr/lib/${TARGET_DEVICE_PYTHON}/site-packages/tvm
                    )
else()
  message(FATAL_ERROR "Compilation on HOST_CPU=${HOST_CPU} for TARGET_CPU=${TARGET_CPU} is not supported.")
endif()



# Build library
function(build_lib lib_name lib_type lib_ver)
    add_library(${lib_name} ${lib_type} ${ARGN})

    SET_TARGET_PROPERTIES(${lib_name}
                        PROPERTIES
                        VERSION ${lib_ver}
                        )

    set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/onnxrt)

    FILE(GLOB HDRS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)

    install(TARGETS ${lib_name}
            EXPORT ${lib_name}Targets
            LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}  # Shared Libs
            ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}  # Static Libs
            RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}  # Executables, DLLs
        )

    # Specify the header files to install
    install(FILES ${HDRS} DESTINATION ${INCLUDE_INSTALL_DIR})
endfunction()

# Build app
function(build_app app_name)
    add_executable(${app_name} ${ARGN})
    
    target_link_libraries(${app_name}
                          -Wl,--unresolved-symbols=ignore-in-shared-libs,--start-group
                          ${COMMON_LINK_LIBS}
                          ${TARGET_LINK_LIBS}
                          ${SYSTEM_LINK_LIBS}
                          -Wl,--end-group
                         )
    set(BIN_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
endfunction()
