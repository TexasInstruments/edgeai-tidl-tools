include(GNUInstallDirs)

# add_compile_options(-std=c++11)


IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE} PROJECT_NAME = ${PROJECT_NAME}")

SET(CMAKE_FIND_LIBRARY_PREFIXES "" "lib")
SET(CMAKE_FIND_LIBRARY_SUFFIXES ".a" ".lib" ".so")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../lib/${CMAKE_BUILD_TYPE})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../bin/${CMAKE_BUILD_TYPE})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../bin/${CMAKE_BUILD_TYPE})


if(NOT EXISTS $ENV{TIDL_TOOLS_PATH}/)
  message (FATAL_ERROR  "TIDL_TOOLS_PATH: $ENV{TIDL_TOOLS_PATH} is not found")
endif()


if (EXISTS $ENV{TIDL_TOOLS_PATH}/osrt_deps/tflite_2.12/) # check for container
  set(TENSORFLOW_INSTALL_DIR $ENV{TIDL_TOOLS_PATH}/osrt_deps/tflite_2.12/)
  message (STATUS  "setting TENSORFLOW_INSTALL_DIR path:${TENSORFLOW_INSTALL_DIR}") # check for PC
elseif (EXISTS $ENV{TIDL_TOOLS_PATH}/osrt_deps/tflite_2.12_x86_u22/)
  set(TENSORFLOW_INSTALL_DIR $ENV{TIDL_TOOLS_PATH}/osrt_deps/tflite_2.12_x86_u22/)
  message (STATUS  "setting TENSORFLOW_INSTALL_DIR path:${TENSORFLOW_INSTALL_DIR}")
else()
  # avoid warning in case of hw accelerated device which have this in filesystem
  if( NOT ( (NOT ${TARGET_DEVICE} STREQUAL  "am62") AND (${TARGET_CPU} STREQUAL  "arm" AND ${HOST_CPU} STREQUAL  "arm"))  )
    message (WARNING  "TENSORFLOW_INSTALL_DIR is not set")
  endif()
endif()


if (EXISTS $ENV{TIDL_TOOLS_PATH}/osrt_deps/onnx_1.14.0_x86_u22/)
  set(ONNXRT_INSTALL_DIR $ENV{TIDL_TOOLS_PATH}/osrt_deps/onnx_1.14.0_x86_u22/)# check for PC
else()
  # avoid warning in case of  hw accelerated device which have this in filesystem
  if( NOT ((NOT ${TARGET_DEVICE} STREQUAL  "am62") AND (${TARGET_CPU} STREQUAL  "arm" AND ${HOST_CPU} STREQUAL  "arm"))  )
    message (WARNING  "ONNXRT_INSTALL_DIR is not set")
  endif()
endif()

if (EXISTS $ENV{TIDL_TOOLS_PATH}/osrt_deps/dlr_1.10.0_x86_u22/)
  set(DLR_INSTALL_DIR $ENV{TIDL_TOOLS_PATH}/osrt_deps/dlr_1.10.0_x86_u22/)
  message (STATUS  "setting DLR_INSTALL_DIR path:${DLR_INSTALL_DIR}")
else()
  # avoid warning in case of hw accelerated device which have this in filesystem
  if( NOT ((NOT ${TARGET_DEVICE} STREQUAL  "am62") AND (${TARGET_CPU} STREQUAL  "arm" AND ${HOST_CPU} STREQUAL  "arm"))  )
    message (WARNING  "DLR_INSTALL_DIR is not set")
  endif()
endif()


if (EXISTS $ENV{TIDL_TOOLS_PATH}/osrt_deps/opencv_4.2.0_x86_u22/)
  set(OPENCV_INSTALL_DIR $ENV{TIDL_TOOLS_PATH}/osrt_deps/opencv_4.2.0_x86_u22/)
  message (STATUS  "setting OPENCV_INSTALL_DIR path:${OPENCV_INSTALL_DIR}")
elseif (EXISTS $ENV{TIDL_TOOLS_PATH}/osrt_deps/opencv-4.1.0/)
  set(OPENCV_INSTALL_DIR $ENV{TIDL_TOOLS_PATH}/osrt_deps/opencv-4.1.0/)
  message (STATUS  "setting opencv path:${OPENCV_INSTALL_DIR}")
else ()
  # avoid warning in case of hw accelerated device which have this in filesystem
  if( NOT ((NOT ${TARGET_DEVICE} STREQUAL  "am62") AND (${TARGET_CPU} STREQUAL  "arm" AND ${HOST_CPU} STREQUAL  "arm"))  )
    message (WARNING  "OPENCV_INSTALL_DIR is not set")
  endif()
endif()


if(ARMNN_ENABLE)
  if( ${TARGET_CPU} STREQUAL  "x86" OR (NOT ${TARGET_DEVICE} STREQUAL "am62") )
    message(WARNING "ARMNN NOT supported on X86 and hw accelerated device")
    set(ARMNN_ENABLE 0)
    add_compile_options(-DARMNN_ENABLE=0)    
  else()
    if(NOT ARMNN_PATH)
      if (EXISTS $ENV{HOME}/armnn)
        set(ARMNN_PATH $ENV{HOME}/armnn)
      else()
        message(WARNING "ARMNN_PATH is not set")
      endif()
    endif()
      add_compile_options(-DARMNN_ENABLE=1)
    endif()
  endif()

set(TFLITE_2_12_LIBS 
  # Enable these when migrating to tflite 2.12
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
  #xnn lib
  XNNPACK
  # Enable these when migrating to tflite 2.12
)

set(TARGET_COMPILE_INCLUDE_DIR 
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/..
  ${PROJECT_SOURCE_DIR}/include
  /usr/local/include
  /usr/local/dlr
  /usr/include/gstreamer-1.0/
  /usr/include/glib-2.0/
  /usr/lib/aarch64-linux-gnu/glib-2.0/include
  /usr/include/opencv4/
  /usr/include/processor_sdk/vision_apps/
  #onnx headers from fs
  /usr/include/onnxruntime/core/session
  #tflite
  ${TENSORFLOW_INSTALL_DIR}/
  ${FLATBUFFERS_DIR}/include

  PUBLIC ${PROJECT_SOURCE_DIR}/post_process
  PUBLIC ${PROJECT_SOURCE_DIR}/pre_process
  PUBLIC ${PROJECT_SOURCE_DIR}/utils
)

set(PC_INCLUDE_DIR 
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/..
  ${PROJECT_SOURCE_DIR}/include

  /usr/local/include
  /usr/local/dlr
  /usr/include/gstreamer-1.0/
  /usr/include/glib-2.0/
  /usr/lib/aarch64-linux-gnu/glib-2.0/include
  /usr/include/opencv4/
  /usr/include/processor_sdk/vision_apps/

  # opencv libraries
  ${OPENCV_INSTALL_DIR}/opencv-4.2.0/modules/core/
  ${OPENCV_INSTALL_DIR}/opencv-4.2.0/build
  ${OPENCV_INSTALL_DIR}/opencv-4.2.0/modules/core/include/
  ${OPENCV_INSTALL_DIR}/opencv-4.2.0/modules/highgui/include/                  
  ${OPENCV_INSTALL_DIR}/opencv-4.2.0/modules/imgcodecs/include/
  ${OPENCV_INSTALL_DIR}/opencv-4.2.0/modules/videoio/include/
  ${OPENCV_INSTALL_DIR}/opencv-4.2.0/modules/imgproc/include/
  ${OPENCV_INSTALL_DIR}/opencv-4.2.0/cmake/

  #tflite              
  ${TENSORFLOW_INSTALL_DIR}/tflite_build/flatbuffers/include
  ${TENSORFLOW_INSTALL_DIR}/tensorflow/
  ${TENSORFLOW_INSTALL_DIR}/tensorflow/tensorflow/lite/tools/pip_package/gen/tflite_pip/python3/cmake_build/flatbuffers/include/

  ${ONNXRT_INSTALL_DIR}/onnxruntime/include
  ${ONNXRT_INSTALL_DIR}/onnxruntime/include/onnxruntime
  ${ONNXRT_INSTALL_DIR}/onnxruntime/include/onnxruntime/core/session

  ${DLR_INSTALL_DIR}/neo-ai-dlr/include
  ${DLR_INSTALL_DIR}/neo-ai-dlr/3rdparty/tvm/3rdparty/dlpack/include

  PUBLIC ${PROJECT_SOURCE_DIR}/post_process
  PUBLIC ${PROJECT_SOURCE_DIR}/pre_process
  PUBLIC ${PROJECT_SOURCE_DIR}/utils
  $ENV{TIDL_TOOLS_PATH}
)
if(${TARGET_DEVICE} STREQUAL  "am62" AND  (${TARGET_CPU} STREQUAL  "x86" AND ${HOST_CPU} STREQUAL  "x86"))
  message(STATUS "Compiling for x86 with am62 config")
  add_compile_options(-DDEVICE_AM62=1)
  set(CMAKE_C_COMPILER gcc)
  set(CMAKE_CXX_COMPILER g++)
  add_compile_options(-DDEVICE_AM62=1)
  add_compile_options(-DXNN_ENABLE=1)

  include_directories(
              ${PROJECT_SOURCE_DIR}
              ${PROJECT_SOURCE_DIR}/..
              ${PROJECT_SOURCE_DIR}/include
              /usr/local/include
              /usr/local/dlr
              /usr/include/gstreamer-1.0/
              /usr/include/glib-2.0/
              /usr/lib/aarch64-linux-gnu/glib-2.0/include
              /usr/include/opencv4/
              /usr/include/processor_sdk/vision_apps/
              
              # opencv libraries
              ${OPENCV_INSTALL_DIR}/cmake              
              ${OPENCV_INSTALL_DIR}/modules/highgui/include
              ${OPENCV_INSTALL_DIR}/modules/core/include
              ${OPENCV_INSTALL_DIR}/modules/imgproc/include/opencv2
              ${OPENCV_INSTALL_DIR}/modules/imgproc/include/
              ${OPENCV_INSTALL_DIR}/modules/imgcodecs/include

              #tflite
              ${TENSORFLOW_INSTALL_DIR}/tensorflow_src
              ${TENSORFLOW_INSTALL_DIR}/tflite_build/flatbuffers/include

              ${ONNXRT_INSTALL_DIR}/include
              ${ONNXRT_INSTALL_DIR}/include/onnxruntime
              ${ONNXRT_INSTALL_DIR}/include/onnxruntime/core/session                    
              ${DLR_INSTALL_DIR}/include
              ${DLR_INSTALL_DIR}/3rdparty/tvm/3rdparty/dlpack/include
              PUBLIC ${PROJECT_SOURCE_DIR}/post_process
              PUBLIC ${PROJECT_SOURCE_DIR}/pre_process
              PUBLIC ${PROJECT_SOURCE_DIR}/utils

              
            )

  set(SYSTEM_LINK_LIBS
                      opencv_imgproc
                      opencv_imgcodecs
                      opencv_core
                      libtiff 
                      libwebp
                      libpng
                      libjpeg-turbo
                      IlmImf
                      zlib
                      libjasper
                      dlr
                      tensorflow-lite
                      onnxruntime
                      vx_tidl_rt
                      pthread
                      dl
                      yaml-cpp
                      stdc++fs
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
                      pthreadpool
                      #xnn lib
                      XNNPACK
                      ${TFLITE_2_12_LIBS}
    )

  link_directories(
                    /usr/lib 
                    /usr/local/dlr
                    /usr/lib/aarch64-linux-gnu
                    /usr/lib/python3.10/site-packages/dlr/
                    $ENV{HOME}/.local/dlr/ 
                    # opencv libraries
                    ${OPENCV_INSTALL_DIR}/cmake/lib
                    ${OPENCV_INSTALL_DIR}/cmake/3rdparty/lib
                    ${OPENCV_INSTALL_DIR}/modules/core/include 

                    #tesnorflow 2.12 and dependencies
                    ${TENSORFLOW_INSTALL_DIR}/tflite_2.12
                    ${TENSORFLOW_INSTALL_DIR}/tflite_2.12/ruy-build/
                    ${TENSORFLOW_INSTALL_DIR}/tflite_2.12/fft2d-build
                    ${TENSORFLOW_INSTALL_DIR}/tflite_2.12/cpuinfo-build
                    ${TENSORFLOW_INSTALL_DIR}/tflite_2.12/flatbuffers-build
                    ${TENSORFLOW_INSTALL_DIR}/tflite_2.12/farmhash-build
                    ${TENSORFLOW_INSTALL_DIR}/tflite_2.12/xnnpack-build
                    ${TENSORFLOW_INSTALL_DIR}/tflite_2.12/abseil-cpp-build
                    ${TENSORFLOW_INSTALL_DIR}/tflite_2.12/pthreadpool
                    
                    #for onnx  lib
                    $ENV{TIDL_TOOLS_PATH}
                  )
endif()

if(NOT ${TARGET_DEVICE} STREQUAL  "am62" AND  (${TARGET_CPU} STREQUAL  "x86" AND ${HOST_CPU} STREQUAL  "x86"))
  message(STATUS "Compiling for x86 with ${TARGET_DEVICE} config")
  add_compile_options(-DDEVICE_AM62A=1)
  set(CMAKE_C_COMPILER gcc)
  set(CMAKE_CXX_COMPILER g++)
  #enbale xnn since tflite 2.12
  add_compile_options(-DXNN_ENABLE=1)

  include_directories(
    ${PC_INCLUDE_DIR}
  )

  set(SYSTEM_LINK_LIBS
    opencv_imgproc
    opencv_imgcodecs
    opencv_core
    # tiff 
    # webp
    # libpng
    # libjpeg-turbo
    # IlmImf
    # zlib
    # libjasper
    dlr
    tensorflow-lite
    onnxruntime
    vx_tidl_rt
    pthread
    dl
    yaml-cpp
    stdc++fs
    ${TFLITE_2_12_LIBS}
    )

  link_directories(
    /usr/lib 
    /usr/local/dlr
    /usr/lib/aarch64-linux-gnu
    /usr/lib/python3.10/site-packages/dlr/
    $ENV{HOME}/.local/dlr/ 

    ${OPENCV_INSTALL_DIR}/opencv/

    ${ONNXRT_INSTALL_DIR}/

    #tesnorflow 2.12 and dependencies
    ${TENSORFLOW_INSTALL_DIR}/
    ${TENSORFLOW_INSTALL_DIR}/tflite_2.12
    ${TENSORFLOW_INSTALL_DIR}/tflite_2.12/ruy-build
    ${TENSORFLOW_INSTALL_DIR}/tflite_2.12/fft2d-build
    ${TENSORFLOW_INSTALL_DIR}/tflite_2.12/cpuinfo-build
    ${TENSORFLOW_INSTALL_DIR}/tflite_2.12/flatbuffers-build
    ${TENSORFLOW_INSTALL_DIR}/tflite_2.12/farmhash-build
    ${TENSORFLOW_INSTALL_DIR}/tflite_2.12/xnnpack-build
    ${TENSORFLOW_INSTALL_DIR}/tflite_2.12/abseil-cpp-build
    ${TENSORFLOW_INSTALL_DIR}/tflite_2.12/pthreadpool
    
    
    $ENV{TIDL_TOOLS_PATH}/osrt_deps/
    $ENV{TIDL_TOOLS_PATH}/
  )
endif()


if((${TARGET_DEVICE} STREQUAL  "am62") AND (${TARGET_CPU} STREQUAL  "arm" AND ${HOST_CPU} STREQUAL  "x86") )
  message(STATUS "cross compiling for AM62")
  add_compile_options(-DDEVICE_AM62=1)
  add_compile_options(-DXNN_ENABLE=1)

  if(NOT ARM64GCC_PATH)
    if (EXISTS $ENV{HOME}/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu)
      set(ARM64GCC_PATH $ENV{HOME}/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu)
    else()
     message(WARNING "ARM64GCC_PATH is not set")
    endif()
  endif()

  if(NOT TARGET_FS_PATH)
    if (EXISTS $ENV{HOME}/targetfs)
      set(TARGET_FS_PATH $ENV{HOME}/targetfs)
    else()
      message(WARNING "TARGET_FS_PATH is not set")
    endif()
  endif()

  set(ARMCC_PREFIX ${ARM64GCC_PATH}/bin/aarch64-none-linux-gnu-)
  set(CMAKE_C_COMPILER ${ARMCC_PREFIX}gcc)
  set(CMAKE_CXX_COMPILER ${ARMCC_PREFIX}g++)

  link_directories(
                  #AM62 targetfs
                  ${TARGET_FS_PATH}/usr/lib
                  ${TARGET_FS_PATH}/usr/lib/glib-2.0
                  ${TARGET_FS_PATH}/usr/lib/python3.10/site-packages  

  )
  set(SYSTEM_LINK_LIBS
                  tensorflow-lite
                  z
                  opencv_imgproc
                  opencv_imgcodecs
                  opencv_core
                  jpeg
                  webp
                  png16
                  tiff
                  onnxruntime
                  dl
                  dlr
                  yaml-cpp
                  fft2d_fftsg2d
                  fft2d_fftsg
                  cpuinfo
                  farmhash
                  XNNPACK
                  pthreadpool
                  pthread
                  tbb
  )
  include_directories(
                  ${PROJECT_SOURCE_DIR}
                  ${PROJECT_SOURCE_DIR}/..
                  ${PROJECT_SOURCE_DIR}/include

                  ${TARGET_FS_PATH}/usr/include
                  ${TARGET_FS_PATH}/usr/include/gstreamer-1.0/
                  ${TARGET_FS_PATH}/usr/include/opencv4/
                  ${TARGET_FS_PATH}/usr/include/processor_sdk/vision_apps/
                  
                  #tflite
                  ${TENSORFLOW_INSTALL_DIR}/tensorflow_src
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build/flatbuffers/include

                  ${TARGET_FS_PATH}/usr/include/onnxruntime
                  ${TARGET_FS_PATH}/usr/include/onnxruntime/core/session                    
                  ${DLR_INSTALL_DIR}/include
                  ${DLR_INSTALL_DIR}/3rdparty/tvm/3rdparty/dlpack/include
                  ${TARGET_FS_PATH}/usr/include/opencv4/opencv2/core/include
                  ${TARGET_FS_PATH}/usr/include/opencv4/opencv2/highgui/include
                  ${TARGET_FS_PATH}/usr/include/opencv4/opencv2/imgcodecs/include
                  ${TARGET_FS_PATH}/usr/include/opencv4/opencv2/videoio/include
                  ${TARGET_FS_PATH}/usr/include/opencv4/opencv2/imgproc/include
                  #${TARGET_FS_PATH}/usr/include/opencv4/opencv2/cmake
                  
                  PUBLIC ${PROJECT_SOURCE_DIR}/post_process
                  PUBLIC ${PROJECT_SOURCE_DIR}/pre_process
                  PUBLIC ${PROJECT_SOURCE_DIR}/utils
  )
endif()

if((NOT ${TARGET_DEVICE} STREQUAL  "am62") AND (${TARGET_CPU} STREQUAL  "arm" AND ${HOST_CPU} STREQUAL  "x86") )
  message(STATUS "cross compiling for TARGET_DEVICE:${TARGET_DEVICE}")
  add_compile_options(-DXNN_ENABLE=1)
  
  if (EXISTS $ENV{TIDL_TOOLS_PATH}/../gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/)
    set(ARM64GCC_PATH $ENV{TIDL_TOOLS_PATH}/../gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu)
    message (STATUS  "setting ARM64GCC_PATH path:${ARM64GCC_PATH}")
  else()
    message(WARNING "ARM64GCC_PATH is not set")
  endif()

  #User need to set this
  if(EXISTS $ENV{TARGET_FS_PATH}/)
    set(TARGET_FS_PATH $ENV{TARGET_FS_PATH})
    message (STATUS  "setting TARGET_FS_PATH path:$ENV{TARGET_FS_PATH}")
  else()
    message(FATAL_ERROR "TARGET_FS_PATH is not set")
  endif()

  set(ARMCC_PREFIX ${ARM64GCC_PATH}/bin/aarch64-none-linux-gnu-)
  set(CMAKE_C_COMPILER ${ARMCC_PREFIX}gcc)
  set(CMAKE_CXX_COMPILER ${ARMCC_PREFIX}g++)

  link_directories(                
    #Device targetfs
    ${TARGET_FS_PATH}/usr/lib
    ${TARGET_FS_PATH}/usr/lib/glib-2.0
    ${TARGET_FS_PATH}/usr/lib/python3.10/site-packages  
    # Enable these when migrating to tflite 2.12
    ${TARGET_FS_PATH}/usr/lib/tflite_2.12/abseil-cpp-build
    ${TARGET_FS_PATH}/usr/lib/tflite_2.12/ruy-build
    ${TARGET_FS_PATH}/usr/lib/tflite_2.12/xnnpack-build
    ${TARGET_FS_PATH}/usr/lib/tflite_2.12/fft2d-build
    ${TARGET_FS_PATH}/usr/lib/tflite_2.12/cpuinfo-build
    ${TARGET_FS_PATH}/usr/lib/tflite_2.12/flatbuffers-build
    ${TARGET_FS_PATH}/usr/lib/tflite_2.12/farmhash-build
    ${TARGET_FS_PATH}/usr/lib/tflite_2.12/pthreadpool
    ${TARGET_FS_PATH}/usr/lib/python3.10/site-packages/dlr
    # Enable these when migrating to tflite 2.12
  )
  set(SYSTEM_LINK_LIBS
    tensorflow-lite
    pcre
    ffi  
    z
    opencv_imgproc
    opencv_imgcodecs
    opencv_core
    tbb
    jpeg
    webp
    png16
    tiff
    onnxruntime
    dl
    dlr
    yaml-cpp
    tivision_apps
    GLESv2
    EGL
    gbm
    glapi
    expat
    drm
    wayland-client
    wayland-server
    pthread
    vx_tidl_rt
    ti_rpmsg_char
    ${TFLITE_2_12_LIBS}
  )
  include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/..
    ${PROJECT_SOURCE_DIR}/include

    ${TARGET_FS_PATH}/usr/include/
    ${TARGET_FS_PATH}/usr/include/opencv-4.2.0/modules/core/include/
    ${TARGET_FS_PATH}/usr/include/opencv-4.2.0/build
    ${TARGET_FS_PATH}/usr/include/opencv-4.2.0/modules/highgui/include/
    ${TARGET_FS_PATH}/usr/include/opencv-4.2.0/modules/imgcodecs/include/
    ${TARGET_FS_PATH}/usr/include/opencv-4.2.0/modules/videoio/include/
    ${TARGET_FS_PATH}/usr/include/opencv-4.2.0/modules/imgproc/include/
    ${TARGET_FS_PATH}/usr/include/tensorflow/lite/tools/pip_package/gen/tflite_pip/python3/cmake_build/flatbuffers/include/
    ${TARGET_FS_PATH}/usr/include/onnxruntime/include/
    ${TARGET_FS_PATH}/usr/include/onnxruntime/include/onnxruntime/core/session/
    ${TARGET_FS_PATH}/usr/lib/python3.10/site-packages/dlr/include/

    #armnn
    ${ARMNN_PATH}/delegate/include
    ${ARMNN_PATH}/include

    PUBLIC ${PROJECT_SOURCE_DIR}/post_process
    PUBLIC ${PROJECT_SOURCE_DIR}/pre_process
    PUBLIC ${PROJECT_SOURCE_DIR}/utils
  )
endif()


if((${TARGET_DEVICE} STREQUAL  "am62") AND (${TARGET_CPU} STREQUAL  "arm" AND ${HOST_CPU} STREQUAL  "arm") )
  message(STATUS "native compiling for AM62")
  if(NOT FLATBUFFERS_DIR)
    if (EXISTS $ENV{HOME}/flatbuffers)
      set(FLATBUFFERS_DIR $ENV{HOME}/flatbuffers)
    else()
      message(WARNING "FLATBUFFERS_DIR is not set")
    endif()
  endif()
  add_compile_options(-DDEVICE_AM62=1)
  add_compile_options(-DXNN_ENABLE=1)

  include_directories(
              ${TARGET_FS_INCLUDE_DIR}
            )

  set(SYSTEM_LINK_LIBS
                opencv_imgproc
                opencv_imgcodecs
                opencv_core
                # dlr
                tensorflow-lite
                onnxruntime            
                pthread
                dl
                yaml-cpp
                pthreadpool
                XNNPACK
    )



    link_directories(
                    /usr/lib 
                    /usr/local/dlr
                    /usr/lib/aarch64-linux-gnu
                    /usr/lib/python3.12/site-packages/dlr/
                    $ENV{HOME}/.local/dlr/                 
    )
endif()

if( ((NOT ${TARGET_DEVICE} STREQUAL  "am62") AND (${TARGET_CPU} STREQUAL  "arm" AND ${HOST_CPU} STREQUAL  "arm"))  )
  message(NOTICE "native compiling for TARGET_DEVICE:${TARGET_DEVICE} ")
  add_compile_options(-DXNN_ENABLE=1)

  set(CMAKE_C_COMPILER gcc)
  set(CMAKE_CXX_COMPILER g++)

  include_directories(
    ${TARGET_FS_INCLUDE_DIR}
  )
  link_directories(  
                  /usr/lib/opencv/ #for container
                  /usr/lib
                  # Enable these when migrating to tflite 2.12
                  /usr/lib/tflite_2.12/abseil-cpp-build
                  /usr/lib/tflite_2.12/ruy-build
                  /usr/lib/tflite_2.12/xnnpack-build
                  /usr/lib/tflite_2.12/fft2d-build
                  /usr/lib/tflite_2.12/cpuinfo-build
                  /usr/lib/tflite_2.12/flatbuffers-build
                  /usr/lib/tflite_2.12/farmhash-build
                  /usr/lib/tflite_2.12/pthreadpool
                  # Enable these when migrating to tflite 2.12
                  /usr/local/dlr
                  /usr/lib/aarch64-linux-gnu
                  /usr/lib/python3.12/site-packages/dlr/
                  /usr/local/lib/python3.12/dist-packages/dlr/
                  $ENV{HOME}/.local/dlr/                  
  )
  set(SYSTEM_LINK_LIBS
                  opencv_imgproc
                  opencv_imgcodecs
                  opencv_core
                  dlr
                  tensorflow-lite
                  onnxruntime
                  vx_tidl_rt
                  pthread
                  dl
                  yaml-cpp
                  jpeg
                  webp
                  png16
                  tiff
                  ${ADV_UTILS_LIB}
                  ${TFLITE_2_12_LIBS}
  )
  include_directories(
                  /usr/include
                  /usr/local/include
                  /usr/local/dlr
                  /usr/lib/python3.12/site-packages/dlr/include #for am68pa evm
                  /usr/include/tensorflow/lite/tools/pip_package/gen/tflite_pip/python3/cmake_build/flatbuffers/include/
                  ${PROJECT_SOURCE_DIR}
                  ${PROJECT_SOURCE_DIR}/..
                  ${PROJECT_SOURCE_DIR}/include
                  /usr/include/opencv4/
                  /usr/include/opencv-4.2.0/# for container
                  /usr/include/opencv-4.2.0/build# for container
                  /usr/include/opencv-4.2.0/modules/core/include/# for container
                  /usr/include/opencv-4.2.0/modules/highgui/include/                  # for container
                  /usr/include/opencv-4.2.0/modules/imgcodecs/include/# for container
                  /usr/include/opencv-4.2.0/modules/videoio/include/# for container
                  /usr/include/opencv-4.2.0/modules/imgproc/include/# for container
                  /usr/include/opencv-4.2.0/cmake/# for container
                  /usr/include/tensorflow/tensorflow/lite/tools/pip_package/gen/tflite_pip/python3/cmake_build/flatbuffers/include/ #for container

                  
                  /usr/include/tensorflow
                  /usr/include/neo-ai-dlr/include
                  /usr/include/neo-ai-dlr/3rdparty/tvm/3rdparty/dlpack/include
                  /usr/include/onnxruntime/include
                  /usr/include/onnxruntime/include/onnxruntime/core/session
                  PUBLIC ${PROJECT_SOURCE_DIR}/post_process
                  PUBLIC ${PROJECT_SOURCE_DIR}/pre_process
                  PUBLIC ${PROJECT_SOURCE_DIR}/utils/include
  )
endif()


if(ARMNN_ENABLE)
  include_directories(
    ${ARMNN_PATH}/delegate/include
    ${ARMNN_PATH}/armnn/include
    ${ARMNN_PATH}/include
  )
  set(SYSTEM_LINK_LIBS
    ${SYSTEM_LINK_LIBS}
    armnn
    armnnDelegate
  )
  link_directories(
    #armnn lib need to remove once added to filesystem
    ${ARMNN_PATH}/build
    ${ARMNN_PATH}/build/delegate
    ${TIDL_TARGET_LIBS}
  )
endif()        

# Function for building a node:
# ARG0: app name
# ARG1: source list
function(build_app)
    set(app ${ARGV0})
    set(src ${ARGV1})
    add_executable(${app} ${${src}})
    target_include_directories(${app} 
        PUBLIC ${PROJECT_SOURCE_DIR}/post_process
        )
    target_include_directories(${app} 
        PUBLIC ${PROJECT_SOURCE_DIR}/utils
        )
    link_directories(${app} 
        PUBLIC ${CMAKE_SOURCE_DIR}/../lib/${CMAKE_BUILD_TYPE}        
        )
    
    if(NOT ${TARGET_DEVICE} STREQUAL  "am62")
      set(ADV_UTILS_LIB "utils_adv")
    endif()
    target_link_libraries(${app}
                          -Wl,--unresolved-symbols=ignore-in-shared-libs,--start-group
                          ${COMMON_LINK_LIBS}
                          ${TARGET_LINK_LIBS}
                          ${SYSTEM_LINK_LIBS}
                          post_process
                          pre_process
                          utils
                          ${ADV_UTILS_LIB}
                          ${lobs}
                          -Wl,--end-group)
endfunction()
