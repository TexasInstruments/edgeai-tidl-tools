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

if(NOT TENSORFLOW_INSTALL_DIR)
  if (EXISTS $ENV{TIDL_TOOLS_PATH}/osrt_deps/tflite_2.8/)
    set(TENSORFLOW_INSTALL_DIR $ENV{TIDL_TOOLS_PATH}/osrt_deps/tflite_2.8/tflite_2.8)
    message (STATUS  "setting TENSORFLOW_INSTALL_DIR path:${TENSORFLOW_INSTALL_DIR}")
  elseif (EXISTS $ENV{TIDL_TOOLS_PATH}/osrt_deps/tflite_2.8_x86_u18/)
    set(TENSORFLOW_INSTALL_DIR $ENV{TIDL_TOOLS_PATH}/osrt_deps/tflite_2.8_x86_u18/tflite_2.8)
    message (STATUS  "setting TENSORFLOW_INSTALL_DIR path:${TENSORFLOW_INSTALL_DIR}")
  else()
    # avoid warning in case of j7 device which have this in filesystem
    if( NOT ((${TARGET_DEVICE} STREQUAL  "j7") AND (${TARGET_CPU} STREQUAL  "arm" AND ${HOST_CPU} STREQUAL  "arm"))  )
      message (WARNING  "TENSORFLOW_INSTALL_DIR is not set")
    endif()
  endif()
endif()

if(NOT ONNXRT_INSTALL_DIR)
  if (EXISTS $ENV{TIDL_TOOLS_PATH}/osrt_deps/onnxruntime)
    set(ONNXRT_INSTALL_DIR $ENV{TIDL_TOOLS_PATH}/osrt_deps/onnxruntime)
  else()
    # avoid warning in case of j7 device which have this in filesystem
    if( NOT ((${TARGET_DEVICE} STREQUAL  "j7") AND (${TARGET_CPU} STREQUAL  "arm" AND ${HOST_CPU} STREQUAL  "arm"))  )
      message (WARNING  "ONNXRT_INSTALL_DIR is not set")
    endif()
  endif()
endif()

if(NOT DLR_INSTALL_DIR)
  if (EXISTS $ENV{TIDL_TOOLS_PATH}/osrt_deps/neo-ai-dlr)
    set(DLR_INSTALL_DIR $ENV{TIDL_TOOLS_PATH}/osrt_deps/neo-ai-dlr)
    message (STATUS  "setting DLR_INSTALL_DIR path:${DLR_INSTALL_DIR}")
  else()
    # avoid warning in case of j7 device which have this in filesystem
    if( NOT ((${TARGET_DEVICE} STREQUAL  "j7") AND (${TARGET_CPU} STREQUAL  "arm" AND ${HOST_CPU} STREQUAL  "arm"))  )
      message (WARNING  "DLR_INSTALL_DIR is not set")
    endif()
  endif()
endif()

if(NOT OPENCV_INSTALL_DIR)
  if (EXISTS $ENV{TIDL_TOOLS_PATH}/osrt_deps/opencv-4.2.0/)
    set(OPENCV_INSTALL_DIR $ENV{TIDL_TOOLS_PATH}/osrt_deps/opencv-4.2.0/)
    message (STATUS  "setting OPENCV_INSTALL_DIR path:${OPENCV_INSTALL_DIR}")
  elseif (EXISTS $ENV{TIDL_TOOLS_PATH}/osrt_deps/opencv-4.1.0/)
    set(OPENCV_INSTALL_DIR $ENV{TIDL_TOOLS_PATH}/osrt_deps/opencv-4.1.0/)
    message (STATUS  "setting opencv path:${OPENCV_INSTALL_DIR}")
  else ()
    # avoid warning in case of j7 device which have this in filesystem
    if( NOT ((${TARGET_DEVICE} STREQUAL  "j7") AND (${TARGET_CPU} STREQUAL  "arm" AND ${HOST_CPU} STREQUAL  "arm"))  )
      message (WARNING  "OPENCV_INSTALL_DIR is not set")
    endif()
  endif()
endif()

if(ARMNN_ENABLE)
  if( ${TARGET_CPU} STREQUAL  "x86" OR ${TARGET_DEVICE} STREQUAL "j7" )
    message(WARNING "ARMNN NOT supported on X86 and j7")
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

set ( TFLITE_2_8_LIBS 
            # Enable these when migrating to tflite 2.8
            flatbuffers
            fft2d_fftsg2d
            fft2d_fftsg
            cpuinfo
            clog
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
            # Enable these when migrating to tflite 2.8 
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
                      clog
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
    )

  link_directories(
                    /usr/lib 
                    /usr/local/dlr
                    /usr/lib/aarch64-linux-gnu
                    /usr/lib/python3.8/site-packages/dlr/
                    $ENV{HOME}/.local/dlr/ 
                    # opencv libraries
                    ${OPENCV_INSTALL_DIR}/cmake/lib
                    ${OPENCV_INSTALL_DIR}/cmake/3rdparty/lib
                    ${OPENCV_INSTALL_DIR}/modules/core/include 

                    #tesnorflow 2.8 and dependencies
                    ${TENSORFLOW_INSTALL_DIR}/tflite_build
                    ${TENSORFLOW_INSTALL_DIR}/tflite_build/_deps/ruy-build/ruy
                    ${TENSORFLOW_INSTALL_DIR}/tflite_build/pthreadpool
                    ${TENSORFLOW_INSTALL_DIR}/tflite_build/_deps/fft2d-build
                    ${TENSORFLOW_INSTALL_DIR}/tflite_build/_deps/cpuinfo-build
                    ${TENSORFLOW_INSTALL_DIR}/tflite_build/_deps/flatbuffers-build
                    ${TENSORFLOW_INSTALL_DIR}/tflite_build/_deps/clog-build
                    ${TENSORFLOW_INSTALL_DIR}/tflite_build/_deps/farmhash-build
                    ${TENSORFLOW_INSTALL_DIR}/tflite_build/_deps/xnnpack-build
                    
                    #for onnx  lib
                    $ENV{TIDL_TOOLS_PATH}
                  )
endif()

if(${TARGET_DEVICE} STREQUAL  "am62a" AND  (${TARGET_CPU} STREQUAL  "x86" AND ${HOST_CPU} STREQUAL  "x86"))
  message(STATUS "Compiling for x86 with am62a config")
  add_compile_options(-DDEVICE_AM62A=1)
  set(CMAKE_C_COMPILER gcc)
  set(CMAKE_CXX_COMPILER g++)
  add_compile_options(-DDEVICE_AM62A=1)
  #enbale xnn since tflite 2.8
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
                      clog
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
    )

  link_directories(
                    /usr/lib 
                    /usr/local/dlr
                    /usr/lib/aarch64-linux-gnu
                    /usr/lib/python3.8/site-packages/dlr/
                    $ENV{HOME}/.local/dlr/ 
                    # opencv libraries
                    ${OPENCV_INSTALL_DIR}/cmake/lib
                    ${OPENCV_INSTALL_DIR}/cmake/3rdparty/lib
                    ${OPENCV_INSTALL_DIR}/modules/core/include 

                    #tesnorflow 2.8 and dependencies
                    ${TENSORFLOW_INSTALL_DIR}/tflite_build
                    ${TENSORFLOW_INSTALL_DIR}/tflite_build/_deps/ruy-build/ruy
                    ${TENSORFLOW_INSTALL_DIR}/tflite_build/pthreadpool
                    ${TENSORFLOW_INSTALL_DIR}/tflite_build/_deps/fft2d-build
                    ${TENSORFLOW_INSTALL_DIR}/tflite_build/_deps/cpuinfo-build
                    ${TENSORFLOW_INSTALL_DIR}/tflite_build/_deps/flatbuffers-build
                    ${TENSORFLOW_INSTALL_DIR}/tflite_build/_deps/clog-build
                    ${TENSORFLOW_INSTALL_DIR}/tflite_build/_deps/farmhash-build
                    ${TENSORFLOW_INSTALL_DIR}/tflite_build/_deps/xnnpack-build
                    
                    #for onnx  lib
                    $ENV{TIDL_TOOLS_PATH}
                  )
endif()

if(${TARGET_DEVICE} STREQUAL  "j7" AND  (${TARGET_CPU} STREQUAL  "x86" AND ${HOST_CPU} STREQUAL  "x86"))
  message(STATUS "Compiling for x86 with j7 config")
  add_compile_options(-DDEVICE_J7=1)
  set(CMAKE_C_COMPILER gcc)
  set(CMAKE_CXX_COMPILER g++)

  link_directories(
                  # opencv libraries
                  $ENV{TIDL_TOOLS_PATH}/osrt_deps/opencv/
                  ${OPENCV_INSTALL_DIR}/cmake/lib/
                  ${OPENCV_INSTALL_DIR}/cmake/3rdparty/lib/
                  #common
                  /usr/lib 
                  /usr/lib/aarch64-linux-gnu                  
                  
                  ${TENSORFLOW_INSTALL_DIR}/ruy-build/ruy
                  ${TENSORFLOW_INSTALL_DIR}/ruy-build/
                  ${TENSORFLOW_INSTALL_DIR}/xnnpack-build
                  ${TENSORFLOW_INSTALL_DIR}/pthreadpool
                  ${TENSORFLOW_INSTALL_DIR}/fft2d-build
                  ${TENSORFLOW_INSTALL_DIR}/cpuinfo-build
                  ${TENSORFLOW_INSTALL_DIR}/flatbuffers-build
                  ${TENSORFLOW_INSTALL_DIR}/clog-build
                  ${TENSORFLOW_INSTALL_DIR}/farmhash-build

                  #tidl tools lib
                  $ENV{TIDL_TOOLS_PATH}
                  $ENV{TIDL_TOOLS_PATH}/osrt_deps

  )
  set(SYSTEM_LINK_LIBS
                  opencv_imgproc
                  opencv_imgcodecs
                  opencv_core
                  dlr
                  ${TFLITE_2_8_LIBS}
                  tensorflow-lite
                  onnxruntime
                  vx_tidl_rt
                  pthread
                  dl
                  yaml-cpp
                  stdc++fs
  )
  include_directories(
                  ${PROJECT_SOURCE_DIR}
                  ${PROJECT_SOURCE_DIR}/..
                  ${PROJECT_SOURCE_DIR}/include
                  
                  #tflite 2.8
                  $ENV{TIDL_TOOLS_PATH}/osrt_deps/tensorflow
                  $ENV{TIDL_TOOLS_PATH}/osrt_deps/tensorflow/tensorflow/lite/tools/pip_package/gen/tflite_pip/python3/cmake_build/flatbuffers/include/

                  ${ONNXRT_INSTALL_DIR}/include
                  ${ONNXRT_INSTALL_DIR}/include/onnxruntime
                  ${ONNXRT_INSTALL_DIR}/include/onnxruntime/core/session                    
                  ${DLR_INSTALL_DIR}/include
                  ${DLR_INSTALL_DIR}/3rdparty/tvm/3rdparty/dlpack/include

                  ${OPENCV_INSTALL_DIR}/
                  ${OPENCV_INSTALL_DIR}/build
                  ${OPENCV_INSTALL_DIR}/modules/core/include/
                  ${OPENCV_INSTALL_DIR}/modules/highgui/include/                  
                  ${OPENCV_INSTALL_DIR}/modules/imgcodecs/include/
                  ${OPENCV_INSTALL_DIR}/modules/videoio/include/
                  ${OPENCV_INSTALL_DIR}/modules/imgproc/include/
                  ${OPENCV_INSTALL_DIR}/cmake/

                  $ENV{TIDL_TOOLS_PATH}
                  PUBLIC ${PROJECT_SOURCE_DIR}/post_process
                  PUBLIC ${PROJECT_SOURCE_DIR}/pre_process
                  PUBLIC ${PROJECT_SOURCE_DIR}/utils
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
                  ${TARGET_FS_PATH}/usr/lib/python3.8/site-packages  

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

if((${TARGET_DEVICE} STREQUAL  "am62a") AND (${TARGET_CPU} STREQUAL  "arm" AND ${HOST_CPU} STREQUAL  "x86") )
  message(STATUS "cross compiling for AM62A")
  add_compile_options(-DDEVICE_AM62A=1)
  #disable xnn tflite 2.8
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
                  #AM62A targetfs
                  ${TARGET_FS_PATH}/usr/lib
                  ${TARGET_FS_PATH}/usr/lib/glib-2.0
                  ${TARGET_FS_PATH}/usr/lib/python3.8/site-packages  

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
                  vx_tidl_rt
                  ti_rpmsg_char
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
                    /usr/lib/python3.8/site-packages/dlr/
                    $ENV{HOME}/.local/dlr/                 
    )
endif()

if((${TARGET_DEVICE} STREQUAL  "am62a") AND (${TARGET_CPU} STREQUAL  "arm" AND ${HOST_CPU} STREQUAL  "arm") )
  message(STATUS "native compiling for AM62A")
  if(NOT FLATBUFFERS_DIR)
    if (EXISTS $ENV{HOME}/flatbuffers)
      set(FLATBUFFERS_DIR $ENV{HOME}/flatbuffers)
    else()
      message(WARNING "FLATBUFFERS_DIR is not set")
    endif()
  endif()
  add_compile_options(-DDEVICE_AM62A=1)
  #disable xnn tflite 2.8
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
              #onnx headers from fs
              /usr/include/onnxruntime/core/session
              #tflite
              ${TENSORFLOW_INSTALL_DIR}/
              ${FLATBUFFERS_DIR}/include

              PUBLIC ${PROJECT_SOURCE_DIR}/post_process
              PUBLIC ${PROJECT_SOURCE_DIR}/pre_process
              PUBLIC ${PROJECT_SOURCE_DIR}/utils
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
                    /usr/lib/python3.8/site-packages/dlr/
                    $ENV{HOME}/.local/dlr/                 
    )
endif()

if((${TARGET_DEVICE} STREQUAL  "j7") AND (${TARGET_CPU} STREQUAL  "arm" AND ${HOST_CPU} STREQUAL  "x86") )
  message(STATUS "cross compiling for J7")
  add_compile_options(-DDEVICE_J7=1)
  add_compile_options(-DXNN_ENABLE=1)

  if(NOT ARM64GCC_PATH)
    if (EXISTS $ENV{HOME}/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu)
      set(ARM64GCC_PATH $ENV{HOME}/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu)
      message (STATUS  "setting ARM64GCC_PATH path:${ARM64GCC_PATH}")
    else()
     message(WARNING "ARM64GCC_PATH is not set")
    endif()
  endif()

  if(NOT TARGET_FS_PATH)
    if (EXISTS $ENV{TARGET_FS_PATH}) 
      set(TARGET_FS_PATH $ENV{TARGET_FS_PATH})
      message (STATUS  "setting TARGET_FS_PATH path:${TARGET_FS_PATH}")
    elseif (EXISTS $ENV{HOME}/targetfs)
      set(TARGET_FS_PATH $ENV{HOME}/targetfs)
      message (STATUS  "setting TARGET_FS_PATH path:${TARGET_FS_PATH}")
    else()
      message(WARNING "TARGET_FS_PATH is not set")
    endif()
  endif()


  set(ARMCC_PREFIX ${ARM64GCC_PATH}/bin/aarch64-none-linux-gnu-)
  set(CMAKE_C_COMPILER ${ARMCC_PREFIX}gcc)
  set(CMAKE_CXX_COMPILER ${ARMCC_PREFIX}g++)

  link_directories(                
                  #J7 targetfs
                  ${TARGET_FS_PATH}/usr/lib
                  ${TARGET_FS_PATH}/usr/lib/glib-2.0
                  ${TARGET_FS_PATH}/usr/lib/python3.8/site-packages  
                  # Enable these when migrating to tflite 2.8
                  ${TARGET_FS_PATH}/usr/lib/tflite_2.8/ruy-build
                  ${TARGET_FS_PATH}/usr/lib/tflite_2.8/xnnpack-build
                  ${TARGET_FS_PATH}/usr/lib/tflite_2.8/pthreadpool
                  ${TARGET_FS_PATH}/usr/lib/tflite_2.8/fft2d-build
                  ${TARGET_FS_PATH}/usr/lib/tflite_2.8/cpuinfo-build
                  ${TARGET_FS_PATH}/usr/lib/tflite_2.8/flatbuffers-build
                  ${TARGET_FS_PATH}/usr/lib/tflite_2.8/clog-build
                  ${TARGET_FS_PATH}/usr/lib/tflite_2.8/farmhash-build
                  # Enable these when migrating to tflite 2.8
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
                  ${TFLITE_2_8_LIBS}
  )
  include_directories(
                  ${PROJECT_SOURCE_DIR}
                  ${PROJECT_SOURCE_DIR}/..
                  ${PROJECT_SOURCE_DIR}/include
                  ${DLR_INSTALL_DIR}/include                
                  ${DLR_INSTALL_DIR}/3rdparty/tvm/3rdparty/dlpack/include
                  ${TARGET_FS_PATH}/usr/include
                  
                  #tesnorflow2.8  and dependencies
                  ${TENSORFLOW_INSTALL_DIR}
                  ${TENSORFLOW_INSTALL_DIR}/../tensorflow
                  ${TENSORFLOW_INSTALL_DIR}/../tensorflow/tensorflow/lite/tools//pip_package/gen/tflite_pip/python3/cmake_build/flatbuffers/include

                  #armnn
                  ${ARMNN_PATH}/delegate/include
                  ${ARMNN_PATH}/include

                  ${ONNXRT_INSTALL_DIR}/include
                  ${ONNXRT_INSTALL_DIR}/include/onnxruntime
                  ${ONNXRT_INSTALL_DIR}/include/onnxruntime/core/session                    
                  ${OPENCV_INSTALL_DIR}/modules/core/include
                  ${OPENCV_INSTALL_DIR}/modules/highgui/include
                  ${OPENCV_INSTALL_DIR}/modules/imgcodecs/include
                  ${OPENCV_INSTALL_DIR}/modules/videoio/include
                  ${OPENCV_INSTALL_DIR}/modules/imgproc/include
                  ${OPENCV_INSTALL_DIR}/cmake
                  ${OPENCV_INSTALL_DIR}/build/
                  
                  PUBLIC ${PROJECT_SOURCE_DIR}/post_process
                  PUBLIC ${PROJECT_SOURCE_DIR}/pre_process
                  PUBLIC ${PROJECT_SOURCE_DIR}/utils
                  
                  #tidl tools lib
                  $ENV{TIDL_TOOLS_PATH}
  )
endif()

if( ((${TARGET_DEVICE} STREQUAL  "j7") AND (${TARGET_CPU} STREQUAL  "arm" AND ${HOST_CPU} STREQUAL  "arm"))  )
  message(NOTICE "native compiling for J7")
  add_compile_options(-DDEVICE_J7=1)
  add_compile_options(-DXNN_ENABLE=1)
  

  set(CMAKE_C_COMPILER gcc)
  set(CMAKE_CXX_COMPILER g++)

  link_directories(  
                  /usr/lib/opencv/ #for container
                  /usr/lib
                  # Enable these when migrating to tflite 2.8
                  /usr/lib/tflite_2.8/ruy-build
                  /usr/lib/tflite_2.8/xnnpack-build
                  /usr/lib/tflite_2.8/pthreadpool
                  /usr/lib/tflite_2.8/fft2d-build
                  /usr/lib/tflite_2.8/cpuinfo-build
                  /usr/lib/tflite_2.8/flatbuffers-build
                  /usr/lib/tflite_2.8/clog-build
                  /usr/lib/tflite_2.8/farmhash-build
                  # Enable these when migrating to tflite 2.8
                  /usr/local/dlr
                  /usr/lib/aarch64-linux-gnu
                  /usr/lib/python3.8/site-packages/dlr/
                  /usr/local/lib/python3.6/dist-packages/dlr/
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
                  ${TFLITE_2_8_LIBS}
  )
  include_directories(
                  /usr/include
                  /usr/local/include
                  /usr/local/dlr
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
                          -Wl,--start-group
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
