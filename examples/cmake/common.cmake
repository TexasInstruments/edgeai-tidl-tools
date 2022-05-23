include(GNUInstallDirs)

# add_compile_options(-std=c++11)


IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

if(NOT DEFINED CMAKE_SYSTEM_PROCESSOR OR CMAKE_SYSTEM_PROCESSOR STREQUAL "")
  message(WARNING "CMAKE_SYSTEM_PROCESSOR is not defined. Perhaps CMake toolchain is broken")
endif()

message(STATUS "Detected processor: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*")
  if(NOT DEFINED DEVICE)
    set(DEVICE x86_64)
  else()
    set(CROSS_COMPILE true)
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64.*|AARCH64.*|arm64.*|ARM64.*)")
  set(AARCH64 1)
endif()


message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE} PROJECT_NAME = ${PROJECT_NAME}")


SET(CMAKE_FIND_LIBRARY_PREFIXES "" "lib")
SET(CMAKE_FIND_LIBRARY_SUFFIXES ".a" ".lib" ".so")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../lib/${CMAKE_BUILD_TYPE})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../bin/${CMAKE_BUILD_TYPE})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../bin/${CMAKE_BUILD_TYPE})

set(TARGET_PLATFORM     J7)
set(TARGET_CPU          A72)
set(TARGET_OS           LINUX)

if(NOT TENSORFLOW_INSTALL_DIR)
  if (EXISTS $ENV{HOME}/tensorflow)
    set(TENSORFLOW_INSTALL_DIR $ENV{HOME}/tensorflow)
  else()
     message(WARNING "TENSORFLOW_INSTALL_DIR is not set")
  endif()
endif()

if(NOT ONNXRT_INSTALL_DIR)
  if (EXISTS $ENV{HOME}/onnxruntime)
    set(ONNXRT_INSTALL_DIR $ENV{HOME}/onnxruntime)
  else()
     message(WARNING "ONNXRT_INSTALL_DIR is not set")
  endif()
endif()

if(NOT DLR_INSTALL_DIR)
  if (EXISTS $ENV{HOME}/neo-ai-dlr)
    set(DLR_INSTALL_DIR $ENV{HOME}/neo-ai-dlr)
  else()
     message(WARNING "DLR_INSTALL_DIR is not set")
  endif()
endif()

if(NOT OPENCV_INSTALL_DIR)
  if (EXISTS $ENV{HOME}/opencv-4.1.0)
    set(OPENCV_INSTALL_DIR $ENV{HOME}/opencv-4.1.0)
  else()
     message(WARNING "OPENCV_INSTALL_DIR is not set")
  endif()
endif()

add_definitions(
    -DTARGET_CPU=${TARGET_CPU}
    -DTARGET_OS=${TARGET_OS}
)

if(${DEVICE} STREQUAL  "x86_64" )
  message(STATUS "Native compiling for X86")
  add_compile_options(-DDEVICE_X86=1)
  set(CMAKE_C_COMPILER gcc)
  set(CMAKE_CXX_COMPILER g++)

  link_directories(
                  # opencv libraries
                  ${OPENCV_INSTALL_DIR}/cmake/lib
                  ${OPENCV_INSTALL_DIR}/cmake/3rdparty/lib
                  ${OPENCV_INSTALL_DIR}/modules/core/include
                  #common
                  /usr/lib 
                  /usr/local/dlr
                  /usr/lib/aarch64-linux-gnu
                  /usr/lib/python3.8/site-packages/dlr/
                  $ENV{HOME}/.local/dlr/                 
                  #tesnorflow and dependencies
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build/_deps/ruy-build/ruy
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build/pthreadpool
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build/_deps/fft2d-build
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build/_deps/cpuinfo-build
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build/_deps/flatbuffers-build
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build/_deps/clog-build
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build/_deps/farmhash-build
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build/_deps/xnnpack-build

                  #tidl tools lib
                  $ENV{TIDL_TOOLS_PATH}

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
                  
                  #tflite
                  ${TENSORFLOW_INSTALL_DIR}/tensorflow_src
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build/flatbuffers/include
                  #armnn
                  ${ARMNN_PATH}/delegate/include
                  ${ARMNN_PATH}/armnn/include

                  ${ONNXRT_INSTALL_DIR}/include
                  ${ONNXRT_INSTALL_DIR}/include/onnxruntime
                  ${ONNXRT_INSTALL_DIR}/include/onnxruntime/core/session                    
                  ${DLR_INSTALL_DIR}/include
                  ${DLR_INSTALL_DIR}/3rdparty/tvm/3rdparty/dlpack/include
                  ${OPENCV_INSTALL_DIR}/modules/core/include
                  ${OPENCV_INSTALL_DIR}/modules/highgui/include
                  ${OPENCV_INSTALL_DIR}/modules/imgcodecs/include
                  ${OPENCV_INSTALL_DIR}/modules/videoio/include
                  ${OPENCV_INSTALL_DIR}/modules/imgproc/include
                  ${OPENCV_INSTALL_DIR}/cmake
                  $ENV{TIDL_TOOLS_PATH}
                  PUBLIC ${PROJECT_SOURCE_DIR}/post_process
                  PUBLIC ${PROJECT_SOURCE_DIR}/pre_process
                  PUBLIC ${PROJECT_SOURCE_DIR}/utils
  )
endif()
if((${DEVICE} STREQUAL  "am62") AND CROSS_COMPILE )

  message(STATUS "cross compiling for AM62")
  add_compile_options(-DDEVICE_AM62=1)

  if(NOT CROSS_COMPILER_PATH)
    if (EXISTS $ENV{HOME}/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu)
      set(CROSS_COMPILER_PATH $ENV{HOME}/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu)
    else()
     message(WARNING "CROSS_COMPILER_PATH is not set")
    endif()
  endif()

  if(NOT TARGET_FS_PATH)
    if (EXISTS $ENV{HOME}/targetfs)
      set(TARGET_FS_PATH $ENV{HOME}/targetfs)
    else()
      message(WARNING "TARGET_FS_PATH is not set")
    endif()
  endif()

  if(NOT ARMNN_PATH)
    if (EXISTS $ENV{HOME}/armnn)
      set(ARMNN_PATH $ENV{HOME}/armnn)
    else()
      message(WARNING "ARMNN_PATH is not set")
    endif()
  endif()

  set(ARMCC_PREFIX ${CROSS_COMPILER_PATH}/bin/aarch64-none-linux-gnu-)
  set(CMAKE_C_COMPILER ${ARMCC_PREFIX}gcc)
  set(CMAKE_CXX_COMPILER ${ARMCC_PREFIX}g++)

  link_directories(
                  # opencv libraries
                  ${OPENCV_INSTALL_DIR}/cmake_static/lib
                  ${OPENCV_INSTALL_DIR}/cmake_static/3rdparty/lib
                  ${OPENCV_INSTALL_DIR}/cmake_static/3rdparty/libjasper/CMakeFiles/libjasper.dir                  
                  #AM62 targetfs
                  ${TARGET_FS_PATH}/usr/lib
                  ${TARGET_FS_PATH}/usr/lib/glib-2.0
                  ${TARGET_FS_PATH}/usr/lib/python3.8/site-packages  
                  
                  #tesnorflow and dependencies
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build_arm
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build_arm/_deps/ruy-build/ruy
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build_arm/pthreadpool
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build_arm/_deps/fft2d-build
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build_arm/_deps/cpuinfo-build
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build_arm/_deps/flatbuffers-build
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build_arm/_deps/clog-build
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build_arm/_deps/farmhash-build
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build_arm/_deps/xnnpack-build

                  #tidl tools lib
                  $ENV{TIDL_TOOLS_PATH}

                  #armnn lib
                  ${ARMNN_PATH}/build
                  ${ARMNN_PATH}/build/delegate
  )
  set(SYSTEM_LINK_LIBS
                  tensorflow-lite
                  pcre
                  ffi
                  ti_rpmsg_char    
                  z
                  tegra_hal
                  opencv_imgproc
                  opencv_imgcodecs
                  opencv_core
                  tbb
                  libjasper
                  jpeg
                  webp
                  png16
                  tiff
                  onnxruntime
                  dl
                  dlr
                  yaml-cpp
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
                  XNNPACK
                  pthreadpool
                  pthread
                  armnn
                  armnnDelegate
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
                  #armnn
                  ${ARMNN_PATH}/delegate/include
                  ${ARMNN_PATH}/include

                  ${ONNXRT_INSTALL_DIR}/include
                  ${ONNXRT_INSTALL_DIR}/include/onnxruntime
                  ${ONNXRT_INSTALL_DIR}/include/onnxruntime/core/session                    
                  ${DLR_INSTALL_DIR}/include
                  ${DLR_INSTALL_DIR}/3rdparty/tvm/3rdparty/dlpack/include
                  ${OPENCV_INSTALL_DIR}/modules/core/include
                  ${OPENCV_INSTALL_DIR}/modules/highgui/include
                  ${OPENCV_INSTALL_DIR}/modules/imgcodecs/include
                  ${OPENCV_INSTALL_DIR}/modules/videoio/include
                  ${OPENCV_INSTALL_DIR}/modules/imgproc/include
                  ${OPENCV_INSTALL_DIR}/cmake
                  
                  PUBLIC ${PROJECT_SOURCE_DIR}/post_process
                  PUBLIC ${PROJECT_SOURCE_DIR}/pre_process
                  PUBLIC ${PROJECT_SOURCE_DIR}/utils
  )
endif()
if((${DEVICE} STREQUAL  "am62") AND (NOT CROSS_COMPILE) )
  message(STATUS "native compiling for AM62")
  if(NOT ARMNN_PATH)
    if (EXISTS $ENV{HOME}/armnn)
      set(ARMNN_PATH $ENV{HOME}/armnn)
    else()
      message(WARNING "ARMNN_PATH is not set")
    endif()
  endif()

  if(NOT FLATBUFFERS_DIR)
    if (EXISTS $ENV{HOME}/flatbuffers)
      set(FLATBUFFERS_DIR $ENV{HOME}/flatbuffers)
    else()
      message(WARNING "FLATBUFFERS_DIR is not set")
    endif()
  endif()

  add_compile_options(-DDEVICE_AM62=1)

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
              
              #tflite
              ${TENSORFLOW_INSTALL_DIR}/
              ${FLATBUFFERS_DIR}/include
              #armnn
              ${ARMNN_PATH}/delegate/include
              ${ARMNN_PATH}/armnn/include
              ${ARMNN_PATH}/include

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
              # dlr
              tensorflow-lite
              onnxruntime            
              pthread
              dl
              yaml-cpp
              pthreadpool
              XNNPACK
              #armnnn libs
              armnn
              armnnDelegate
  )

  link_directories(
                  #armnn lib need to remove once added to filesystem
                  ${ARMNN_PATH}/build
                  ${ARMNN_PATH}/build/delegate
                  /usr/lib 
                  /usr/local/dlr
                  /usr/lib/aarch64-linux-gnu
                  /usr/lib/python3.8/site-packages/dlr/
                  $ENV{HOME}/.local/dlr/                 
  )
  
  endif()
if((${DEVICE} STREQUAL  "j7") AND CROSS_COMPILE )
  message(STATUS "cross compiling for J7")
  add_compile_options(-DDEVICE_J7=1)
  if(NOT CROSS_COMPILER_PATH)
    if (EXISTS $ENV{HOME}/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu)
      set(CROSS_COMPILER_PATH $ENV{HOME}/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu)
    else()
     message(WARNING "CROSS_COMPILER_PATH is not set")
    endif()
  endif()

  if(NOT TARGET_FS_PATH)
    if (EXISTS $ENV{HOME}/targetfs)
      set(TARGET_FS_PATH $ENV{HOME}/targetfs)
    else()
      message(WARNING "TARGET_FS_PATH is not set")
    endif()
  endif()

  if(NOT ARMNN_PATH)
    if (EXISTS $ENV{HOME}/armnn)
      set(ARMNN_PATH $ENV{HOME}/armnn)
    else()
      message(WARNING "ARMNN_PATH is not set")
    endif()
  endif()

  set(ARMCC_PREFIX ${CROSS_COMPILER_PATH}/bin/aarch64-none-linux-gnu-)
  set(CMAKE_C_COMPILER ${ARMCC_PREFIX}gcc)
  set(CMAKE_CXX_COMPILER ${ARMCC_PREFIX}g++)

  link_directories(
                  # opencv libraries
                  ${OPENCV_INSTALL_DIR}/cmake_static/lib
                  ${OPENCV_INSTALL_DIR}/cmake_static/3rdparty/lib
                  ${OPENCV_INSTALL_DIR}/cmake_static/3rdparty/libjasper/CMakeFiles/libjasper.dir                  
                  #AM62 targetfs
                  ${TARGET_FS_PATH}/usr/lib
                  ${TARGET_FS_PATH}/usr/lib/glib-2.0
                  ${TARGET_FS_PATH}/usr/lib/python3.8/site-packages  
                  
                  #tesnorflow and dependencies
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build_arm
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build_arm/_deps/ruy-build/ruy
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build_arm/pthreadpool
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build_arm/_deps/fft2d-build
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build_arm/_deps/cpuinfo-build
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build_arm/_deps/flatbuffers-build
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build_arm/_deps/clog-build
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build_arm/_deps/farmhash-build
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build_arm/_deps/xnnpack-build

                  #tidl tools lib
                  $ENV{TIDL_TOOLS_PATH}

                  #armnn lib
                  ${ARMNN_PATH}/build
                  ${ARMNN_PATH}/build/delegate
  )
  set(SYSTEM_LINK_LIBS
                  tensorflow-lite
                  pcre
                  ffi
                  ti_rpmsg_char    
                  z
                  tegra_hal
                  opencv_imgproc
                  opencv_imgcodecs
                  opencv_core
                  tbb
                  libjasper
                  jpeg
                  webp
                  png16
                  tiff
                  onnxruntime
                  dl
                  dlr
                  yaml-cpp
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
                  XNNPACK
                  pthreadpool
                  pthread
                  armnn
                  armnnDelegate
                  vx_tidl_rt
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
                  #armnn
                  ${ARMNN_PATH}/delegate/include
                  ${ARMNN_PATH}/include

                  ${ONNXRT_INSTALL_DIR}/include
                  ${ONNXRT_INSTALL_DIR}/include/onnxruntime
                  ${ONNXRT_INSTALL_DIR}/include/onnxruntime/core/session                    
                  ${DLR_INSTALL_DIR}/include
                  ${DLR_INSTALL_DIR}/3rdparty/tvm/3rdparty/dlpack/include
                  ${OPENCV_INSTALL_DIR}/modules/core/include
                  ${OPENCV_INSTALL_DIR}/modules/highgui/include
                  ${OPENCV_INSTALL_DIR}/modules/imgcodecs/include
                  ${OPENCV_INSTALL_DIR}/modules/videoio/include
                  ${OPENCV_INSTALL_DIR}/modules/imgproc/include
                  ${OPENCV_INSTALL_DIR}/cmake
                  
                  PUBLIC ${PROJECT_SOURCE_DIR}/post_process
                  PUBLIC ${PROJECT_SOURCE_DIR}/pre_process
                  PUBLIC ${PROJECT_SOURCE_DIR}/utils
                  
                  #tidl tools lib
                  $ENV{TIDL_TOOLS_PATH}
  )
endif()
if( ((${DEVICE} STREQUAL  "j7") AND (NOT CROSS_COMPILE ))  )
  message(NOTICE "native compiling for J7")
  add_compile_options(-DDEVICE_J7=1)
  
  if(NOT ARMNN_PATH)
    if (EXISTS $ENV{HOME}/armnn)
      set(ARMNN_PATH $ENV{HOME}/armnn)
    else()
      message(WARNING "ARMNN_PATH is not set")
    endif()
  endif()

  set(CMAKE_C_COMPILER gcc)
  set(CMAKE_CXX_COMPILER g++)

  link_directories(
                  #tesnorflow and dependencies need to remove once 2.8 is added
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build_arm
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build_arm/_deps/ruy-build/ruy
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build_arm/pthreadpool
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build_arm/_deps/fft2d-build
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build_arm/_deps/cpuinfo-build
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build_arm/_deps/flatbuffers-build
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build_arm/_deps/clog-build
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build_arm/_deps/farmhash-build
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build_arm/_deps/xnnpack-build

                  #armnn lib need to remove once added to filesystem
                  ${ARMNN_PATH}/build
                  ${ARMNN_PATH}/build/delegate

                  /usr/lib 
                  /usr/local/dlr
                  /usr/lib/aarch64-linux-gnu
                  /usr/lib/python3.8/site-packages/dlr/
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
                  onnxruntime
                  dl
                  dlr
                  yaml-cpp
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
                  XNNPACK
                  pthreadpool
                  pthread
                  armnn
                  armnnDelegate
                  vx_tidl_rt
  )
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
                  
                  #tflite
                  ${TENSORFLOW_INSTALL_DIR}/tensorflow_src
                  ${TENSORFLOW_INSTALL_DIR}/tflite_build/flatbuffers/include
                  #armnn
                  ${ARMNN_PATH}/delegate/include
                  ${ARMNN_PATH}/armnn/include
                  ${ARMNN_PATH}/include

                  ${ONNXRT_INSTALL_DIR}/include
                  ${ONNXRT_INSTALL_DIR}/include/onnxruntime
                  ${ONNXRT_INSTALL_DIR}/include/onnxruntime/core/session                    
                  ${DLR_INSTALL_DIR}/include
                  ${DLR_INSTALL_DIR}/3rdparty/tvm/3rdparty/dlpack/include
                  #opencv
                  ${OPENCV_INSTALL_DIR}/modules/core/include
                  ${OPENCV_INSTALL_DIR}/modules/highgui/include
                  ${OPENCV_INSTALL_DIR}/modules/imgcodecs/include
                  ${OPENCV_INSTALL_DIR}/modules/videoio/include
                  ${OPENCV_INSTALL_DIR}/modules/imgproc/include
                  ${OPENCV_INSTALL_DIR}/cmake
                  $ENV{TIDL_TOOLS_PATH}
                  PUBLIC ${PROJECT_SOURCE_DIR}/post_process
                  PUBLIC ${PROJECT_SOURCE_DIR}/pre_process
                  PUBLIC ${PROJECT_SOURCE_DIR}/utils
  )
endif()



if (EXISTS $ENV{CONDA_PREFIX}/dlr)
link_directories(
                 $ENV{CONDA_PREFIX}/dlr
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
    
    if(NOT ${DEVICE} STREQUAL  "am62")
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
