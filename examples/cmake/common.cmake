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
     message(WARNING "ONNXRT_INSTALL_DIR is not set(: ignore the warning if device is am62)")
  endif()
endif()

if(NOT DLR_INSTALL_DIR)
  if (EXISTS $ENV{HOME}/neo-ai-dlr)
    set(DLR_INSTALL_DIR $ENV{HOME}/neo-ai-dlr)
  else()
     message(WARNING "DLR_INSTALL_DIR is not set (: ignore the warning if device is am62)")
  endif()
endif()

if (${HOST_CPU} STREQUAL  "x86")
  if(NOT OPENCV_INSTALL_DIR)
    if (EXISTS $ENV{HOME}/opencv-4.1.0)
      set(OPENCV_INSTALL_DIR $ENV{HOME}/opencv-4.1.0)
    else()
      message(WARNING "OPENCV_INSTALL_DIR is not set")
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

if(${TARGET_DEVICE} STREQUAL  "am62" AND  (${TARGET_CPU} STREQUAL  "x86" AND ${HOST_CPU} STREQUAL  "x86"))
  message(STATUS "Compiling for x86 with am62 config")
  add_compile_options(-DDEVICE_AM62=1)
  set(CMAKE_C_COMPILER gcc)
  set(CMAKE_CXX_COMPILER g++)
  add_compile_options(-DDEVICE_AM62=1)
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
  #disable xnn tflite 2.4
  add_compile_options(-DXNN_ENABLE=0)
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
                  #tesnorflow2.4  and dependencies
                  ${TENSORFLOW_INSTALL_DIR}
                  ${TENSORFLOW_INSTALL_DIR}/tensorflow/lite/tools/make/downloads/flatbuffers/include

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
                  
                  #tflite 2.4
                  ${TENSORFLOW_INSTALL_DIR}
                  ${TENSORFLOW_INSTALL_DIR}/tensorflow/lite/tools/make/downloads/flatbuffers/include
                  ${TENSORFLOW_INSTALL_DIR}/tensorflow/lite/tools/make/downloads/flatbuffers/flatbuffers-1.12.0/include
                  ${TENSORFLOW_INSTALL_DIR}/tensorflow/lite/tools/make/downloads/flatbuffers-1.12.0/include

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

if((${TARGET_DEVICE} STREQUAL  "am62") AND (${TARGET_CPU} STREQUAL  "arm" AND ${HOST_CPU} STREQUAL  "x86") )
  message(STATUS "cross compiling for AM62")
  add_compile_options(-DDEVICE_AM62=1)
  #disable xnn tflite 2.8
  add_compile_options(-DXNN_ENABLE=1)

  if(NOT ARM64_GCC_PATH)
    if (EXISTS $ENV{HOME}/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu)
      set(ARM64_GCC_PATH $ENV{HOME}/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu)
    elseif(DEFINED  ENV{ARM64_GCC_PATH})
      set(ARM64_GCC_PATH $ENV{ARM64_GCC_PATH})
    else()
     message(WARNING "ARM64_GCC_PATH is not set")
    endif()
  endif()

  if(NOT TARGET_FS_PATH)
    if (EXISTS $ENV{HOME}/targetfs)
      set(TARGET_FS_PATH $ENV{HOME}/targetfs)
    else()
      message(WARNING "TARGET_FS_PATH is not set")
    endif()
  endif()

  set(ARMCC_PREFIX ${ARM64_GCC_PATH}/bin/aarch64-none-linux-gnu-)
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
  #disable xnn tflite 2.4
  add_compile_options(-DXNN_ENABLE=0)
  if(NOT ARM64_GCC_PATH)
    if (EXISTS $ENV{HOME}/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu)
      set(ARM64_GCC_PATH $ENV{HOME}/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu)
    elseif(DEFINED  ENV{ARM64_GCC_PATH})
      set(ARM64_GCC_PATH $ENV{ARM64_GCC_PATH})
    else()
     message(WARNING "ARM64_GCC_PATH is not set")
    endif()
  endif()

  if(NOT TARGET_FS_PATH)
    if (EXISTS $ENV{HOME}/targetfs)
      set(TARGET_FS_PATH $ENV{HOME}/targetfs)
    else()
      message(WARNING "TARGET_FS_PATH is not set")
    endif()
  endif()


  set(ARMCC_PREFIX ${ARM64_GCC_PATH}/bin/aarch64-none-linux-gnu-)
  set(CMAKE_C_COMPILER ${ARMCC_PREFIX}gcc)
  set(CMAKE_CXX_COMPILER ${ARMCC_PREFIX}g++)

  link_directories(                
                  #J7 targetfs
                  ${TARGET_FS_PATH}/usr/lib
                  ${TARGET_FS_PATH}/usr/lib/glib-2.0
                  ${TARGET_FS_PATH}/usr/lib/python3.8/site-packages  

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
                  pthread
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
                  
                  #tesnorflow2.4  and dependencies
                  ${TENSORFLOW_INSTALL_DIR}
                  ${TENSORFLOW_INSTALL_DIR}/tensorflow/lite/tools/make/downloads/flatbuffers/include

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

if( ((${TARGET_DEVICE} STREQUAL  "j7") AND (${TARGET_CPU} STREQUAL  "arm" AND ${HOST_CPU} STREQUAL  "arm"))  )
  message(NOTICE "native compiling for J7")
  add_compile_options(-DDEVICE_J7=1)
  #disable xnn tflite 2.4
  add_compile_options(-DXNN_ENABLE=0)
  

  set(CMAKE_C_COMPILER gcc)
  set(CMAKE_CXX_COMPILER g++)

  link_directories(
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
                  
                  #tesnorflow2.4  and dependencies
                  ${TENSORFLOW_INSTALL_DIR}
                  ${TENSORFLOW_INSTALL_DIR}/tensorflow/lite/tools/make/downloads/flatbuffers/include

                  ${ONNXRT_INSTALL_DIR}/include
                  ${ONNXRT_INSTALL_DIR}/include/onnxruntime
                  ${ONNXRT_INSTALL_DIR}/include/onnxruntime/core/session                    
                  ${DLR_INSTALL_DIR}/include
                  ${DLR_INSTALL_DIR}/3rdparty/tvm/3rdparty/dlpack/include
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
    /home/root
    ${ARMNN_PATH}/build
    ${ARMNN_PATH}/build/delegate
    $ENV{TIDL_TARGET_LIBS}
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
