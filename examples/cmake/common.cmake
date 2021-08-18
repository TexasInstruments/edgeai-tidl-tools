include(GNUInstallDirs)

add_compile_options(-std=c++11)

# Specific compile optios across all targets
#add_compile_definitions(MINIMAL_LOGGING)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE} PROJECT_NAME = ${PROJECT_NAME}")

set(CMAKE_C_COMPILER gcc-5)
set(CMAKE_CXX_COMPILER g++-5)

SET(CMAKE_FIND_LIBRARY_PREFIXES "" "lib")
SET(CMAKE_FIND_LIBRARY_SUFFIXES ".a" ".lib" ".so")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../lib/${CMAKE_BUILD_TYPE})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../bin/${CMAKE_BUILD_TYPE})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../bin/${CMAKE_BUILD_TYPE})

set(TARGET_PLATFORM     J7)
set(TARGET_CPU          A72)
set(TARGET_OS           LINUX)

if(NOT TENSORFLOW_INSTALL_DIR)
  if (EXISTS /home/root/tensorflow)
    set(TENSORFLOW_INSTALL_DIR /home/root/tensorflow)
  else()
     message(WARNING "TENSORFLOW_INSTALL_DIR is not set")
  endif()
endif()

if(NOT ONNXRT_INSTALL_DIR)
  if (EXISTS /home/root/onnxruntime)
    set(ONNXRT_INSTALL_DIR /home/root/onnxruntime)
  else()
     message(WARNING "ONNXRT_INSTALL_DIR is not set")
  endif()
endif()

if(NOT DLR_INSTALL_DIR)
  if (EXISTS /home/root/neo-ai-dlr)
    set(DLR_INSTALL_DIR /home/root/neo-ai-dlr)
  else()
     message(WARNING "DLR_INSTALL_DIR is not set")
  endif()
endif()

if(NOT OPENCV_INSTALL_DIR)
  if (EXISTS /home/root/opencv-4.1.0)
    set(OPENCV_INSTALL_DIR /home/root/opencv-4.1.0)
  else()
     message(WARNING "OPENCV_INSTALL_DIR is not set")
  endif()
endif()

add_definitions(
    -DTARGET_CPU=${TARGET_CPU}
    -DTARGET_OS=${TARGET_OS}
)

link_directories(/usr/local/dlr
                 /usr/lib/aarch64-linux-gnu
                 /usr/lib/
                 ${OPENCV_INSTALL_DIR}/cmake/lib
                 ${OPENCV_INSTALL_DIR}/cmake/3rdparty/lib
                 $ENV{TIDL_TOOLS_PATH}
                 )

#message("PROJECT_SOURCE_DIR =" ${PROJECT_SOURCE_DIR})
#message("CMAKE_SOURCE_DIR =" ${CMAKE_SOURCE_DIR})

include_directories(${PROJECT_SOURCE_DIR}
                    ${PROJECT_SOURCE_DIR}/..
                    ${PROJECT_SOURCE_DIR}/include
                    /usr/local/include
                    /usr/local/dlr
                    /usr/include/gstreamer-1.0/
                    /usr/include/glib-2.0/
                    /usr/lib/aarch64-linux-gnu/glib-2.0/include
                    /usr/include/opencv4/
                    /usr/include/processor_sdk/vision_apps/
                    ${TENSORFLOW_INSTALL_DIR}
                    ${TENSORFLOW_INSTALL_DIR}/tensorflow/lite/tools/make/downloads/flatbuffers/include
                    ${ONNXRT_INSTALL_DIR}/include
                    ${ONNXRT_INSTALL_DIR}/include/onnxruntime
                    ${ONNXRT_INSTALL_DIR}/include/onnxruntime/core/session                    
                    ${DLR_INSTALL_DIR}/include
                    ${OPENCV_INSTALL_DIR}/modules/core/include
                    ${OPENCV_INSTALL_DIR}/modules/highgui/include
                    ${OPENCV_INSTALL_DIR}/modules/imgcodecs/include
                    ${OPENCV_INSTALL_DIR}/modules/videoio/include
                    ${OPENCV_INSTALL_DIR}/modules/imgproc/include
                    ${OPENCV_INSTALL_DIR}/cmake
                    $ENV{TIDL_TOOLS_PATH}
                    )

#set(COMMON_LINK_LIBS
#    utils
#    common
#    )

set(SYSTEM_LINK_LIBS
    glib-2.0
    gobject-2.0
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
    )


# Function for building a node:
# ARG0: app name
# ARG1: source list
function(build_app)
    set(app ${ARGV0})
    set(src ${ARGV1})
    add_executable(${app} ${${src}})
    target_link_libraries(${app}
                          -Wl,--start-group
                          ${COMMON_LINK_LIBS}
                          ${TARGET_LINK_LIBS}
                          ${SYSTEM_LINK_LIBS}
                          -Wl,--end-group)
endfunction()
