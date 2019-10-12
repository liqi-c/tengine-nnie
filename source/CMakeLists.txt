cmake_minimum_required(VERSION 3.6)

include_directories(${NNIE_SDK_PATH}/include)
include_directories(${TENGINE_PATH}/include)
include_directories(${TENGINE_NNIE_PATH}/include)

link_directories(${TENGINE_PATH}/lib)
link_directories(${TENGINE_NNIE_PATH}/lib)
link_directories(${NNIE_SDK_PATH}/lib)

include_directories(${OPENCV_PATH}/include)
link_directories(${OPENCV_PATH}/lib)

set(LINK_LIBS opencv_core opencv_highgui opencv_imgproc ${LINK_LIBS})

set(LINK_LIBS tengine nnie VoiceEngine upvqe dnvqe mpi securec)
add_definitions(-rdynamic)
add_definitions(-mfloat-abi=softfp -mfpu=neon-vfpv4 -mcpu=cortex-a53 -mfpu=neon -Wno-ignored-attributes -lpthread)

add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-std=c++11>)

if(ACL_DIR)
	set( ACL_LIB ${ACL_DIR}/build/libarm_compute_core.so ${ACL_DIR}/build/libarm_compute.so )
	link_directories(${ACL_DIR}/build)
	set(LINK_LIBS ${ACL_LIB} ${LINK_LIBS})
endif()

set(CMAKE_EXE_LINKER_FLAGS "-rdynamic -ldl")

project(sample_nnie_plugin)
FILE(GLOB CODE_SRCS sample_nnie_plugin.cpp)
add_executable(sample_nnie_plugin ${CODE_SRCS})

target_link_libraries(sample_nnie_plugin ${LINK_LIBS} )
install (TARGETS sample_nnie_plugin  DESTINATION install) 

#------------------------test_nnie_all-------------------------------------
project(test_nnie_all)
FILE(GLOB CODE_SRCS test_nnie_all.cpp)

#include_directories(${NNIE_SDK_PATH}${PLATFORM}/include)
#include_directories(../nnie/include)
#
#link_directories(${LIB_PATH}/lib)
#link_directories(${NNIE_SDK_PATH}${PLATFORM}/lib)
#link_directories(${PROTOBUF_DIR}/install${PLATFORM}/lib)
#link_directories(${OPENCV_PATH}/install${PLATFORM}/lib)
include_directories(${OPENCV_PATH}/include)
link_directories(${OPENCV_PATH}/lib)

#set(LINK_LIBS tengine nnie nnieplugin VoiceEngine upvqe dnvqe mpi securec protobuf pthread)
set(LINK_LIBS tengine nnie nnieplugin VoiceEngine upvqe dnvqe mpi securec pthread)
set(LINK_LIBS opencv_core opencv_highgui opencv_imgproc ${LINK_LIBS})

add_executable(test_nnie_all ${CODE_SRCS})
add_definitions(-rdynamic)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-std=c++11>)

set(CMAKE_EXE_LINKER_FLAGS "-rdynamic -ldl")

target_link_libraries(test_nnie_all ${LINK_LIBS} )
install (TARGETS test_nnie_all  DESTINATION install)