cmake_minimum_required(VERSION 2.8)

project(SuperResolution)

find_package(OpenCV REQUIRED)
#find_package(gflags REQUIRED)
#find_package(glog REQUIRED)
#find_package(protobuf REQUIRED)
#find_package(gtest REQUIRED)
#find_package(gmock REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Default to Release mode.
IF(NOT DEFINED CMAKE_BUILD_TYPE)
  SET(${CMAKE_BUILD_TYPE} Release ... FORCE)
ENDIF()

# Release mode makes the code run faster.
IF(NOT ${CMAKE_BUILD_TYPE} MATCHES Debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -Ofast")
ENDIF()

# Require C++ 11.
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Libraries will be stored in the "lib" directory, and binaries in "bin".
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Grab all files in the source directory.
file(GLOB SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*")


# Create a library for all of the added Super Resolution dependencies (i.e. all
# the classes).
file(GLOB data_generator_SRC "src/data_generator/*.cpp")
file(GLOB util_SRC "src/util/*.cpp")
file(GLOB video_SRC "src/video/*.cpp")

add_library(
  LibSuperResolution
  ${data_generator_SRC}
  ${util_SRC}
  ${video_SRC}
)


# Add the GenerateData binary.
add_executable(
  GenerateData
  src/generate_data.cpp
)
target_link_libraries(
  GenerateData
  LibSuperResolution
  glog
  gflags
  ${OpenCV_LIBS}
)

# Add the Shift-Add Fusion binary.
add_executable(
  ShiftAddFusion
  src/shift_add_fusion.cpp
)
target_link_libraries(
  ShiftAddFusion
  LibSuperResolution
  glog
  gflags
  ${OpenCV_LIBS}
)

# Add the SuperResolution binary.
add_executable(
  SuperResolution
  src/super_resolution.cpp
)
target_link_libraries(
  SuperResolution
  LibSuperResolution
  glog
  gflags
  ${OpenCV_LIBS}
)
