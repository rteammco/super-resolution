cmake_minimum_required(VERSION 2.8)

project(SuperResolution)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs)

# Default to Release mode.
IF(NOT DEFINED CMAKE_BUILD_TYPE)
  SET(${CMAKE_BUILD_TYPE} Release ... FORCE)
ENDIF()

# Release mode makes the code run faster.
IF(NOT ${CMAKE_BUILD_TYPE} MATCHES Debug)
  MESSAGE("Build type: Release.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -Ofast")
ELSE()
  MESSAGE("WARNING: Building in DEBUG MODE! Use 'cmake -DCMAKE_BUILD_TYPE=Release' for faster runtime.")
ENDIF()

# Require C++ 11.
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_definitions(-DROOT_CODE_DIRECTORY="${CMAKE_CURRENT_SOURCE_DIR}")

# Libraries will be stored in the "lib" directory, and binaries in "bin".
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Grab all binary source files in the src directory.
file(GLOB SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*")


# Create a library for all of the added Super Resolution dependencies (i.e. all
# the classes).
file(GLOB   alglib_SRC          "libs/alglib/src/*.cpp")
file(GLOB   hyperspectral_SRC   "src/hyperspectral/*.cpp")
file(GLOB   evaluation_SRC      "src/evaluation/*.cpp")
file(GLOB   image_SRC           "src/image/*.cpp")
file(GLOB   image_model_SRC     "src/image_model/*.cpp")
file(GLOB   motion_SRC          "src/motion/*.cpp")
file(GLOB   optimization_SRC    "src/optimization/*.cpp")
file(GLOB   util_SRC            "src/util/*.cpp")
file(GLOB   video_SRC           "src/video/*.cpp")
file(GLOB   wavelet_SRC         "src/wavelet/*.cpp")

add_library(
  LibSuperResolution
  ${alglib_SRC}
  ${hyperspectral_SRC}
  ${evaluation_SRC}
  ${image_SRC}
  ${image_model_SRC}
  ${motion_SRC}
  ${optimization_SRC}
  ${util_SRC}
  ${video_SRC}
  ${wavelet_SRC}
)


# Set up the gtest/gmock testing framework.
enable_testing()
file(GLOB test_SRC "test/*.cpp")
add_executable(
  Test
  ${test_SRC}
)
target_link_libraries(
  Test
  LibSuperResolution
  glog
  gflags
  gmock
  gmock_main
  gtest
  gtest_main
  ${OpenCV_LIBS}
)


# Add the VisualizeImage binary.
add_executable(
  VisualizeImage
  src/visualize_image.cpp
)
target_link_libraries(
  VisualizeImage
  LibSuperResolution
  glog
  gflags
  ${OpenCV_LIBS}
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
