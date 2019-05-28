cmake_minimum_required(VERSION 3.12)
project(panorama_stitching)

set(CMAKE_CXX_STANDARD 14)
find_package( OpenCV REQUIRED )
set(SOURCE_FILES main.cpp)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(panorama_stitching main.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( panorama_stitching ${OpenCV_LIBS} )