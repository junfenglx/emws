cmake_minimum_required(VERSION 3.3)
project(emws-cpp)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# You can tweak some common (for all subprojects) stuff here. For example:

set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES  ON)

if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(SEND_ERROR "In-source builds are not allowed.")
endif ()

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_COLOR_MAKEFILE   ON)

# common include
set(Eigen_INCLUDE_DIR /usr/include/eigen3)
include_directories("${Eigen_INCLUDE_DIR}")

set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
include_directories("${PROJECT_INCLUDE_DIR}")

# include_directories("${PROJECT_SOURCE_DIR}/src")

# sub project
add_subdirectory(benchmark/)

add_subdirectory(some_test/)

add_subdirectory(src/)