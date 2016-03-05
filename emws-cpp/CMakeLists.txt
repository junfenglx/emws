cmake_minimum_required(VERSION 3.3)
project(zh-seger)

# output dir
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

# compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# openmp
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# You can tweak some common (for all subprojects) stuff here. For example:

set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES  ON)

if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(SEND_ERROR "In-source builds are not allowed.")
endif ()

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_COLOR_MAKEFILE   ON)

# test option
option(ZH_SEGER_BUILD_TESTS "Build zh-seger tests." ON)

# common include

# Eigen
set(Eigen_INCLUDE_DIR /usr/include/eigen3)
set(Eigen_Unsupported_INCLUDE_DIR /usr/include/eigen3/unsupported)

# libigl
set(Libigl_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/third_party/libigl/include)

# Armadillo
find_package(Armadillo)
include_directories(${ARMADILLO_INCLUDE_DIRS})
# target_link_libraries(foo ${ARMADILLO_LIBRARIES})
# MESSAGE(STATUS "${ARMADILLO_LIBRARIES}")

# Easylogging++
set(Easyloggingpp_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/third_party/easyloggingpp/src)
include_directories(${Easyloggingpp_INCLUDE_DIR})

# RapidJSON
set(RapidJSON_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/third_party/rapidjson/include)
include_directories(${RapidJSON_INCLUDE_DIR})

# project include dir
set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/src)
include_directories("${PROJECT_INCLUDE_DIR}")

# include_directories("${PROJECT_SOURCE_DIR}/src")

# sub project
add_subdirectory(benchmark/)

add_subdirectory(some_test/)

add_subdirectory(src/)

if(ZH_SEGER_BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    include_directories(${GTEST_INCLUDE_DIRS})
    add_subdirectory(test/)
endif()