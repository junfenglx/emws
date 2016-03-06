file(GLOB SEGER_UTILS_UNITTEST_SOURCES *.cpp)

# find_package(GTest REQUIRED)
set(GTEST_INCLUDE_DIRS ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
set(GTEST_BOTH_LIBRARIES gtest gtest_main)
message(STATUS ${GTEST_INCLUDE_DIRS})
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(seger_utils_unittest ${SEGER_UTILS_UNITTEST_SOURCES})

target_link_libraries(seger_utils_unittest ${GTEST_BOTH_LIBRARIES})
message(STATUS ${SEGER_UTILS})

# find_library(SEGER_UTILS_LIB ${SEGER_UTILS} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
target_link_libraries(seger_utils_unittest ${SEGER_UTILS})

add_test(NAME AllTestsInSegerUtils
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/seger_utils_unittest
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/)

# use Catch
message(STATUS ${CATCH_INCLUDE_DIR})
include_directories(${CATCH_INCLUDE_DIR})
add_executable(test_catch_func ./test_catch_func.cpp)
add_test(test_catch_func test_catch_func)

