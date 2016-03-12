# file(GLOB SEGER_UTILS_UNITTEST_SOURCES *.cpp)
set(SEGER_UTILS_UNITTEST_SOURCES test_utf8_io.cpp test_str_op.cpp test_emws_seger.cpp)

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

# score_ret test
add_executable( score_ret_unittest test_score_ret.cpp ../src/score_ret.cpp)
target_link_libraries(score_ret_unittest ${GTEST_BOTH_LIBRARIES})
target_link_libraries(score_ret_unittest ${SEGER_UTILS})

add_test(NAME AllTestsInScoreRet
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/score_ret_unittest
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/)

# emws_seger test
include_directories(${Cereal_INCLUDE_DIR})
add_executable( emws_seger_unittest test_emws_seger.cpp ../src/emws_seger.cpp  ../src/base_seger.cpp ../src/score_ret.cpp)
target_link_libraries(emws_seger_unittest ${GTEST_BOTH_LIBRARIES})
target_link_libraries(emws_seger_unittest ${SEGER_UTILS} ${ARMADILLO_LIBRARIES})

add_test(NAME AllTestsInEmwsSeger
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/emws_seger_unittest
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/)

# use Catch
message(STATUS ${CATCH_INCLUDE_DIR})
include_directories(${CATCH_INCLUDE_DIR})
add_executable(test_catch_func ./test_catch_func.cpp)
add_test(test_catch_func test_catch_func)

