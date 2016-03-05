file(GLOB SEGER_UTILS_UNITTEST_SOURCES *.cpp)

add_executable(seger_utils_unittest ${SEGER_UTILS_UNITTEST_SOURCES})

target_link_libraries(seger_utils_unittest ${GTEST_BOTH_LIBRARIES})
target_link_libraries(seger_utils_unittest ${SEGER_UTILS_LIB})

add_test(NAME AllTestsInSegerUtils
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/seger_utils_unittest)

