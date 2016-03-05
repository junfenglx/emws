file(GLOB SEGER_UTILS_UNITTEST_SOURCES *.cpp)

add_executable(seger_utils_unittest ${SEGER_UTILS_UNITTEST_SOURCES})

target_link_libraries(seger_utils_unittest ${GTEST_BOTH_LIBRARIES})
message(STATUS ${SEGER_UTILS})

# find_library(SEGER_UTILS_LIB ${SEGER_UTILS} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
target_link_libraries(seger_utils_unittest ${SEGER_UTILS})

add_test(NAME AllTestsInSegerUtils
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/seger_utils_unittest
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/)

