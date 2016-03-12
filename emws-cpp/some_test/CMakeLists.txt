project(some_test)

include_directories("${Eigen_INCLUDE_DIR}")
include_directories("${Eigen_Unsupported_INCLUDE_DIR}")
include_directories("${Libigl_INCLUDE_DIR}")

add_executable(test_eigen_dot test_eigen_dot.cc)

add_executable(test_first_eigen test_first_eigen.cc)

add_executable(test_armadillo_example ./test_armadillo_example.cpp)
target_link_libraries( test_armadillo_example ${ARMADILLO_LIBRARIES})

add_executable(test_libigl_hello ./test_libigl_hello.cpp)


add_executable(test_armadillo_solve ./test_armadillo_solve.cpp)
target_link_libraries( test_armadillo_solve ${ARMADILLO_LIBRARIES})

add_executable(test_easylogging ./test_easylogging.cpp)

add_executable(test_rapidjson_pretty ./test_rapidjson_pretty.cpp)

add_executable(test_sigaction ./test_sigaction.c)

add_executable(test_utf8_io ./test_utf8_io.cpp)
target_link_libraries(test_utf8_io ${SEGER_UTILS})

add_executable(test_str_op ./test_str_op.cpp)
target_link_libraries(test_str_op ${SEGER_UTILS})

add_executable(test_locale ./test_locale.cpp)

include_directories(${Cereal_INCLUDE_DIR})

add_executable(test_cereal_u32 ./test_cereal_u32.cpp)