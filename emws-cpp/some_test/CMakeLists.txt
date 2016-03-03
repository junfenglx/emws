project(some_test)

add_executable(test_eigen_dot test_eigen_dot.cc)

add_executable(test_first_eigen test_first_eigen.cc)

add_executable(test_armadillo_example ./test_armadillo_example.cpp)
target_link_libraries( test_armadillo_example ${ARMADILLO_LIBRARIES})

add_executable(test_libigl_hello ./test_libigl_hello.cpp)


add_executable(test_armadillo_solve ./test_armadillo_solve.cpp)
target_link_libraries( test_armadillo_solve ${ARMADILLO_LIBRARIES})

add_executable(test_easylogging ./test_easylogging.cpp)

add_executable(test_rapidjson_pretty ./test_rapidjson_pretty.cpp)