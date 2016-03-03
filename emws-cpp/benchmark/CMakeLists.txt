project(matrix_benchmark)
file(GLOB BENCH_SRCS *.cpp)

include_directories("${Eigen_INCLUDE_DIR}")
include_directories("${Eigen_Unsupported_INCLUDE_DIR}")
include_directories("${Libigl_INCLUDE_DIR}")

add_executable(${PROJECT_NAME} ${BENCH_SRCS})

# boost program_options
find_package(Boost REQUIRED COMPONENTS program_options)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
endif()

target_link_libraries(${PROJECT_NAME} ${ARMADILLO_LIBRARIES})