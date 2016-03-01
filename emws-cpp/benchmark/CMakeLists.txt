project(matrix_benchmark)
file(GLOB BENCH_SRCS *.cpp)

add_executable(${PROJECT_NAME} ${BENCH_SRCS})

# boost program_options
find_package(Boost REQUIRED COMPONENTS program_options)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
endif()

target_link_libraries(${PROJECT_NAME} ${ARMADILLO_LIBRARIES})