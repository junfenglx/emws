project(emws-cpp)

file(GLOB SEGER_SRCS *_seger.cpp)

set(TRAIN_SRCS emws-train.cpp score_ret.cpp)
set(PREDICT_SRCS emws-predict.cpp score_ret.cpp)

set(TRAIN_EXEC emws-train)
set(PREDICT_EXEC emws-predict)

add_executable(${TRAIN_EXEC} ${TRAIN_SRCS} ${SEGER_SRCS})
add_executable(${PREDICT_EXEC} ${PREDICT_SRCS} ${SEGER_SRCS})

# boost program_options
find_package(Boost REQUIRED COMPONENTS program_options)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(${TRAIN_EXEC} ${Boost_LIBRARIES})
    target_link_libraries(${PREDICT_EXEC} ${Boost_LIBRARIES})
endif()

target_link_libraries(${TRAIN_EXEC} ${ARMADILLO_LIBRARIES})
target_link_libraries(${PREDICT_EXEC} ${ARMADILLO_LIBRARIES})


set(STR_OP_SRCS str_op.cpp)
set(UTF8_IO_SRCS utf8_io.cpp)

set(SEGER_UTILS seger_utils CACHE INTERNAL "seger_utils library name" FORCE)
add_library(${SEGER_UTILS} SHARED ${STR_OP_SRCS} ${UTF8_IO_SRCS})

target_link_libraries(${TRAIN_EXEC} ${SEGER_UTILS} ${ARMADILLO_LIBRARIES})
target_link_libraries(${PREDICT_EXEC} ${SEGER_UTILS} ${ARMADILLO_LIBRARIES})