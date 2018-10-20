cmake_minimum_required(VERSION 2.8)
project(PhotoMosaic)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(PhotoMosaic PhotoMosaic.cpp)
target_link_libraries(PhotoMosaic ${OpenCV_LIBS} Threads::Threads)
add_executable(Preprocess Preprocess.cpp)
target_link_libraries(Preprocess ${OpenCV_LIBS})
