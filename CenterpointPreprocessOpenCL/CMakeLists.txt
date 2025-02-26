cmake_minimum_required(VERSION 3.0)
project(centerpointPreprocessOpenCL)

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS}
                    /usr/local/cuda-12.3/targets/x86_64-linux/include)

add_executable(centerpointPreprocessOpenCL main.cpp)
target_link_libraries(centerpointPreprocessOpenCL 
                    ${Boost_LIBRARIES}
                    OpenCL)