cmake_minimum_required(VERSION 3.0)
project(OpenClSort)

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS}
                    /usr/local/cuda-12.3/targets/x86_64-linux/include)

add_executable(OpenClSort compute_sort.cpp)
target_link_libraries(OpenClSort 
                    ${Boost_LIBRARIES}
                    OpenCL)