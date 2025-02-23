cmake_minimum_required(VERSION 3.0)
project(OpenCLSimpleStructure)

# Find Boost package
find_package(Boost REQUIRED)

# Create a shared library target
add_library(kernelCode STATIC)

# Add target specific directories
target_include_directories(
  kernelCode
  PUBLIC ${BOOST_INCLUDE_DIRS}
         /usr/local/cuda-12.3/targets/x86_64-linux/include
         ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add sources to the target created above
target_sources(kernelCode PRIVATE src/elementwise_multiply.cpp)

# Create an executable target
add_executable(${PROJECT_NAME} main.cpp)

# Link the executable target with the shared library target
target_link_libraries(${PROJECT_NAME} OpenCL kernelCode)
