cmake_minimum_required(VERSION 3.16)
project(autograd_engine_v2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

find_package(Eigen3 REQUIRED CONFIG)

add_executable(autograd_engine main.cpp)
target_link_libraries(autograd_engine PRIVATE Eigen3::Eigen)
target_compile_options(autograd_engine PRIVATE -g -Wall -Wextra)

set_target_properties(autograd_engine PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)