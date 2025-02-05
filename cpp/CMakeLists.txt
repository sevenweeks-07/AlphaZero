cmake_minimum_required(VERSION 3.1)
project(main)

set(Torch_DIR "/mnt/c/libtorch/share/cmake/Torch")

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage -g")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")

# Find and include Torch
find_package(Torch REQUIRED)
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

# Set the C++ standard property to C++17 (not 14)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
