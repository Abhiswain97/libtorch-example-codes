cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(Mnist)

set(Torch_DIR "/home/abhishek/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(Mnist "Mnist.cpp" "MnistSimple.cpp")
target_link_libraries(Mnist "${TORCH_LIBRARIES}")
set_property(TARGET Mnist PROPERTY CXX_STANDARD 14)

