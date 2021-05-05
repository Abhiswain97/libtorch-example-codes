# CMakeList.txt : CMake project for LibtorchDemo, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project(Mnist)

set(Torch_DIR "/home/abhishek/libtorch/share/cmake/Torch")
set(PYTHON_EXECUTABLE "/home/abhishek/miniconda3/bin/python")

find_package(Torch REQUIRED)

set(CXXFLAGS "-W -Wall -Wextra -Werror -std=c++14")

# Add source to this project's executable.
add_executable(Mnist "Main.cpp" "Main.h")

target_link_libraries(Mnist "${TORCH_LIBRARIES}")
