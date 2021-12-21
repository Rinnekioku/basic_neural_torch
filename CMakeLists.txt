
cmake_minimum_required(VERSION 3.14)
project(torch-demo)
include(FetchContent)

set(CMAKE_C_COMPILER /usr/bin/clang-12)
set(CMAKE_CXX_COMPILER /usr/bin/clang++-12)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -D_GLIBCXX_USE_CXX11_ABI=0")
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(
  libtorch
  URL https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.10.1%2Bcpu.zip
)

FetchContent_MakeAvailable(libtorch)

list(APPEND CMAKE_PREFIX_PATH ${libtorch_SOURCE_DIR})
find_package(Torch REQUIRED)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(torch-demo torch-demo.cpp)
target_link_libraries(torch-demo PRIVATE ${CONAN_LIBS} ${TORCH_LIBRARIES})
