cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(torch-demo)
include(FetchContent)
include(GoogleTest)

set(CMAKE_C_COMPILER /usr/bin/clang-13)
set(CMAKE_CXX_COMPILER /usr/bin/clang++-13)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(
  libtorch
  URL https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.10.1%2Bcpu.zip
)

FetchContent_Declare(
  dataset
  URL https://github.com/Rinnekioku/datasets/raw/master/geometric-shapes/validate.tar.gz
  SOURCE_DIR data/geometric-shapes/validate/
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}
)

FetchContent_MakeAvailable(libtorch dataset)

list(APPEND CMAKE_PREFIX_PATH ${libtorch_SOURCE_DIR})
find_package(Torch REQUIRED)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_subdirectory(${CMAKE_SOURCE_DIR}/modules/classification)

set(TORCH_DEMO_LIBS
  classification-module
  ${TORCH_LIBRARIES}
  ${CONAN_LIBS}
)

enable_testing()

add_executable(${PROJECT_NAME} 
  ${CMAKE_CURRENT_SOURCE_DIR}/torch-demo.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC ${TORCH_DEMO_LIBS})

add_dependencies(${PROJECT_NAME} run-classification-module-tests)
