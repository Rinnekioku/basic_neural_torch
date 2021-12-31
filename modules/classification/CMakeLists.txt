cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(classification-module VERSION 1.0.0)
include(GoogleTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CLASSIFICATION_SOURCES
    # this module currently unavailable
    ${CMAKE_CURRENT_SOURCE_DIR}/src/classifier.cpp  
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dataset.cpp  
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model.cpp
)
set(CLASSIFICATION_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set(CLASSIFICATION_TESTS
    ${PROJECT_NAME}-test
)


add_library(${PROJECT_NAME}
    STATIC
    ${CLASSIFICATION_SOURCES}
)
target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${CLASSIFICATION_INCLUDE_DIRS}
)

list(APPEND CMAKE_PREFIX_PATH ${libtorch_SOURCE_DIR})
find_package(Torch REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} ${CONAN_LIBS})

enable_testing()

add_executable(${CLASSIFICATION_TESTS} 
    ${CMAKE_CURRENT_SOURCE_DIR}/test/classifier.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test/dataset.cpp)

target_link_libraries(${CLASSIFICATION_TESTS} PRIVATE ${PROJECT_NAME} ${TORCH_LIBRARIES} ${CONAN_LIBS})

gtest_discover_tests(${CLASSIFICATION_TESTS})
