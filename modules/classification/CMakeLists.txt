cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(classification-module VERSION 1.0.0)

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
