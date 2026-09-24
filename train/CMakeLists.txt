cmake_minimum_required(VERSION 3.18)
project(dragons_rl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# the engine port, as a static library
set(DRAGONS_PYTHON OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../dragons-env ${CMAKE_BINARY_DIR}/dragons-env)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_native native/observation.cpp native/facts.cpp native/critic.cpp native/collector.cpp
                    native/bindings.cpp)
target_include_directories(_native PRIVATE native)
target_compile_options(_native PRIVATE -O3 -Wall -Wextra)
target_link_libraries(_native PRIVATE dragons)
install(TARGETS _native DESTINATION dragons_rl)
