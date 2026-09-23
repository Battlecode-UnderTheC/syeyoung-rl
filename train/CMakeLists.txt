cmake_minimum_required(VERSION 3.18)
project(dragons_train LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# the environment, as a static library (no Python module)
set(DRAGONS_PYTHON OFF CACHE BOOL "" FORCE)
add_subdirectory(../dragons-env ${CMAKE_BINARY_DIR}/dragons-env)

# The model runs through PJRT (the plugin is loaded at run time), so nothing
# here links against CUDA or JAX.
add_executable(trainer src/main.cpp src/pjrt.cpp src/artifacts.cpp src/rollout.cpp src/update.cpp)
target_include_directories(trainer PRIVATE src third_party)
target_link_libraries(trainer PRIVATE dragons ${CMAKE_DL_LIBS})
target_compile_options(trainer PRIVATE -O2 -Wall -Wextra)
