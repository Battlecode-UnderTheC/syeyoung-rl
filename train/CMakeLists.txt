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

# facts, collection and replay: shared by the Python module (tests, build steps) and the runtime
add_library(rl_native STATIC native/observation.cpp native/facts.cpp native/critic.cpp native/collector.cpp
            native/policy.cpp)
target_include_directories(rl_native PUBLIC native)
target_compile_options(rl_native PRIVATE -O3 -Wall -Wextra)
set_target_properties(rl_native PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(rl_native PUBLIC dragons)

# evaluation on the CPU with the plain-C++ policy (no Python, no GPU)
add_executable(dragons_eval native/eval_main.cpp)
target_compile_options(dragons_eval PRIVATE -O3 -Wall -Wextra)
find_package(Threads REQUIRED)
target_link_libraries(dragons_eval PRIVATE rl_native Threads::Threads)

option(DRAGONS_RL_PYTHON "build the Python module" ON)
if(DRAGONS_RL_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_native native/bindings.cpp)
  target_compile_options(_native PRIVATE -O3 -Wall -Wextra)
  target_link_libraries(_native PRIVATE rl_native)
  install(TARGETS _native DESTINATION dragons_rl)
endif()

# the C++ runtime: compiled programs through a PJRT plugin, no Python
option(DRAGONS_RL_RUNTIME "build the C++ runtime" OFF)
if(DRAGONS_RL_RUNTIME)
  find_package(Threads REQUIRED)
  add_library(rl_runtime STATIC runtime/pjrt.cpp runtime/worker.cpp runtime/learner.cpp)
  target_include_directories(rl_runtime PUBLIC runtime third_party/pjrt)
  target_compile_options(rl_runtime PRIVATE -O3 -Wall -Wextra)
  target_link_libraries(rl_runtime PUBLIC rl_native Threads::Threads ${CMAKE_DL_LIBS})
  add_executable(dragons_collect runtime/collect_main.cpp)
  target_link_libraries(dragons_collect PRIVATE rl_runtime)
  add_executable(dragons_train runtime/trainer_main.cpp)
  target_link_libraries(dragons_train PRIVATE rl_runtime)
endif()
