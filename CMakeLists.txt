cmake_minimum_required(VERSION 3.24)

project(UnderTheGun LANGUAGES CXX CUDA)
# ------------------------------------------------------------------------------
# Language standards
# ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures" FORCE)

find_package(CUDAToolkit REQUIRED)
# ------------------------------------------------------------------------------
# Core project library
# ------------------------------------------------------------------------------

add_library(kuhn_gpu_cfr_lib
        include/game.hpp
        include/kuhn_builder.hpp
        include/cfr_cpu.hpp
        include/exploitability.hpp
        include/gpu_state.hpp
        include/cfr_gpu.hpp

        src/kuhn_builder.cpp
        src/cfr_cpu.cpp
        src/exploitability.cpp
        src/cfr_gpu.cpp

        cuda/kernels.cu
        cuda/kernels.hpp
)

target_include_directories(kuhn_gpu_cfr_lib
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda
)

target_compile_features(kuhn_gpu_cfr_lib
        PUBLIC
        cxx_std_20
)

target_link_libraries(kuhn_gpu_cfr_lib
        PRIVATE
        CUDA::cudart
)

target_compile_definitions(kuhn_gpu_cfr_lib
        PUBLIC
        UNDER_THE_GUN_ENABLE_CUDA
)

set_target_properties(kuhn_gpu_cfr_lib
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

# ------------------------------------------------------------------------------
# Main executable
# ------------------------------------------------------------------------------

add_executable(UnderTheGun
        src/main.cpp
)

target_link_libraries(UnderTheGun
        PRIVATE
        kuhn_gpu_cfr_lib
)

set_target_properties(UnderTheGun
        PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------

enable_testing()

add_executable(test_kuhn_tree
        tests/test_kuhn_tree.cpp
)

target_link_libraries(test_kuhn_tree
        PRIVATE
        kuhn_gpu_cfr_lib
)

add_test(
        NAME test_kuhn_tree
        COMMAND test_kuhn_tree
)

add_executable(test_cfr_cpu
        tests/test_cfr_cpu.cpp
)

target_link_libraries(test_cfr_cpu
        PRIVATE
        kuhn_gpu_cfr_lib
)

add_test(
        NAME test_cfr_cpu
        COMMAND test_cfr_cpu
)

add_executable(test_gpu_vs_cpu
        tests/test_gpu_vs_cpu.cpp
)

target_link_libraries(test_gpu_vs_cpu
        PRIVATE
        kuhn_gpu_cfr_lib
)

set_target_properties(test_gpu_vs_cpu
        PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_test(
        NAME test_gpu_vs_cpu
        COMMAND test_gpu_vs_cpu
)