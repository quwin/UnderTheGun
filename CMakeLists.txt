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

        include/poker/card.hpp
        include/poker/deck_mask.hpp
        include/poker/hand.hpp
        include/poker/board.hpp
        include/poker/range.hpp
        include/poker/hand_evaluator.hpp

        include/holdem/street.hpp
        include/holdem/action.hpp
        include/holdem/betting_state.hpp
        include/holdem/betting_abstraction.hpp
        include/holdem/betting_engine.hpp
        include/holdem/public_state.hpp
        include/holdem/private_state.hpp
        include/holdem/chance_model.hpp
        include/holdem/terminal_utility.hpp
        include/holdem/infoset_key.hpp
        include/holdem/hand_abstraction.hpp
        include/holdem/board_abstraction.hpp
        include/holdem/subgame_config.hpp
        include/holdem/subgame_builder.hpp

        src/kuhn_builder.cpp
        src/cfr_cpu.cpp
        src/exploitability.cpp
        src/cfr_gpu.cpp

        src/poker/hand_evaluator.cpp
        src/holdem/betting_engine.cpp
        src/holdem/subgame_builder.cpp

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

add_executable(test_betting_engine
        tests/test_betting_engine.cpp
)
target_link_libraries(test_betting_engine
        PRIVATE
        kuhn_gpu_cfr_lib
)
add_test(
        NAME test_betting_engine
        COMMAND test_betting_engine
)

add_executable(test_deck_mask
        tests/test_deck_mask.cpp
)
target_link_libraries(test_deck_mask
        PRIVATE
        kuhn_gpu_cfr_lib
)
add_test(
        NAME test_deck_mask
        COMMAND test_deck_mask
)

add_executable(test_flop_subgame_tree
        tests/test_flop_subgame_tree.cpp
)
target_link_libraries(test_flop_subgame_tree
        PRIVATE
        kuhn_gpu_cfr_lib
)
add_test(
        NAME test_flop_subgame_tree
        COMMAND test_flop_subgame_tree
)

add_executable(test_hand_evaluator
        tests/test_hand_evaluator.cpp
)
target_link_libraries(test_hand_evaluator
        PRIVATE
        kuhn_gpu_cfr_lib
)
add_test(
        NAME test_hand_evaluator
        COMMAND test_hand_evaluator
)

add_executable(test_infoset_key_privacy
        tests/test_infoset_key_privacy.cpp
)
target_link_libraries(test_infoset_key_privacy
        PRIVATE
        kuhn_gpu_cfr_lib
)
add_test(
        NAME test_infoset_key_privacy
        COMMAND test_infoset_key_privacy
)

add_executable(test_range_card_removal
        tests/test_range_card_removal.cpp
)
target_link_libraries(test_range_card_removal
        PRIVATE
        kuhn_gpu_cfr_lib
)
add_test(
        NAME test_range_card_removal
        COMMAND test_range_card_removal
)
add_executable(test_river_subgame_tree
        tests/test_river_subgame_tree.cpp
)
target_link_libraries(test_river_subgame_tree
        PRIVATE
        kuhn_gpu_cfr_lib
)
add_test(
        NAME test_river_subgame_tree
        COMMAND test_river_subgame_tree
)

add_executable(test_turn_subgame_tree
        tests/test_turn_subgame_tree.cpp
)
target_link_libraries(test_turn_subgame_tree
        PRIVATE
        kuhn_gpu_cfr_lib
)
add_test(
        NAME test_turn_subgame_tree
        COMMAND test_turn_subgame_tree
)

add_executable(test_terminal_utility
        tests/test_terminal_utility.cpp
)
target_link_libraries(test_terminal_utility
        PRIVATE
        kuhn_gpu_cfr_lib
)
add_test(
        NAME test_terminal_utility
        COMMAND test_terminal_utility
)