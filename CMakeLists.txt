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

set(CMAKE_CUDA_ARCHITECTURES 61 75 80 86 89)

# 3. Find dependencies
find_package(CUDAToolkit REQUIRED)

# ---- GUI executable ----
set(FLTK_DIR "C:/fltk-1.4.5/build")
find_package(FLTK 1.4 REQUIRED CONFIG)

add_library(gui_library
    src/gui/components/CardButton.cpp
    src/gui/components/CardButton.hh
    src/gui/components/ComboStrategyDisplay.cpp
    src/gui/components/ComboStrategyDisplay.hh
    src/gui/components/Page1_Settings.cpp
    src/gui/components/Page1_Settings.hh
    src/gui/components/Page2_Board.cpp
    src/gui/components/Page2_Board.hh
    src/gui/components/Page3_HeroRange.cpp
    src/gui/components/Page3_HeroRange.hh
    src/gui/components/Page4_VillainRange.cpp
    src/gui/components/Page4_VillainRange.hh
    src/gui/components/Page5_Progress.cpp
    src/gui/components/Page5_Progress.hh
    src/gui/components/Page6_Strategy.cpp
    src/gui/components/Page6_Strategy.hh
    src/gui/utils/Colors.hh
    src/gui/utils/EmbeddedLogo.hh
    src/gui/utils/MemoryUtil.hh
)

target_link_libraries(gui_library
        PRIVATE
        fltk::fltk
        fltk::images
)


add_executable(under_the_gun_gui WIN32
        src/gui/gui.cpp
)

set_target_properties(under_the_gun_gui
        PROPERTIES
        OUTPUT_NAME "UnderTheGun"
)

target_include_directories(under_the_gun_gui
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gui
        ${FLTK_INCLUDE_DIR}
)

target_link_libraries(under_the_gun_gui
        PRIVATE
        kuhn_gpu_cfr_lib
        gui_library
        fltk::fltk
        fltk::images
)

if (WIN32)
    target_link_libraries(under_the_gun_gui
            PRIVATE
            dwmapi
            comctl32
            gdi32
            user32
            ole32
            uuid
    )
endif()
# ---------------------------------------------------------------------------
# PokerHandEvaluator
# ---------------------------------------------------------------------------

set(BUILD_CARD5 OFF CACHE BOOL "" FORCE)
set(BUILD_CARD6 OFF CACHE BOOL "" FORCE)
set(BUILD_CARD7 OFF  CACHE BOOL "" FORCE)
set(BUILD_PLO4 OFF CACHE BOOL "" FORCE)
set(BUILD_PLO5 OFF CACHE BOOL "" FORCE)
set(BUILD_PLO6 OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(external/PokerHandEvaluator/cpp pheval EXCLUDE_FROM_ALL)

get_target_property(PHEVAL_OPTIONS pheval INTERFACE_COMPILE_OPTIONS)
if (PHEVAL_OPTIONS)
    list(REMOVE_ITEM PHEVAL_OPTIONS "-O3")
    set_target_properties(pheval PROPERTIES
            INTERFACE_COMPILE_OPTIONS "${PHEVAL_OPTIONS}"
    )
endif()

# ------------------------------------------------------------------------------
# Core project library
# ------------------------------------------------------------------------------

add_library(kuhn_gpu_cfr_lib
        include/game.hpp
        include/cfr_cpu.hpp
        include/exploitability.hpp
        include/gpu_state.hpp
        include/cfr_gpu.hpp

        include/poker/board.hpp
        include/poker/deck_mask.hpp
        include/poker/hand.hpp
        include/poker/range.hpp

        include/holdem/action.hpp
        include/holdem/all_in_equity_cache.hpp
        include/holdem/betting_abstraction.hpp
        include/holdem/betting_engine.hpp
        include/holdem/betting_state.hpp
        include/holdem/board_abstraction.hpp
        include/holdem/chance_model.hpp
        include/holdem/hand_abstraction.hpp
        include/holdem/infoset_key.hpp
        include/holdem/public_state.hpp
        include/holdem/private_state.hpp
        include/holdem/subgame_builder.hpp
        include/holdem/subgame_config.hpp
        include/holdem/terminal_utility.hpp

        src/cfr_cpu.cpp
        src/exploitability.cpp
        src/cfr_gpu.cpp

        src/holdem/betting_engine.cpp
        src/holdem/subgame_builder.cpp

        cuda/kernels.cu
        cuda/kernels.hpp
        src/game.cpp
        src/evaluator_table.cpp
)

add_library(cuda_evaluator7 STATIC IMPORTED)

set_target_properties(cuda_evaluator7 PROPERTIES
        IMPORTED_LOCATION
        "../external/CUDA-Poker-Calculator/lib-cuda/evaluator7.lib"
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
        pheval
        CUDA::cudart
        cuda_evaluator7
)

target_compile_definitions(kuhn_gpu_cfr_lib
        PUBLIC
        UNDER_THE_GUN_ENABLE_CUDA
)

set_target_properties(kuhn_gpu_cfr_lib
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ------------------------------------------------------------------------------
# Main executable
# ------------------------------------------------------------------------------

add_executable(UnderTheGunBenchmark
        src/main.cpp
)

target_link_libraries(UnderTheGunBenchmark
        PRIVATE
        kuhn_gpu_cfr_lib
)

set_target_properties(UnderTheGunBenchmark
        PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------

enable_testing()

add_executable(test_solver_validity
        tests/test_solver_validity.cpp
)
target_link_libraries(test_solver_validity
        PRIVATE
        kuhn_gpu_cfr_lib
)
add_test(
        NAME test_solver_validity
        COMMAND test_solver_validity
)
add_executable(test_gpu_strategy
        tests/test_gpu_strategy.cpp
)
target_link_libraries(test_gpu_strategy
        PRIVATE
        kuhn_gpu_cfr_lib
)
add_test(
        NAME test_gpu_strategy
        COMMAND test_gpu_strategy
)

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
add_executable(test_holdem_gpu
        tests/test_holdem_gpu.cpp
)
target_link_libraries(test_holdem_gpu
        PRIVATE
        kuhn_gpu_cfr_lib
)
add_test(
        NAME test_holdem_gpu
        COMMAND test_holdem_gpu
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

include(GNUInstallDirs)

install(
        TARGETS under_the_gun_gui
        RUNTIME DESTINATION .
)

install(
        FILES
        $<TARGET_RUNTIME_DLLS:under_the_gun_gui>
        DESTINATION .
)

install(
        CODE [[
        file(GET_RUNTIME_DEPENDENCIES
            EXECUTABLES "$<TARGET_FILE:under_the_gun_gui>"
            RESOLVED_DEPENDENCIES_VAR resolved_deps
            UNRESOLVED_DEPENDENCIES_VAR unresolved_deps
        )

        foreach(dep IN LISTS resolved_deps)
            file(INSTALL
                DESTINATION "${CMAKE_INSTALL_PREFIX}"
                TYPE FILE
                FILES "${dep}"
            )
        endforeach()

        if(unresolved_deps)
            message(WARNING "Unresolved runtime dependencies: ${unresolved_deps}")
        endif()
    ]]
)

set(CPACK_PACKAGE_NAME "UnderTheGun")
set(CPACK_PACKAGE_VENDOR "UnderTheGun")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_FILE_NAME "UnderTheGun-${CPACK_PACKAGE_VERSION}-windows-x64")

include(CPack)