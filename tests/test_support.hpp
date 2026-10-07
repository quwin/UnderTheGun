#pragma once

#include "game.hpp"
#include "holdem/action.hpp"
#include "holdem/subgame_builder.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace test_support {

inline void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

inline void check_near(double actual, double expected, double tolerance,
                       const std::string& message) {
    check(std::isfinite(actual) && std::isfinite(expected) &&
              std::abs(actual - expected) <= tolerance,
          message + " actual=" + std::to_string(actual) +
              " expected=" + std::to_string(expected));
}

inline poker::holdem::HoldemSubgameConfig tiny_config(int board_cards = 5) {
    poker::holdem::HoldemSubgameConfig config;
    config.board = poker::Board{{phevaluator::Card("As"), phevaluator::Card("7h"),
                                phevaluator::Card("Jh")}};
    if (board_cards >= 4) config.board.cards.push_back(phevaluator::Card("Ts"));
    if (board_cards == 5) config.board.cards.push_back(phevaluator::Card("4s"));
    config.pot_size = 1000;
    config.effective_stack = 2000;
    config.p0_range.clear();
    config.p1_range.clear();
    config.p0_range.set_weight(poker::make_hand(phevaluator::Card("Kh"), phevaluator::Card("Qh")), 1.0f);
    config.p0_range.set_weight(poker::make_hand(phevaluator::Card("Ks"), phevaluator::Card("Kd")), 1.0f);
    config.p1_range.set_weight(poker::make_hand(phevaluator::Card("Qc"), phevaluator::Card("Qd")), 1.0f);
    config.p1_range.set_weight(poker::make_hand(phevaluator::Card("Th"), phevaluator::Card("9h")), 1.0f);
    config.betting_abstraction.first_bet_sizes = {poker::holdem::BetSize::pot_fraction(1.0)};
    config.betting_abstraction.raise_sizes.clear();
    config.betting_abstraction.max_raises_per_street = 0;
    config.betting_abstraction.always_allow_all_in = false;
    config.terminal_mode = poker::TerminalMode::ValuePrecomputed;
    return config;
}

inline std::vector<float> uniform_strategy(const poker::Game& game) {
    std::vector<float> strategy(game.cfr_tensor_entries(), 0.0f);
    for (const auto& state : game.action_states)
        for (int bucket = 0; bucket < state.bucket_count; ++bucket)
            for (int action = 0; action < state.action_count; ++action)
                strategy.at(state.tensor_index(bucket, action)) = 1.0f / state.action_count;
    return strategy;
}

inline void check_strategy(const poker::Game& game, const std::vector<float>& strategy,
                           double tolerance = 1e-5) {
    check(strategy.size() == game.cfr_tensor_entries(), "Strategy tensor size mismatch.");
    for (const auto& state : game.action_states) {
        for (int bucket = 0; bucket < state.bucket_count; ++bucket) {
            double sum = 0.0;
            for (int action = 0; action < state.action_count; ++action) {
                const float p = strategy.at(state.tensor_index(bucket, action));
                check(std::isfinite(p) && p >= 0.0f && p <= 1.0f,
                      "Strategy contains an invalid probability.");
                sum += p;
            }
            check_near(sum, 1.0, tolerance, "Strategy bucket must sum to one.");
        }
    }
}

inline int local_action(const poker::Game& game, const poker::ActionState& state,
                        poker::holdem::ActionType type) {
    const auto& node = game.node(state.node);
    for (int a = 0; a < state.action_count; ++a)
        if (game.action(game.edge(node.first_edge + a).action_index).action_type == static_cast<int>(type))
            return a;
    return -1;
}

template <typename Test>
int run(Test test) {
    try {
        test();
        std::cout << "[pass] all tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[fail] " << error.what() << '\n';
        return 1;
    }
}

} // namespace test_support
