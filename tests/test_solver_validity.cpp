#include "game.hpp"
#include "cfr_cpu.hpp"
#include "test_support.hpp"

#include "holdem/action.hpp"
#include "holdem/betting_abstraction.hpp"
#include "holdem/subgame_builder.hpp"
#include "holdem/subgame_config.hpp"

#include "poker/board.hpp"
#include "poker/range.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kEvTol = 1e-4;

void check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void check_near(
    double actual,
    double expected,
    double tolerance,
    const std::string& message
) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        std::ostringstream oss;
        oss << message
            << " actual=" << actual
            << " expected=" << expected
            << " tolerance=" << tolerance;
        throw std::runtime_error(oss.str());
    }
}

poker::Board make_river_board() {
    return poker::Board{
        {
            phevaluator::Card("As"),
            phevaluator::Card("7h"),
            phevaluator::Card("Jh"),
            phevaluator::Card("Ts"),
            phevaluator::Card("3d"),
        }
    };
}

poker::Range make_p0_tiny_range() {
    poker::Range range;
    range.clear();

    // Pair of aces on the fixture board.
    range.set_weight(
        poker::make_hand(phevaluator::Card("Ah"), phevaluator::Card("Kh")),
        1.0f
    );

    // Pair of kings.
    range.set_weight(
        poker::make_hand(phevaluator::Card("Ks"), phevaluator::Card("Kd")),
        1.0f
    );

    return range;
}

poker::Range make_p1_tiny_range() {
    poker::Range range;
    range.clear();

    // Pair of queens.
    range.set_weight(
        poker::make_hand(phevaluator::Card("Qc"), phevaluator::Card("Qd")),
        1.0f
    );

    // Pair of tens on the fixture board.
    range.set_weight(
        poker::make_hand(phevaluator::Card("Th"), phevaluator::Card("9h")),
        1.0f
    );

    return range;
}

poker::holdem::BettingAbstraction make_check_only_betting() {
    poker::holdem::BettingAbstraction abstraction;

    // No bet sizes means the only legal unopened action should be check.
    abstraction.first_bet_sizes = {};
    abstraction.raise_sizes = {};
    abstraction.max_raises_per_street = 0;
    abstraction.always_allow_all_in = false;

    return abstraction;
}

poker::holdem::BettingAbstraction make_pot_bet_no_raise_betting() {
    poker::holdem::BettingAbstraction abstraction;

    // Unopened:
    //   check
    //   bet pot
    //
    // Facing bet:
    //   fold
    //   call
    //
    // No raises.
    abstraction.first_bet_sizes = {
        poker::holdem::BetSize::pot_fraction(1.0)
    };

    abstraction.raise_sizes = {};
    abstraction.max_raises_per_street = 0;
    abstraction.always_allow_all_in = false;

    return abstraction;
}

poker::holdem::HoldemSubgameConfig make_base_config() {
    poker::holdem::HoldemSubgameConfig config;

    config.board = make_river_board();

    config.pot_size = 1000;
    config.effective_stack = 2000;
    config.player_to_act = poker::Player::P0;

    config.p0_range = make_p0_tiny_range();
    config.p1_range = make_p1_tiny_range();

    config.collapse_all_in_runouts_to_ev = true;
    config.terminal_mode = poker::TerminalMode::ValuePrecomputed;
    return config;
}

poker::Game build_check_only_game() {
    poker::holdem::HoldemSubgameConfig config = make_base_config();
    config.betting_abstraction = make_check_only_betting();

    return poker::holdem::HoldemSubgameBuilder(config).build();
}

poker::Game build_pot_bet_game() {
    poker::holdem::HoldemSubgameConfig config = make_base_config();
    config.betting_abstraction = make_pot_bet_no_raise_betting();

    return poker::holdem::HoldemSubgameBuilder(config).build();
}

void test_check_only_river_ev_constant() {
    const auto game = build_check_only_game();
    check(game.hand_pairs.pair_count() == 4, "Expected four private matchups.");
    const auto uniform = test_support::uniform_strategy(game);
    test_support::check_strategy(game, uniform);
    poker::TerminalValueProvider terminal_values;
    poker::PublicExploitabilityEvaluator evaluator(game, terminal_values);
    // AhKh has a pair of aces; KK beats both QQ and Th9h (a pair of tens).
    check_near(evaluator.expected_value_p0(uniform), 1000.0, kEvTol,
               "Check-only EV should be 1000.");
    check_near(evaluator.exploitability(uniform).exploitability, 0.0, kEvTol,
               "A game with no decisions should have zero exploitability.");
    poker::CpuCfrSolver solver(game, terminal_values);
    solver.run_iterations(10);
    test_support::check_strategy(game, solver.average_strategy());
    check_near(solver.stats().last_root_value_p0, 1000.0, kEvTol,
               "CPU traversal must reproduce analytical EV.");
    std::cout << "[pass] check-only river EV\n";
}

void test_river_training_has_learning_signal() {
    const auto game = build_pot_bet_game();
    poker::TerminalValueProvider terminal_values;
    poker::CpuCfrSolver solver(game, terminal_values);
    test_support::check_strategy(game, solver.average_strategy());
    const auto initial = solver.current_strategy();
    solver.run_iterations(1000);
    const auto current = solver.current_strategy();
    const auto average = solver.average_strategy();
    test_support::check_strategy(game, current);
    test_support::check_strategy(game, average);
    check(solver.stats().iterations_run == 1000, "Iteration count mismatch.");
    bool learned = false;
    for (std::size_t i = 0; i < current.size(); ++i)
        learned |= std::abs(current[i] - initial[i]) > 0.05f;
    check(learned, "River strategy must change in response to terminal payoffs.");
    std::cout << "[pass] river training learns and produces normalized tensors\n";
}

void test_known_single_pair_river_converges() {
    auto config = test_support::tiny_config();
    config.p0_range.clear();
    config.p1_range.clear();
    config.p0_range.set_weight(poker::make_hand(phevaluator::Card("Kh"), phevaluator::Card("Qh")), 1.0f);
    config.p1_range.set_weight(poker::make_hand(phevaluator::Card("Qc"), phevaluator::Card("Qd")), 1.0f);
    const auto game = poker::holdem::HoldemSubgameBuilder(config).build();
    check(game.hand_pairs.pair_count() == 1, "Convergence fixture must have exactly one pair.");
    poker::TerminalValueProvider terminal_values;
    poker::PublicExploitabilityEvaluator evaluator(game, terminal_values);
    poker::CpuCfrSolver solver(game, terminal_values);
    const auto initial = evaluator.exploitability(solver.average_strategy());
    // The current best-response evaluator maximizes per private pair. With one
    // pair there is no hidden opponent hand, so it is a valid convergence oracle.
    solver.run_iterations(4000);
    const auto average = solver.average_strategy();
    test_support::check_strategy(game, average);
    const auto result = evaluator.exploitability(average);
    check(initial.exploitability > 100.0, "Uniform strategy must be exploitable.");
    check(result.exploitability < 1.0, "Known river game must converge to low exploitability.");
    check_near(result.strategy_value_p0, 1000.0, 1.0,
               "Winning straight must retain the starting pot at equilibrium.");
    for (const auto& state : game.action_states) {
        const int fold = test_support::local_action(game, state, poker::holdem::ActionType::Fold);
        if (state.player == 1 && fold >= 0)
            check(average.at(state.tensor_index(0, fold)) > 0.99f,
                  "Known losing hand must fold to a river bet.");
    }
    std::cout << "[pass] single-pair river convergence: exploitability=" << result.exploitability << '\n';
}

} // namespace

int main() {
    return test_support::run([] {
        test_check_only_river_ev_constant();
        test_river_training_has_learning_signal();
        test_known_single_pair_river_converges();
    });
}
