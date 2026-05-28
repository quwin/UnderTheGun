#include "game.hpp"
#include "cfr_cpu.hpp"
#include "exploitability.hpp"

#include "cfr_gpu.hpp"

#include "holdem/action.hpp"
#include "holdem/betting_abstraction.hpp"
#include "holdem/subgame_builder.hpp"
#include "holdem/subgame_config.hpp"
#include "holdem/street.hpp"

#include "poker/board.hpp"
#include "poker/range.hpp"

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kProbTol = 1e-6;
constexpr double kEvTol = 1e-4;
    
    void dump_terminal_paths_recursive(
        const poker::Game& game,
        int node_id,
        double reach,
        std::vector<std::string>& path
    ) {
        const poker::Node& node = game.node(node_id);

        if (node.id != game.root) {
            path.push_back(poker::to_string(node.incoming_action));
        }

        if (node.terminal) {
            std::cout << "terminal reach=" << reach
                      << " utility_p0=" << node.utility_p0
                      << " path=";

            for (const std::string& item : path) {
                std::cout << item << " / ";
            }

            std::cout << "\n";

            if (node.id != game.root) {
                path.pop_back();
            }

            return;
        }

        for (int child_id : node.children) {
            const poker::Node& child = game.node(child_id);

            double child_reach = reach;

            if (node.player == poker::Player::Chance) {
                child_reach *= static_cast<double>(child.chance_prob);
            }

            dump_terminal_paths_recursive(
                game,
                child_id,
                child_reach,
                path
            );
        }

        if (node.id != game.root) {
            path.pop_back();
        }
    }

    void dump_terminal_paths(const poker::Game& game) {
        std::vector<std::string> path;
        dump_terminal_paths_recursive(game, game.root, 1.0, path);
    }

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
    if (std::abs(actual - expected) > tolerance) {
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

    // K-high hand.
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

    // T-high hand.
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

    config.start_street = poker::holdem::Street::River;
    config.board = make_river_board();

    config.pot_size = 1000;
    config.effective_stack = 2000;
    config.player_to_act = poker::Player::P0;

    config.p0_range = make_p0_tiny_range();
    config.p1_range = make_p1_tiny_range();

    config.expand_all_in_runouts = true;
    config.collapse_all_in_runouts_to_ev = false;
    config.validate_tree_during_build = true;
    config.reject_preflop = true;

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

void check_strategy_is_normalized(
    const poker::Game& game,
    const std::vector<float>& strategy,
    double tolerance
) {
    check(
        static_cast<int>(strategy.size()) == game.num_q(),
        "Strategy size must match game.num_q()."
    );

    for (const poker::InfoSet& infoset : game.infosets) {
        double sum = 0.0;

        for (int q : infoset.q_indices) {
            const double p = static_cast<double>(strategy[q]);

            check(p >= -tolerance, "Strategy probability is negative.");
            check(p <= 1.0 + tolerance, "Strategy probability exceeds 1.");

            sum += p;
        }

        check_near(
            sum,
            1.0,
            tolerance,
            "Strategy probabilities must sum to 1 at each infoset."
        );
    }
}

std::vector<float> make_uniform_strategy(const poker::Game& game) {
    std::vector<float> strategy(game.num_q(), 0.0f);

    for (const poker::InfoSet& infoset : game.infosets) {
        const float p = 1.0f / static_cast<float>(infoset.q_indices.size());

        for (int q : infoset.q_indices) {
            strategy[q] = p;
        }
    }

    return strategy;
}

int local_action_for_type(
    const poker::InfoSet& infoset,
    poker::holdem::ActionType action_type
) {
    for (int local = 0; local < static_cast<int>(infoset.actions.size()); ++local) {
        if (infoset.actions[local].action_type ==
            static_cast<int>(action_type)) {
            return local;
        }
    }

    return -1;
}

double probability_for_action_type(
    const poker::InfoSet& infoset,
    const std::vector<float>& strategy,
    poker::holdem::ActionType action_type
) {
    const int local = local_action_for_type(infoset, action_type);

    if (local < 0) {
        throw std::runtime_error("Requested action type not found in infoset.");
    }

    const int q = infoset.q_indices[local];
    return static_cast<double>(strategy[q]);
}

void test_check_only_river_ev_constant() {
    const poker::Game game = build_check_only_game();

    dump_terminal_paths(game);

    check(game.num_nodes() > 0, "River game should have nodes.");
    check(game.num_infosets() > 0, "River game should have infosets.");
    check(game.num_q() > 0, "River game should have q entries.");

    const std::vector<float> uniform = make_uniform_strategy(game);
    check_strategy_is_normalized(game, uniform, kProbTol);

    poker::ExploitabilityEvaluator evaluator(game);

    const double ev = evaluator.expected_value_p0(uniform);
    std::cout << "computed EV = " << ev << "\n";

    // P0 wins 3 of 4 private matchups.
    // Utility convention:
    //   win  = +1000
    //   lose = 0
    // So EV = 750.
    check_near(
        ev,
        750.0,
        kEvTol,
        "Check-only river fixture should have analytically known P0 EV."
    );

    const poker::ExploitabilityResult result = evaluator.exploitability(uniform);

    check_near(
        result.strategy_value_p0,
        750.0,
        kEvTol,
        "Exploitability evaluator strategy value should match direct EV."
    );

    check_near(
        result.exploitability,
        0.0,
        kEvTol,
        "Check-only game should have zero exploitability because there are no real choices."
    );

    std::cout << "[pass] test_check_only_river_ev_constant\n";
}

void test_pot_bet_river_solver_converges() {
    const poker::Game game = build_pot_bet_game();

    poker::CfrConfig config;
    config.num_players = 2;
    config.simultaneous_updates = true;
    config.use_cfr_plus = false;
    config.linear_averaging = false;

    poker::CpuCfrSolver solver(game, config);

    solver.run_iterations(200000);

    const std::vector<float> avg = solver.average_strategy();

    check_strategy_is_normalized(game, avg, 1e-5);

    poker::ExploitabilityEvaluator evaluator(game);
    const poker::ExploitabilityResult result =
        evaluator.exploitability(avg);

    std::cout << "River pot-bet fixture:\n";
    std::cout << "  strategy_value_p0      = "
              << result.strategy_value_p0 << "\n";
    std::cout << "  best_response_value_p0 = "
              << result.best_response_value_p0 << "\n";
    std::cout << "  best_response_value_p1 = "
              << result.best_response_value_p1 << "\n";
    std::cout << "  p0_exploitability      = "
              << result.p0_exploitability << "\n";
    std::cout << "  p1_exploitability      = "
              << result.p1_exploitability << "\n";
    std::cout << "  exploitability         = "
              << result.exploitability << "\n";

    // Start loose. Tighten after you know the fixture is stable.
    check(
        result.exploitability < 5.0,
        "River pot-bet fixture should solve to low exploitability."
    );

    std::cout << "[pass] test_pot_bet_river_solver_converges\n";
}

void test_pot_bet_policy_tendencies() {
    const poker::Game game = build_pot_bet_game();

    poker::CfrConfig config;
    config.num_players = 2;
    config.simultaneous_updates = true;
    config.use_cfr_plus = false;
    config.linear_averaging = false;

    poker::CpuCfrSolver solver(game, config);
    solver.run_iterations(200000);

    const std::vector<float> avg = solver.average_strategy();

    bool checked_at_least_one_value_bet = false;
    bool checked_at_least_one_fold = false;

    for (const poker::InfoSet& infoset : game.infosets) {
        const int bet_local =
            local_action_for_type(infoset, poker::holdem::ActionType::Bet);

        const int fold_local =
            local_action_for_type(infoset, poker::holdem::ActionType::Fold);

        if (infoset.player == poker::Player::P0 && bet_local >= 0) {
            const double p_bet =
                probability_for_action_type(
                    infoset,
                    avg,
                    poker::holdem::ActionType::Bet
                );

            // This is intentionally not exact. It is a sanity assertion:
            // at least one P0 river infoset should choose bet sometimes.
            if (p_bet > 0.05) {
                checked_at_least_one_value_bet = true;
            }
        }

        if (infoset.player == poker::Player::P1 && fold_local >= 0) {
            const double p_fold =
                probability_for_action_type(
                    infoset,
                    avg,
                    poker::holdem::ActionType::Fold
                );

            // Facing a pot bet, at least one weak P1 infoset should fold sometimes.
            if (p_fold > 0.05) {
                checked_at_least_one_fold = true;
            }
        }
    }

    check(
        checked_at_least_one_value_bet,
        "Expected at least one P0 infoset to bet with nontrivial probability."
    );

    check(
        checked_at_least_one_fold,
        "Expected at least one P1 infoset to fold with nontrivial probability."
    );

    std::cout << "[pass] test_pot_bet_policy_tendencies\n";
}

void run_all_tests() {
    test_check_only_river_ev_constant();
    test_pot_bet_river_solver_converges();
    test_pot_bet_policy_tendencies();
}

} // namespace

int main() {
    try {
        run_all_tests();
    } catch (const std::exception& e) {
        std::cerr << "[fail] " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "[pass] all river solver validity tests passed\n";
    return EXIT_SUCCESS;
}