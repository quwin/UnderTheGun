#include "game.hpp"

#include "holdem/betting_abstraction.hpp"
#include "holdem/subgame_builder.hpp"
#include "holdem/subgame_config.hpp"
#include "holdem/street.hpp"
#include "exploitability.hpp"
#include "kuhn_builder.hpp"

#include "poker/board.hpp"
#include "poker/card.hpp"
#include "poker/range.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cfr_cpu.hpp"
#include "cfr_gpu.hpp"

namespace {
    void check(bool condition, const std::string& message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    void check_eq(
        int actual,
        int expected,
        const std::string& message
    ) {
        if (actual != expected) {
            std::ostringstream oss;
            oss << message
                << " actual=" << actual
                << " expected=" << expected;
            throw std::runtime_error(oss.str());
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
    void check_strategy_is_normalized_by_infoset(
        const poker::Game& game,
        const std::vector<float>& strategy,
        double tolerance,
        const std::string& label
    ) {
        check(
            static_cast<int>(strategy.size()) == game.num_q(),
            label + ": strategy size should equal game.num_q()."
        );

        for (const poker::InfoSet& infoset : game.infosets) {
            double sum = 0.0;

            for (int q : infoset.q_indices) {
                check(q >= 0 && q < static_cast<int>(strategy.size()),
                      label + ": q index out of range.");

                check(std::isfinite(strategy[q]),
                      label + ": strategy probability is not finite.");

                check(strategy[q] >= -tolerance,
                      label + ": strategy probability is negative.");

                check(strategy[q] <= 1.0 + tolerance,
                      label + ": strategy probability is greater than one.");

                sum += static_cast<double>(strategy[q]);
            }

            check_near(
                sum,
                1.0,
                tolerance,
                label + ": probabilities should sum to one within each infoset."
            );
        }
    }
    double max_abs_diff(
        const std::vector<float>& a,
        const std::vector<float>& b
    ) {
            check(a.size() == b.size(), "Cannot compare strategies of different size.");

            double max_diff = 0.0;

            for (std::size_t i = 0; i < a.size(); ++i) {
                max_diff = std::max(
                    max_diff,
                    std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]))
                );
            }

            return max_diff;
        }

    poker::Board make_test_river_board() {
        return poker::Board{
            {
                poker::make_card(poker::Rank::Ace, poker::Suit::Spades),
                poker::make_card(poker::Rank::Seven, poker::Suit::Hearts),
                poker::make_card(poker::Rank::Two, poker::Suit::Clubs),
                poker::make_card(poker::Rank::Jack, poker::Suit::Diamonds),
                poker::make_card(poker::Rank::Four, poker::Suit::Spades)
            }
        };
    }

    poker::Range make_tiny_p0_range() {
        poker::Range range;
        range.clear();

        range.set_weight(
            poker::make_hand(
                poker::make_card(poker::Rank::King, poker::Suit::Hearts),
                poker::make_card(poker::Rank::Queen, poker::Suit::Hearts)
            ),
            1.0f
        );

        range.set_weight(
            poker::make_hand(
                poker::make_card(poker::Rank::King, poker::Suit::Spades),
                poker::make_card(poker::Rank::King, poker::Suit::Diamonds)
            ),
            1.0f
        );

        return range;
    }

    poker::Range make_tiny_p1_range() {
        poker::Range range;
        range.clear();

        range.set_weight(
            poker::make_hand(
                poker::make_card(poker::Rank::Queen, poker::Suit::Clubs),
                poker::make_card(poker::Rank::Queen, poker::Suit::Diamonds)
            ),
            1.0f
        );

        range.set_weight(
            poker::make_hand(
                poker::make_card(poker::Rank::Ten, poker::Suit::Hearts),
                poker::make_card(poker::Rank::Nine, poker::Suit::Hearts)
            ),
            1.0f
        );

        return range;
    }

    poker::holdem::BettingAbstraction make_tiny_betting_abstraction() {
        poker::holdem::BettingAbstraction abstraction;

        // Tiny structural tree:
        //
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

        return abstraction;
    }

    poker::holdem::HoldemSubgameConfig make_test_config() {
        poker::holdem::HoldemSubgameConfig config;

        config.start_street = poker::holdem::Street::River;
        config.board = make_test_river_board();

        config.pot_size = 1000;
        config.effective_stack = 2000;
        config.player_to_act = poker::Player::P0;

        config.p0_range = make_tiny_p0_range();
        config.p1_range = make_tiny_p1_range();

        config.betting_abstraction = make_tiny_betting_abstraction();

        // config.hand_abstraction = ;
        // config.board_abstraction = ;

        return config;
    }

    poker::Game build_test_game() {
        const poker::holdem::HoldemSubgameConfig config = make_test_config();
        return poker::holdem::HoldemSubgameBuilder(config).build();
    }

    int count_terminal_nodes(const poker::Game& game) {
        int count = 0;

        for (const poker::Node& node : game.nodes) {
            if (node.terminal) {
                ++count;
            }
        }

        return count;
    }

    int count_nodes_with_player(
        const poker::Game& game,
        poker::Player player
    ) {
        int count = 0;

        for (const poker::Node& node : game.nodes) {
            if (node.player == player) {
                ++count;
            }
        }

        return count;
    }

    bool has_action_type(const poker::InfoSet& infoset, poker::holdem::ActionType action_type) {
        for (const poker::GameAction action : infoset.actions) {
            if (action.action_type == static_cast<int>(action_type)) {
                return true;
            }
        }
        return false;
    }

    void test_holdem_subgame_flattens_for_gpu() {
        poker::Game game = build_test_game();

        poker::FlatGame flat = poker::flatten_game_for_gpu(game);

        check(flat.valid_basic_shape(), "FlatGame should have valid basic shape.");
        check(flat.num_nodes == game.num_nodes(), "Flat node count mismatch.");
        check(flat.num_infosets == game.num_infosets(), "Flat infoset count mismatch.");
        check(flat.num_q == game.num_q(), "Flat q count mismatch.");
        check(flat.root == game.root, "Flat root mismatch.");

        for (const poker::InfoSet& infoset : game.infosets) {
            check(!infoset.actions.empty(), "Infoset should have actions.");
            check(
                infoset.actions.size() == infoset.q_indices.size(),
                "Infoset actions and q_indices should align."
            );

            for (int i = 0; i < static_cast<int>(infoset.q_indices.size()); ++i) {
                check(
                    infoset.q_indices[i] == infoset.q_indices.front() + i,
                    "Infoset q_indices must be contiguous for GPU."
                );
            }
        }
        std::cout << "[pass] test_holdem_subgame_flattens_for_gpu\n";
    }

    void test_holdem_subgame_runs_one_gpu_iteration() {
        poker::Game game = build_test_game();

        poker::GpuCfrConfig config;
        config.synchronize_each_iteration = true;

        poker::GpuCfrSolver solver(game, config);
        solver.run_one_iteration();

        check(
            solver.stats().iterations_run == 1,
            "GPU solver should run one iteration."
        );

        std::vector<float> avg = solver.average_strategy();

        check(
            static_cast<int>(avg.size()) == game.num_q(),
            "Average strategy size should match game.num_q()."
        );
        std::cout << "[pass] test_holdem_subgame_runs_one_gpu_iteration\n";
    }

    void test_cpu_gpu_one_iteration_average_strategy_close() {
        const poker::Game game = poker::KuhnGameBuilder().build();

        poker::CfrConfig cpu_config;
        cpu_config.use_cfr_plus = false;
        cpu_config.linear_averaging = false;

        poker::GpuCfrConfig gpu_config;
        gpu_config.use_cfr_plus = false;
        gpu_config.linear_averaging = false;
        gpu_config.synchronize_each_iteration = true;

        poker::CpuCfrSolver cpu(game, cpu_config);
        poker::GpuCfrSolver gpu(game, gpu_config);

        cpu.run_one_iteration();
        gpu.run_one_iteration();

        const std::vector<float> cpu_current = cpu.average_strategy();
        const std::vector<float> gpu_current = gpu.average_strategy();

        check_strategy_is_normalized_by_infoset(
            game,
            cpu_current,
            1e-5,
            "CPU average strategy"
        );

        check_strategy_is_normalized_by_infoset(
            game,
            gpu_current,
            1e-5,
            "GPU average strategy"
        );

        check_near(
            max_abs_diff(cpu_current, gpu_current),
            0.0,
            1e-4,
            "CPU and GPU average strategies should match after one iteration"
        );

        check(cpu.stats().iterations_run == 1, "CPU should report one iteration.");
        check(gpu.stats().iterations_run == 1, "GPU should report one iteration.");

        std::cout << "[pass] test_cpu_gpu_one_iteration_current_strategy_close\n";
    }

    void test_cpu_gpu_average_strategy_close_after_short_run() {
        const poker::Game game = poker::KuhnGameBuilder().build();

        constexpr int kIterations = 500;

        poker::CfrConfig cpu_config;
        cpu_config.use_cfr_plus = false;
        cpu_config.linear_averaging = false;

        poker::GpuCfrConfig gpu_config;
        gpu_config.use_cfr_plus = false;
        gpu_config.linear_averaging = false;
        gpu_config.synchronize_each_iteration = true;

        poker::CpuCfrSolver cpu(game, cpu_config);
        poker::GpuCfrSolver gpu(game, gpu_config);

        cpu.run_iterations(kIterations);
        gpu.run_iterations(kIterations);

        const std::vector<float> cpu_avg = cpu.average_strategy();
        const std::vector<float> gpu_avg = gpu.average_strategy();

        check_strategy_is_normalized_by_infoset(
            game,
            cpu_avg,
            1e-5,
            "CPU average strategy"
        );

        check_strategy_is_normalized_by_infoset(
            game,
            gpu_avg,
            1e-5,
            "GPU average strategy"
        );

        check_near(
            max_abs_diff(cpu_avg, gpu_avg),
            0.0,
            0.1,
            "CPU and GPU average strategies should be close after short run"
        );
        std::cout << "[pass] test_cpu_gpu_average_strategy_close_after_short_run\n";
    }

    void test_cpu_gpu_exploitability_close_after_short_run() {
        const poker::Game game = poker::KuhnGameBuilder().build();

        constexpr int kIterations = 500;

        poker::CfrConfig cpu_config;
        cpu_config.use_cfr_plus = false;
        cpu_config.linear_averaging = false;

        poker::GpuCfrConfig gpu_config;
        gpu_config.use_cfr_plus = false;
        gpu_config.linear_averaging = false;
        gpu_config.synchronize_each_iteration = true;

        poker::CpuCfrSolver cpu(game, cpu_config);
        poker::GpuCfrSolver gpu(game, gpu_config);

        cpu.run_iterations(kIterations);
        gpu.run_iterations(kIterations);

        const std::vector<float> cpu_avg = cpu.average_strategy();
        const std::vector<float> gpu_avg = gpu.average_strategy();

        poker::ExploitabilityEvaluator evaluator(game);

        const poker::ExploitabilityResult cpu_result =
            evaluator.exploitability(cpu_avg);

        const poker::ExploitabilityResult gpu_result =
            evaluator.exploitability(gpu_avg);

        check(std::isfinite(cpu_result.exploitability),
              "CPU exploitability should be finite.");

        check(std::isfinite(gpu_result.exploitability),
              "GPU exploitability should be finite.");

        check_near(
            gpu_result.strategy_value_p0,
            cpu_result.strategy_value_p0,
            1e-2,
            "CPU and GPU average strategies should have similar EV"
        );

        check_near(
            gpu_result.exploitability,
            cpu_result.exploitability,
            1e-2,
            "CPU and GPU average strategies should have similar exploitability"
        );

        std::cout << "[pass] test_cpu_gpu_exploitability_close_after_short_run\n";
    }
    void run_all_tests() {
        test_holdem_subgame_flattens_for_gpu();
        test_holdem_subgame_runs_one_gpu_iteration();
        test_cpu_gpu_one_iteration_average_strategy_close();
        test_cpu_gpu_average_strategy_close_after_short_run();
        test_cpu_gpu_exploitability_close_after_short_run();
    }

} // namespace

int main() {
    try {
        run_all_tests();
    } catch (const std::exception& e) {
        std::cerr << "[fail] " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "[pass] all river subgame tree tests passed\n";
    return EXIT_SUCCESS;
}
