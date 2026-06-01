#include "../include/cfr_cpu.hpp"
#include "../include/cfr_gpu.hpp"
#include "../include/game.hpp"

#include "../include/holdem/subgame_builder.hpp"
#include "../include/holdem/subgame_config.hpp"
#include "../include/holdem/betting_abstraction.hpp"

#include "../include/poker/board.hpp"
#include "../include/poker/hand.hpp"
#include "../include/poker/range.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kTol = 1e-5;

poker::Board make_test_board() {
    return poker::Board{
        {
            phevaluator::Card("As"),
            phevaluator::Card("7h"),
            phevaluator::Card("Jh"),
            phevaluator::Card("Ts"),
            phevaluator::Card("4d")
        }
    };
}

poker::Range make_tiny_p0_range() {
    poker::Range range;
    range.clear();

    range.set_weight(
        poker::make_hand(
            phevaluator::Card("Kh"),
            phevaluator::Card("Qh")
        ),
        1.0f
    );

    range.set_weight(
        poker::make_hand(
            phevaluator::Card("Ks"),
            phevaluator::Card("Kd")
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
            phevaluator::Card("Qc"),
            phevaluator::Card("Qd")
        ),
        1.0f
    );

    range.set_weight(
        poker::make_hand(
            phevaluator::Card("Th"),
            phevaluator::Card("9h")
        ),
        1.0f
    );

    return range;
}

poker::holdem::HoldemSubgameConfig make_test_config() {
    poker::holdem::HoldemSubgameConfig config;

    config.board = make_test_board();
    config.pot_size = 100;
    config.effective_stack = 2000;
    config.player_to_act = poker::Player::P0;
    config.oop_player = poker::Player::P0;
    config.ip_player = poker::Player::P1;
    config.collapse_all_in_runouts_to_ev = true;

    config.p0_range = make_tiny_p0_range();
    config.p1_range = make_tiny_p1_range();

    config.betting_abstraction =
        poker::holdem::make_standard_abstraction();

    return config;
}

poker::Game make_test_game() {
    poker::holdem::HoldemSubgameBuilder builder(make_test_config());
    poker::Game game = builder.build();
    game.validate();

    if (game.terminal_value_p0.empty()) {
        throw std::runtime_error(
            "Test game has empty terminal_value_p0. "
            "GPU HostPrecomputed terminal mode cannot learn without terminal values."
        );
    }

    return game;
}

bool nearly_equal(double a, double b, double tol = kTol) {
    return std::abs(a - b) <= tol;
}

bool block_is_uniform(
    const poker::ActionState& state,
    const std::vector<float>& tensor,
    int bucket
) {
    const double expected =
        1.0 / static_cast<double>(state.action_count);

    for (int a = 0; a < state.action_count; ++a) {
        const std::size_t idx = state.tensor_index(bucket, a);
        if (!nearly_equal(tensor[idx], expected)) {
            return false;
        }
    }

    return true;
}

bool block_has_positive_regret(
    const poker::ActionState& state,
    const std::vector<float>& regret_sum,
    int bucket
) {
    for (int a = 0; a < state.action_count; ++a) {
        const std::size_t idx = state.tensor_index(bucket, a);
        if (regret_sum[idx] > kTol) {
            return true;
        }
    }

    return false;
}

std::string block_to_string(
    const poker::ActionState& state,
    const std::vector<float>& tensor,
    int bucket
) {
    std::ostringstream out;
    out << "[";

    for (int a = 0; a < state.action_count; ++a) {
        if (a > 0) {
            out << ", ";
        }

        const std::size_t idx = state.tensor_index(bucket, a);
        out << tensor[idx];
    }

    out << "]";
    return out.str();
}

const poker::ActionState& root_action_state(const poker::Game& game) {
    const poker::PublicNode& root = game.node(game.root);

    if (root.type != poker::PublicNodeType::Action) {
        throw std::runtime_error("Root is not an action node.");
    }

    if (root.action_state_index < 0 ||
        root.action_state_index >= game.num_action_states()) {
        throw std::runtime_error("Root has invalid action_state_index.");
    }

    return game.action_state(root.action_state_index);
}

struct RootStrategySnapshot {
    std::vector<float> current;
    std::vector<float> average;
    std::vector<float> regret;
};

RootStrategySnapshot run_gpu(
    const poker::Game& game,
    int iterations
) {
    poker::GpuCfrConfig config;
    config.num_players = 2;
    config.threads_per_block = 256;
    config.use_cfr_plus = false;
    config.linear_averaging = false;
    config.synchronize_each_iteration = true;
    config.terminal_mode = poker::GpuTerminalMode::HostPrecomputed;
    poker::GpuCfrSolver solver(
        game,
        config
    );
    solver.set_terminal_values(game.terminal_value_p0);
    solver.run_iterations(iterations);

    RootStrategySnapshot snapshot;
    snapshot.current = solver.current_strategy();
    snapshot.average = solver.average_strategy();
    snapshot.regret = solver.regret_sum();

    if (solver.stats().iterations_run != iterations) {
        throw std::runtime_error("GPU solver did not run expected iterations.");
    }

    return snapshot;
}

RootStrategySnapshot run_cpu(
    const poker::Game& game,
    int iterations
) {
    poker::CfrConfig config;
    config.num_players = 2;
    config.use_cfr_plus = false;
    config.linear_averaging = false;
    config.simultaneous_updates = true;

    poker::TerminalValueProvider terminal_values;
    poker::CpuCfrSolver solver(game, terminal_values, config);

    solver.run_iterations(iterations);

    RootStrategySnapshot snapshot;
    snapshot.current = solver.current_strategy();
    snapshot.average = solver.average_strategy();
    snapshot.regret = solver.regret_sum();

    if (solver.stats().iterations_run != iterations) {
        throw std::runtime_error("CPU solver did not run expected iterations.");
    }

    return snapshot;
}

void require(
    bool condition,
    const std::string& message
) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// -----------------------------------------------------------------------------
// Test 1:
// Confirm the test game is capable of producing nonuniform root strategy on CPU.
//
// This prevents false positives. If CPU itself is uniform, the test game is not
// useful for diagnosing the GPU.
// -----------------------------------------------------------------------------

void test_cpu_root_has_learning_signal() {
    const poker::Game game = make_test_game();
    const poker::ActionState& root_state = root_action_state(game);

    const int iterations = 50;
    const RootStrategySnapshot cpu = run_cpu(game, iterations);

    int buckets_with_positive_regret = 0;
    int buckets_with_nonuniform_current = 0;
    int buckets_with_nonuniform_average = 0;

    for (int bucket = 0; bucket < root_state.bucket_count; ++bucket) {
        if (block_has_positive_regret(root_state, cpu.regret, bucket)) {
            ++buckets_with_positive_regret;
        }

        if (!block_is_uniform(root_state, cpu.current, bucket)) {
            ++buckets_with_nonuniform_current;
        }

        if (!block_is_uniform(root_state, cpu.average, bucket)) {
            ++buckets_with_nonuniform_average;
        }
    }

    require(
        buckets_with_positive_regret > 0 ||
        buckets_with_nonuniform_current > 0 ||
        buckets_with_nonuniform_average > 0,
        "CPU root stayed fully uniform/no-regret on this test game. "
        "Use a different test game or more iterations before diagnosing GPU."
    );

    std::cout
        << "[PASS] CPU root has learning signal: "
        << "positive_regret_buckets=" << buckets_with_positive_regret
        << ", nonuniform_current_buckets=" << buckets_with_nonuniform_current
        << ", nonuniform_average_buckets=" << buckets_with_nonuniform_average
        << "\n";
}
void test_gpu_root_regret_inputs() {
    const poker::Game game = make_test_game();

    poker::GpuCfrConfig config;
    config.terminal_mode = poker::GpuTerminalMode::HostPrecomputed;
    config.synchronize_each_iteration = true;

    poker::GpuCfrSolver solver(
        game,
        config
    );

    solver.run_iterations(1);

    const auto action_value = solver.debug_action_value_p0();
    const auto cf_reach = solver.debug_state_bucket_cf_reach();
    const auto own_reach = solver.debug_state_bucket_own_reach();
    const auto state_value = solver.debug_state_bucket_value_p0();
    const auto regret = solver.regret_sum();

    const auto& root = game.node(game.root);
    const auto& state = game.action_state(root.action_state_index);

    int nonzero_action_values = 0;
    int nonzero_cf_reach = 0;
    int nonzero_own_reach = 0;
    int positive_regrets = 0;

    for (int bucket = 0; bucket < state.bucket_count; ++bucket) {
        const std::size_t sb =
            static_cast<std::size_t>(state.state_bucket_offset) +
            static_cast<std::size_t>(bucket);

        if (std::abs(cf_reach[sb]) > 1e-6f) {
            ++nonzero_cf_reach;
        }

        if (std::abs(own_reach[sb]) > 1e-6f) {
            ++nonzero_own_reach;
        }

        for (int a = 0; a < state.action_count; ++a) {
            const std::size_t idx = state.tensor_index(bucket, a);

            if (std::abs(action_value[idx]) > 1e-6f) {
                ++nonzero_action_values;
            }

            if (regret[idx] > 1e-6f) {
                ++positive_regrets;
            }
        }
    }

    std::cout
        << "root nonzero_action_values=" << nonzero_action_values << "\n"
        << "root nonzero_cf_reach=" << nonzero_cf_reach << "\n"
        << "root nonzero_own_reach=" << nonzero_own_reach << "\n"
        << "root positive_regrets=" << positive_regrets << "\n"
        << "gpu last_root_value_p0=" << solver.stats().last_root_value_p0 << "\n";

    if (nonzero_action_values == 0) {
        throw std::runtime_error(
            "Root action values are all zero. Suspect terminal upload, "
            "backward value pass, or action-value computation."
        );
    }

    if (nonzero_cf_reach == 0) {
        throw std::runtime_error(
            "Root counterfactual reach is zero. Suspect reach initialization "
            "or state-bucket reach aggregation."
        );
    }

    if (positive_regrets == 0) {
        throw std::runtime_error(
            "Root action values/reaches exist but regrets are not positive. "
            "Suspect sign convention, player perspective conversion, or "
            "launch_public_update_regrets."
        );
    }
}

// -----------------------------------------------------------------------------
// Test 2:
// Detect whether GPU is stuck in regret-matching fallback.
//
// Regret matching fallback condition:
//   positive_sum == 0 -> sigma = sigma_init, usually uniform.
//
// Therefore, if after many iterations every root bucket has:
//   current strategy uniform
//   and no positive regrets
// then the GPU is almost certainly staying in fallback at root.
// -----------------------------------------------------------------------------

void test_gpu_not_stuck_in_regret_matching_fallback() {
    const poker::Game game = make_test_game();
    const poker::ActionState& root_state = root_action_state(game);

    const int iterations = 50;
    const RootStrategySnapshot gpu = run_gpu(game, iterations);

    int fallback_like_buckets = 0;
    int positive_regret_buckets = 0;
    int nonuniform_current_buckets = 0;

    for (int bucket = 0; bucket < root_state.bucket_count; ++bucket) {
        const bool current_uniform =
            block_is_uniform(root_state, gpu.current, bucket);

        const bool has_positive_regret =
            block_has_positive_regret(root_state, gpu.regret, bucket);

        if (has_positive_regret) {
            ++positive_regret_buckets;
        }

        if (!current_uniform) {
            ++nonuniform_current_buckets;
        }

        if (current_uniform && !has_positive_regret) {
            ++fallback_like_buckets;
        }
    }

    require(
        positive_regret_buckets > 0 || nonuniform_current_buckets > 0,
        "GPU appears stuck in regret-matching fallback at the root: "
        "current_strategy is uniform and regret_sum has no positive root regrets."
    );

    std::cout
        << "[PASS] GPU is not fully stuck in root regret-matching fallback: "
        << "positive_regret_buckets=" << positive_regret_buckets
        << ", nonuniform_current_buckets=" << nonuniform_current_buckets
        << ", fallback_like_buckets=" << fallback_like_buckets
        << "\n";
}

// -----------------------------------------------------------------------------
// Test 3:
// Detect average-strategy accumulation/normalization bug.
//
// Failure signature:
//   current_strategy is nonuniform
//   average_strategy is uniform
//
// That means regret matching is working, but average accumulation or average
// normalization is not recording/normalizing the strategy correctly.
// -----------------------------------------------------------------------------

void test_gpu_average_not_uniform_when_current_is_nonuniform() {
    const poker::Game game = make_test_game();
    const poker::ActionState& root_state = root_action_state(game);

    const int iterations = 50;
    const RootStrategySnapshot gpu = run_gpu(game, iterations);

    int suspicious_buckets = 0;

    for (int bucket = 0; bucket < root_state.bucket_count; ++bucket) {
        const bool current_uniform =
            block_is_uniform(root_state, gpu.current, bucket);

        const bool average_uniform =
            block_is_uniform(root_state, gpu.average, bucket);

        if (!current_uniform && average_uniform) {
            ++suspicious_buckets;

            std::cerr
                << "Suspicious bucket " << bucket << "\n"
                << "  current = "
                << block_to_string(root_state, gpu.current, bucket)
                << "\n"
                << "  average = "
                << block_to_string(root_state, gpu.average, bucket)
                << "\n"
                << "  regret  = "
                << block_to_string(root_state, gpu.regret, bucket)
                << "\n";
        }
    }

    require(
        suspicious_buckets == 0,
        "GPU current strategy is nonuniform while average strategy is uniform. "
        "This points to average-strategy accumulation or normalization, not "
        "regret matching."
    );

    std::cout
        << "[PASS] No root bucket has nonuniform current strategy with uniform average strategy.\n";
}
void test_flatten_contains_root_action_edges() {
    const poker::Game game = make_test_game();
    const poker::FlatPublicGame flat = poker::flatten_public_game_for_gpu(game);

    const auto& root = game.node(game.root);
    const auto& root_state = game.action_state(root.action_state_index);

    int root_action_edges = 0;

    for (std::size_t i = 0; i < flat.action_edge_parent.size(); ++i) {
        if (flat.action_edge_parent[i] == game.root) {
            ++root_action_edges;

            std::cout
                << "root action edge i=" << i
                << " child=" << flat.action_edge_child[i]
                << " state=" << flat.action_edge_state[i]
                << " local_action=" << flat.action_edge_local_action[i]
                << "\n";
        }
    }

    std::cout
        << "root.edge_count=" << root.edge_count << "\n"
        << "root_state.action_count=" << root_state.action_count << "\n"
        << "flat root action edges=" << root_action_edges << "\n";

    if (root_action_edges != root_state.action_count) {
        throw std::runtime_error(
            "Flattened GPU action-edge table does not contain all root actions."
        );
    }
    std::cout << "[PASS] FlatPublicGame contains root action edges" << std::endl;
}

// -----------------------------------------------------------------------------
// Test 4:
// CPU/GPU root classification comparison.
//
// This does not require exact float equality. It checks whether GPU at least
// shows the same kind of root signal as CPU.
// -----------------------------------------------------------------------------

void test_gpu_root_signal_matches_cpu_classification() {
    const poker::Game game = make_test_game();
    const poker::ActionState& root_state = root_action_state(game);

    const int iterations = 50;
    const RootStrategySnapshot cpu = run_cpu(game, iterations);
    const RootStrategySnapshot gpu = run_gpu(game, iterations);

    int cpu_signal_buckets = 0;
    int gpu_signal_buckets = 0;

    for (int bucket = 0; bucket < root_state.bucket_count; ++bucket) {
        const bool cpu_signal =
            block_has_positive_regret(root_state, cpu.regret, bucket) ||
            !block_is_uniform(root_state, cpu.current, bucket) ||
            !block_is_uniform(root_state, cpu.average, bucket);

        const bool gpu_signal =
            block_has_positive_regret(root_state, gpu.regret, bucket) ||
            !block_is_uniform(root_state, gpu.current, bucket) ||
            !block_is_uniform(root_state, gpu.average, bucket);

        if (cpu_signal) {
            ++cpu_signal_buckets;
        }

        if (gpu_signal) {
            ++gpu_signal_buckets;
        }
    }

    require(
        cpu_signal_buckets > 0,
        "CPU produced no root signal; test game is not diagnostic."
    );

    require(
        gpu_signal_buckets > 0,
        "CPU produced root learning signal, but GPU produced none. "
        "This strongly suggests GPU terminal/value/reach/regret update is broken "
        "or terminal values were not uploaded."
    );

    std::cout
        << "[PASS] GPU has at least some root learning signal like CPU: "
        << "cpu_signal_buckets=" << cpu_signal_buckets
        << ", gpu_signal_buckets=" << gpu_signal_buckets
        << "\n";
}

} // namespace

int main() {
    try {
        test_cpu_root_has_learning_signal();
        test_gpu_root_regret_inputs();
        test_gpu_not_stuck_in_regret_matching_fallback();
        test_flatten_contains_root_action_edges();
        test_gpu_average_not_uniform_when_current_is_nonuniform();
        test_gpu_root_signal_matches_cpu_classification();

        std::cout << "\nAll GPU strategy diagnostic tests passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "\n[FAIL] " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}