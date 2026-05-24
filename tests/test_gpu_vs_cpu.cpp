#include "kuhn_builder.hpp"
#include "cfr_cpu.hpp"
#include "cfr_gpu.hpp"
#include "exploitability.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float kProbTol = 1e-4f;

// GPU and CPU may not match bit-for-bit because of different update order,
// float arithmetic, atomics, and kernel scheduling. Keep this loose at first.
constexpr double kStrategyL1TolEarly = 0.40;
constexpr double kExploitabilityTol = 0.10;

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

void check_finite_vector(
    const std::vector<float>& values,
    const std::string& name
) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) {
            std::ostringstream oss;
            oss << name << "[" << i << "] is not finite: " << values[i];
            throw std::runtime_error(oss.str());
        }
    }
}

void check_strategy_distribution(
    const poker::Game& game,
    const std::vector<float>& strategy,
    const std::string& name
) {
    check(
        static_cast<int>(strategy.size()) == game.num_q(),
        name + " size does not match game.num_q()."
    );

    check_finite_vector(strategy, name);

    for (const poker::InfoSet& infoset : game.infosets) {
        double sum = 0.0;

        for (int q : infoset.q_indices) {
            const float p = strategy[q];

            if (p < -kProbTol || p > 1.0f + kProbTol) {
                std::ostringstream oss;
                oss << name
                    << " has invalid probability at q=" << q
                    << ": " << p;
                throw std::runtime_error(oss.str());
            }

            sum += p;
        }

        std::ostringstream msg;
        msg << name
            << " probabilities do not sum to 1 at infoset "
            << infoset.id;

        check_near(sum, 1.0, 1e-3, msg.str());
    }
}

double l1_distance(
    const std::vector<float>& a,
    const std::vector<float>& b
) {
    check(a.size() == b.size(), "Cannot compare vectors with different sizes.");

    double total = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i) {
        total += std::abs(
            static_cast<double>(a[i]) - static_cast<double>(b[i])
        );
    }

    return total;
}

double max_abs_difference(
    const std::vector<float>& a,
    const std::vector<float>& b
) {
    check(a.size() == b.size(), "Cannot compare vectors with different sizes.");

    double result = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i) {
        result = std::max(
            result,
            std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]))
        );
    }

    return result;
}

std::vector<float> run_cpu(
    const poker::Game& game,
    int iterations,
    double* elapsed_ms
) {
    poker::CfrConfig config;
    config.use_cfr_plus = false;
    config.linear_averaging = false;

    poker::CpuCfrSolver solver(game, config);

    const auto start = std::chrono::steady_clock::now();
    solver.run_iterations(iterations);
    const auto end = std::chrono::steady_clock::now();

    *elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    check(
        solver.stats().iterations_run == iterations,
        "CPU solver ran the wrong number of iterations."
    );

    return solver.average_strategy();
}

std::vector<float> run_gpu(
    const poker::Game& game,
    int iterations,
    double* elapsed_ms
) {
    poker::GpuCfrConfig config;
    config.use_cfr_plus = false;
    config.linear_averaging = false;

    // Turn this on while debugging kernel errors.
    // Turn it off for timing.
    config.synchronize_each_iteration = true;

    poker::GpuCfrSolver solver(game, config);

    const auto start = std::chrono::steady_clock::now();
    solver.run_iterations(iterations);
    const auto end = std::chrono::steady_clock::now();

    *elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    check(
        solver.stats().iterations_run == iterations,
        "GPU solver ran the wrong number of iterations."
    );

    return solver.average_strategy();
}

void test_initial_gpu_strategy_is_valid() {
    const poker::Game game = poker::build_kuhn_game();
    auto config = poker::GpuCfrConfig{};
    config.synchronize_each_iteration = true;
    poker::GpuCfrSolver gpu(game, config);

    const std::vector<float> current = gpu.current_strategy();
    const std::vector<float> average = gpu.average_strategy();

    check_strategy_distribution(game, current, "initial GPU current strategy");
    check_strategy_distribution(game, average, "initial GPU average strategy");

    std::cout << "[pass] test_initial_gpu_strategy_is_valid\n";
}

void test_gpu_one_iteration_is_valid() {
    const poker::Game game = poker::build_kuhn_game();
    auto config = poker::GpuCfrConfig{};
    config.synchronize_each_iteration = true;
    poker::GpuCfrSolver gpu(game, config);
    gpu.run_one_iteration();

    check(
        gpu.stats().iterations_run == 1,
        "GPU solver should report exactly one iteration."
    );

    const std::vector<float> current = gpu.current_strategy();
    const std::vector<float> average = gpu.average_strategy();

    check_strategy_distribution(game, current, "GPU current strategy after one iteration");
    check_strategy_distribution(game, average, "GPU average strategy after one iteration");

    std::cout << "[pass] test_gpu_one_iteration_is_valid\n";
}

void test_cpu_gpu_average_strategies_are_close() {
    const poker::Game game = poker::build_kuhn_game();

    const int iterations = 500;

    double cpu_ms = 0.0;
    double gpu_ms = 0.0;

    const std::vector<float> cpu_avg = run_cpu(game, iterations, &cpu_ms);
    const std::vector<float> gpu_avg = run_gpu(game, iterations, &gpu_ms);

    check_strategy_distribution(game, cpu_avg, "CPU average strategy");
    check_strategy_distribution(game, gpu_avg, "GPU average strategy");

    const double l1 = l1_distance(cpu_avg, gpu_avg);
    const double max_diff = max_abs_difference(cpu_avg, gpu_avg);

    std::cout << "[info] CPU time ms: " << cpu_ms << "\n";
    std::cout << "[info] GPU time ms: " << gpu_ms << "\n";
    std::cout << "[info] strategy L1 distance: " << l1 << "\n";
    std::cout << "[info] strategy max abs diff: " << max_diff << "\n";

    check(
        l1 < kStrategyL1TolEarly,
        "CPU and GPU average strategies are too different."
    );

    std::cout << "[pass] test_cpu_gpu_average_strategies_are_close\n";
}

void test_cpu_gpu_exploitability_is_close() {
    const poker::Game game = poker::build_kuhn_game();

    const int iterations = 500;

    double cpu_ms = 0.0;
    double gpu_ms = 0.0;

    const std::vector<float> cpu_avg = run_cpu(game, iterations, &cpu_ms);
    const std::vector<float> gpu_avg = run_gpu(game, iterations, &gpu_ms);

    const poker::ExploitabilityResult cpu_result =
        poker::compute_exploitability(game, cpu_avg);

    const poker::ExploitabilityResult gpu_result =
        poker::compute_exploitability(game, gpu_avg);

    check(
        std::isfinite(cpu_result.exploitability),
        "CPU exploitability should be finite."
    );

    check(
        std::isfinite(gpu_result.exploitability),
        "GPU exploitability should be finite."
    );

    std::cout << "[info] CPU exploitability: " << cpu_result.exploitability << "\n";
    std::cout << "[info] GPU exploitability: " << gpu_result.exploitability << "\n";
    std::cout << "[info] CPU value P0: " << cpu_result.strategy_value_p0 << "\n";
    std::cout << "[info] GPU value P0: " << gpu_result.strategy_value_p0 << "\n";

    check(
        gpu_result.exploitability < 0.15,
        "GPU CFR exploitability is too high after training."
    );

    check(
        std::abs(cpu_result.exploitability - gpu_result.exploitability)
            < kExploitabilityTol,
        "CPU and GPU exploitability differ too much."
    );

    std::cout << "[pass] test_cpu_gpu_exploitability_is_close\n";
}

std::vector<float> make_forced_nonuniform_strategy(const poker::Game& game) {
    std::vector<float> strategy(
        static_cast<std::size_t>(game.num_q()),
        0.0f
    );

    for (const poker::InfoSet& infoset : game.infosets) {
        check(
            infoset.q_indices.size() == 2,
            "This diagnostic assumes two actions per Kuhn infoset."
        );

        const int q0 = infoset.q_indices[0];
        const int q1 = infoset.q_indices[1];

        // Deliberately asymmetric, but still valid.
        // Alternate by infoset id so all infosets are not identical.
        if ((infoset.id % 2) == 0) {
            strategy[q0] = 0.80f;
            strategy[q1] = 0.20f;
        } else {
            strategy[q0] = 0.30f;
            strategy[q1] = 0.70f;
        }
    }

    check_strategy_distribution(game, strategy, "forced nonuniform strategy");
    return strategy;
}

void accumulate_expected_own_reach_recursive(
    const poker::Game& game,
    const std::vector<float>& strategy,
    int node_id,
    float reach_p0,
    float reach_p1,
    std::vector<float>& expected_weight_sum
) {
    const poker::Node& node = game.node(node_id);

    if (node.terminal) {
        return;
    }

    if (node.player == poker::Player::Chance) {
        for (int child_id : node.children) {
            // Match CpuCfrSolver: chance does not change reach_p0/reach_p1.
            accumulate_expected_own_reach_recursive(
                game,
                strategy,
                child_id,
                reach_p0,
                reach_p1,
                expected_weight_sum
            );
        }

        return;
    }

    check(
        node.player == poker::Player::P0 ||
        node.player == poker::Player::P1,
        "Unexpected nonterminal node player."
    );

    const poker::InfoSet& infoset = game.infoset(node.infoset);

    const float owner_reach = (node.player == poker::Player::P0) ? reach_p0 : reach_p1;

    expected_weight_sum[static_cast<std::size_t>(infoset.id)] += owner_reach;

    for (int local_action = 0;
         local_action < static_cast<int>(node.children.size());
         ++local_action) {
        const int q = infoset.q_indices[local_action];
        const float action_prob = strategy[q];

        float next_reach_p0 = reach_p0;
        float next_reach_p1 = reach_p1;

        if (node.player == poker::Player::P0) {
            next_reach_p0 *= action_prob;
        } else {
            next_reach_p1 *= action_prob;
        }

        accumulate_expected_own_reach_recursive(
            game,
            strategy,
            node.children[local_action],
            next_reach_p0,
            next_reach_p1,
            expected_weight_sum
        );
    }
}

std::vector<float> expected_cpu_style_own_reach_weights(
    const poker::Game& game,
    const std::vector<float>& strategy
) {
    std::vector<float> expected(
        static_cast<std::size_t>(game.num_infosets()),
        0.0f
    );

    accumulate_expected_own_reach_recursive(
        game,
        strategy,
        game.root,
        1.0f,
        1.0f,
        expected
    );

    return expected;
}

void test_gpu_average_accumulation_uses_acting_player_reach() {
    const poker::Game game = poker::build_kuhn_game();

    poker::GpuCfrConfig config;
    config.use_cfr_plus = false;
    config.linear_averaging = false;
    config.synchronize_each_iteration = true;

    poker::GpuCfrSolver gpu(game, config);

    const std::vector<float> forced =
        make_forced_nonuniform_strategy(game);

    gpu.set_current_strategy_for_testing(forced);

    const std::vector<float> before = gpu.current_strategy();
    check_strategy_distribution(game, before, "forced GPU current strategy");

    const double forced_l1 = l1_distance(before, forced);
    check_near(
        forced_l1,
        0.0,
        1e-5,
        "GPU current strategy was not overwritten correctly"
    );

    gpu.run_one_iteration();

    const std::vector<float> actual_weight_sum = gpu.debug_strategy_weight_sum();

    const std::vector<float> expected_weight_sum = expected_cpu_style_own_reach_weights(game, forced);

    check(
        actual_weight_sum.size() == expected_weight_sum.size(),
        "strategy_weight_sum size mismatch."
    );

    std::cout << "[info] average accumulation reach diagnostic\n";

    for (const poker::InfoSet& infoset : game.infosets) {
        const int h = infoset.id;

        std::cout
            << "[info] infoset=" << h
            << " player=" << poker::to_string(infoset.player)
            << " card=" << poker::to_string(infoset.private_card)
            << " history=\"" << infoset.public_history << "\""
            << " expected_own_weight=" << expected_weight_sum[h]
            << " actual_gpu_weight=" << actual_weight_sum[h]
            << "\n";

        std::ostringstream msg;
        msg << "GPU average-strategy denominator is not using "
            << "CPU-style acting-player reach at infoset " << h;

        check_near(
            actual_weight_sum[h],
            expected_weight_sum[h],
            1e-4,
            msg.str()
        );
    }

    const std::vector<float> strategy_sum = gpu.debug_strategy_sum();

    for (const poker::InfoSet& infoset : game.infosets) {
        const int h = infoset.id;

        for (int q : infoset.q_indices) {
            const double expected_numerator =
                static_cast<double>(expected_weight_sum[h]) *
                static_cast<double>(forced[q]);

            std::ostringstream msg;
            msg << "GPU strategy_sum[q] mismatch at q=" << q
                << " infoset=" << h;

            check_near(
                strategy_sum[q],
                expected_numerator,
                1e-4,
                msg.str()
            );
        }
    }

    std::cout
        << "[pass] test_gpu_average_accumulation_uses_acting_player_reach\n";
}


void run_all_tests() {
    test_initial_gpu_strategy_is_valid();
    test_gpu_one_iteration_is_valid();
    test_gpu_average_accumulation_uses_acting_player_reach();
    test_cpu_gpu_average_strategies_are_close();
    test_cpu_gpu_exploitability_is_close();
}

} // namespace

int main() {
    try {
        run_all_tests();
    } catch (const std::exception& e) {
        std::cerr << "[fail] " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "[pass] all GPU-vs-CPU CFR tests passed\n";
    return EXIT_SUCCESS;
}