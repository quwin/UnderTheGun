#include "cfr_cpu.hpp"
#include "exploitability.hpp"

#include "cfr_gpu.hpp"

#include "holdem/subgame_builder.hpp"
#include "holdem/subgame_config.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>

namespace {
poker::Board make_test_board() {
    return poker::Board{
        {
            phevaluator::Card("As"),
            phevaluator::Card("7h"),
            phevaluator::Card("Jh"),
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
    config.collapse_all_in_runouts_to_ev = true;
    config.p0_range = make_tiny_p0_range();
    config.p1_range = make_tiny_p1_range();
    config.betting_abstraction = poker::holdem::make_standard_abstraction();

    return config;
}

using Clock = std::chrono::steady_clock;

struct BenchResult {
    std::string name;
    double seconds = 0.0;
    double iters_per_sec = 0.0;
    std::vector<float> avg_strategy;
};

double max_abs_diff(
    const std::vector<float>& a,
    const std::vector<float>& b
) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Strategy sizes differ.");
    }

    double diff = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        diff = std::max(diff, std::abs(
            static_cast<double>(a[i]) - static_cast<double>(b[i])
        ));
    }
    return diff;
}

BenchResult run_cpu_benchmark(
    const poker::Game& game,
    int iterations
) {
    const poker::CfrConfig config;
    const poker::TerminalValueProvider terminal_values;
    poker::CpuCfrSolver solver(game,terminal_values, config);

    const auto t0 = Clock::now();
    solver.run_iterations(iterations);
    const auto t1 = Clock::now();

    BenchResult r;
    r.name = "CPU";
    r.seconds = std::chrono::duration<double>(t1 - t0).count();
    r.iters_per_sec = iterations / r.seconds;
    r.avg_strategy = solver.average_strategy();
    return r;
}

BenchResult run_gpu_benchmark(
    const poker::Game& game,
    int iterations
) {
    poker::GpuCfrConfig config;
    config.synchronize_each_iteration = false; // true only while debugging
    poker::GpuCfrSolver solver(game, config);
    const auto t0 = Clock::now();
    solver.run_iterations(iterations);
    // average_strategy() performs a device-to-host copy, forcing completion.
    std::vector<float> avg = solver.average_strategy();

    const auto t1 = Clock::now();

    BenchResult r;
    r.name = "GPU";
    r.seconds = std::chrono::duration<double>(t1 - t0).count();
    r.iters_per_sec = iterations / r.seconds;
    r.avg_strategy = std::move(avg);
    return r;
}

void print_result(const BenchResult& r) {
    std::cout
        << std::left << std::setw(8) << r.name
        << " time=" << std::setw(10) << r.seconds << "s"
        << " iter/s=" << std::setw(12) << r.iters_per_sec
        << "\n";
}

} // namespace

int main() {
    try {
        constexpr int iterations = 100;

        const poker::holdem::HoldemSubgameConfig config = make_test_config();
        const poker::Game game = poker::holdem::HoldemSubgameBuilder(config).build();
        game.print_game_memory_usage();
        std::cout << "Benchmarking " << iterations << " CFR iterations\n";
        std::cout << "Nodes: " << game.num_nodes() << "\n";

        const BenchResult gpu = run_gpu_benchmark(game, iterations);
        print_result(gpu);

        // const BenchResult cpu = run_cpu_benchmark(game, iterations);
        // print_result(cpu);


        // std::cout << "GPU speed relative to CPU: " << cpu.seconds / gpu.seconds << "x\n";
        // std::cout << "Max avg-strategy abs diff: " << max_abs_diff(cpu.avg_strategy, gpu.avg_strategy) << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << "\n";
        return 1;
    }
}