#include "cfr_cpu.hpp"
#include "exploitability.hpp"
#include "cfr_gpu.hpp"

#include "holdem/subgame_builder.hpp"
#include "holdem/subgame_config.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {
poker::Board make_test_board() {
    return poker::Board{
        {
            phevaluator::Card("As"),
            phevaluator::Card("7s"),
            phevaluator::Card("Js"),
            // phevaluator::Card("4s"),
            // phevaluator::Card("4d"),
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
poker::Range small_symmetric_range() {
    poker::Range range;
    range.clear();

    const std::vector<std::string> small_range = {
        "AA", "KK", "QQ", "JJ", "TT",
        "AKs", "AQs", "AJs", "KQs", "QJs",
        "AKo", "AQo", "KQo"
    };

    for (const std::string& token : small_range) {
        const char r1 = token[0];
        const char r2 = token[1];

        const bool pair = (r1 == r2);
        const char suffix =
            token.size() == 3
                ? static_cast<char>(std::tolower(static_cast<unsigned char>(token[2])))
                : '\0';
        const std::vector<phevaluator::Card> c1 = {
            phevaluator::Card(std::string{r1, 'c'}),
            phevaluator::Card(std::string{r1, 'd'}),
            phevaluator::Card(std::string{r1, 'h'}),
            phevaluator::Card(std::string{r1, 's'})
        };
        const std::vector<phevaluator::Card> c2 = {
            phevaluator::Card(std::string{r2, 'c'}),
            phevaluator::Card(std::string{r2, 'd'}),
            phevaluator::Card(std::string{r2, 'h'}),
            phevaluator::Card(std::string{r2, 's'})
        };
        for (phevaluator::Card a : c1) {
            for (phevaluator::Card b : c2) {
                if (a == b) {
                    continue;
                }
                const bool suited = a.describeSuit() == b.describeSuit();
                if (pair) {
                    if (static_cast<int>(a) >= static_cast<int>(b)) {
                        continue;
                    }
                } else {
                    if (suffix == 's' && !suited) {
                        continue;
                    }
                    if (suffix == 'o' && suited) {
                        continue;
                    }
                }
                range.set_weight(poker::make_hand(a, b), 1.0f);
            }
        }
    }

    return range;
}


poker::holdem::HoldemSubgameConfig make_test_config() {
    poker::holdem::HoldemSubgameConfig config;

    config.board = make_test_board();
    config.pot_size = 100;
    config.effective_stack = 2000;
    config.player_to_act = poker::Player::P0;
    config.collapse_all_in_runouts_to_ev = true;
    config.p0_range = small_symmetric_range();
    config.p1_range = small_symmetric_range();
    config.p0_range = make_tiny_p0_range();
    config.p1_range = make_tiny_p1_range();
    config.betting_abstraction = poker::holdem::make_standard_abstraction();
    config.board_abstraction = poker::holdem::make_isomorphic_board_abstraction(config.p0_range,config.p1_range);
    config.terminal_mode = poker::TerminalMode::RecordComputed;
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
void print_result(const BenchResult& r) {
    std::cout
        << std::left << std::setw(8) << r.name
        << " time=" << std::setw(6) << r.seconds << "s"
        << " iter/s=" << std::setw(12) << r.iters_per_sec
        << "\n";
}

BenchResult run_cpu_benchmark(
    poker::holdem::HoldemSubgameConfig& build_config,
    int iterations
) {
    build_config.terminal_mode = poker::TerminalMode::ValuePrecomputed;
    const auto t0 = Clock::now();
    const poker::Game game_tree = poker::holdem::HoldemSubgameBuilder(build_config).build();

    constexpr poker::CfrConfig config;
    const poker::TerminalValueProvider terminal_values;
    poker::CpuCfrSolver solver(game_tree,terminal_values, config);

    solver.run_iterations(iterations);
    const auto t1 = Clock::now();

    BenchResult r;
    r.name = "CPU";
    r.seconds = std::chrono::duration<double>(t1 - t0).count();
    r.iters_per_sec = iterations / r.seconds;
    r.avg_strategy = solver.average_strategy();
    return r;
}

BenchResult run_gpu_tree_benchmark(
    const poker::holdem::HoldemSubgameConfig& build_config,
    int iterations
) {
    poker::GpuCfrConfig config;
    const auto t0 = Clock::now();
    const poker::Game game_tree = poker::holdem::HoldemSubgameBuilder(build_config).build();
    game_tree.print_game_memory_usage();
    std::cout << "Nodes: " << game_tree.num_nodes() << "\n";
    poker::GpuCfrSolver solver(game_tree, config);
    solver.run_iterations(iterations);
    // average_strategy() performs a device-to-host copy, forcing completion.
    std::vector<float> avg = solver.average_strategy();

    const auto t1 = Clock::now();

    BenchResult r1;
    r1.name = "GPU Tree build";
    r1.seconds = std::chrono::duration<double>(t1 - t0).count();
    r1.iters_per_sec = iterations / r1.seconds;
    r1.avg_strategy = std::move(avg);

    return r1;
}

// BenchResult run_gpu_flat_benchmark(
//     const poker::holdem::HoldemSubgameConfig& build_config,
//     int iterations
// ) {
//     poker::GpuCfrConfig config;
//     const auto t2 = Clock::now();
//     const poker::FlatGpuBuildResult flat_game = poker::holdem::HoldemSubgameBuilder(build_config).build_flat_for_gpu();
//     poker::GpuCfrSolver flat_solver(flat_game, config);
//     flat_solver.run_iterations(iterations);
//     // average_strategy() performs a device-to-host copy, forcing completion.
//     std::vector<float> flat_avg = flat_solver.average_strategy();
//     const auto t3 = Clock::now();
//
//     BenchResult r2;
//     r2.name = "GPU Flat build";
//     r2.seconds = std::chrono::duration<double>(t3 - t2).count();
//     r2.iters_per_sec = iterations / r2.seconds;
//     r2.avg_strategy = std::move(flat_avg);
//
//     return r2;
// }

} // namespace

int main() {
    try {
        constexpr int iterations = 100;
        poker::holdem::HoldemSubgameConfig config = make_test_config();
        std::cout << "Expected Memory: " << config.memoryEstimate() << " bytes\n";
        std::cout << "Benchmarking " << iterations << " CFR iterations\n";
        // const BenchResult gpu_flat = run_gpu_flat_benchmark(config, iterations);
        // print_result(gpu_flat);

        const BenchResult gpu_tree = run_gpu_tree_benchmark(config, iterations);
        print_result(gpu_tree);

        const BenchResult cpu = run_cpu_benchmark(config, iterations);
        print_result(cpu);

        std::cout << "GPU speed relative to CPU: " << cpu.seconds / gpu_tree.seconds << "x\n";
        std::cout << "Max avg-strategy abs diff: " << max_abs_diff(cpu.avg_strategy, gpu_tree.avg_strategy) << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << "\n";
        return 1;
    }
}