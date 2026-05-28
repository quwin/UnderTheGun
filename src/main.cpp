#include "cfr_cpu.hpp"
#include "exploitability.hpp"

#ifdef UNDER_THE_GUN_ENABLE_CUDA
#include "cfr_gpu.hpp"
#endif

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kuhn_builder.hpp"

namespace {

struct ProgramOptions {
    int iterations = 1000;
    bool use_gpu = false;
    bool print_strategy = false;
};

void print_usage(const char* program_name) {
    std::cout
        << "Usage:\n"
        << "  " << program_name << " [--iterations N] [--gpu] [--print-strategy]\n\n"
        << "Examples:\n"
        << "  " << program_name << "\n"
        << "  " << program_name << " --iterations 100000\n"
        << "  " << program_name << " --gpu --iterations 100000\n";
}

ProgramOptions parse_args(int argc, char** argv) {
    ProgramOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }

        if (arg == "--gpu") {
            options.use_gpu = true;
            continue;
        }

        if (arg == "--print-strategy") {
            options.print_strategy = true;
            continue;
        }

        if (arg == "--iterations") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--iterations requires a value.");
            }

            options.iterations = std::stoi(argv[++i]);

            if (options.iterations < 0) {
                throw std::invalid_argument("iterations must be nonnegative.");
            }

            continue;
        }

        throw std::invalid_argument("Unknown argument: " + arg);
    }

    return options;
}

void print_game_summary(const poker::Game& game) {
    int terminal_count = 0;
    int chance_count = 0;
    int p0_count = 0;
    int p1_count = 0;

    for (const poker::Node& node : game.nodes) {
        if (node.terminal) {
            ++terminal_count;
        }

        if (node.player == poker::Player::Chance) {
            ++chance_count;
        } else if (node.player == poker::Player::P0) {
            ++p0_count;
        } else if (node.player == poker::Player::P1) {
            ++p1_count;
        }
    }

    std::cout << "Game summary\n";
    std::cout << "------------\n";
    std::cout << "Nodes:      " << game.num_nodes() << "\n";
    std::cout << "Terminals:  " << terminal_count << "\n";
    std::cout << "Infosets:   " << game.num_infosets() << "\n";
    std::cout << "Q entries:  " << game.num_q() << "\n";
    std::cout << "Max depth:  " << game.max_depth << "\n";
    std::cout << "Chance:     " << chance_count << " nodes\n";
    std::cout << "P0:         " << p0_count << " nodes\n";
    std::cout << "P1:         " << p1_count << " nodes\n";
    std::cout << "\n";
}

void print_strategy(
    const poker::Game& game,
    const std::vector<float>& strategy,
    const std::string& title
) {
    if (static_cast<int>(strategy.size()) != game.num_q()) {
        throw std::invalid_argument("Strategy size does not match game.num_q().");
    }

    std::cout << title << "\n";
    std::cout << std::string(title.size(), '-') << "\n";

    std::cout << std::fixed << std::setprecision(6);

    for (const poker::InfoSet& infoset : game.infosets) {
        std::cout
            << "Infoset " << infoset.id
            << " | player=" << poker::to_string(infoset.player)
            << " | history=\"" << infoset.key << "\"\n";

        for (int local_action = 0;
             local_action < static_cast<int>(infoset.actions.size());
             ++local_action) {
            const int q = infoset.q_indices[local_action];

            std::cout
                << "  "
                << poker::to_string(infoset.actions[local_action])
                << ": "
                << strategy[q]
                << "\n";
        }
    }

    std::cout << "\n";
}

void print_exploitability_report(
    const poker::ExploitabilityResult& result
) {
    std::cout << std::fixed << std::setprecision(8);

    std::cout << "Exploitability report\n";
    std::cout << "---------------------\n";
    std::cout << "Strategy value P0:       " << result.strategy_value_p0 << "\n";
    std::cout << "Best response value P0:  " << result.best_response_value_p0 << "\n";
    std::cout << "Best response value P1:  " << result.best_response_value_p1 << "\n";
    std::cout << "P0 exploitability:       " << result.p0_exploitability << "\n";
    std::cout << "P1 exploitability:       " << result.p1_exploitability << "\n";
    std::cout << "Exploitability:          " << result.exploitability << "\n";
    std::cout << "\n";
}

std::vector<float> run_cpu_solver(
    const poker::Game& game,
    int iterations
) {
    std::cout << "Running CPU CFR for " << iterations << " iterations...\n";

    poker::CfrConfig config;
    config.use_cfr_plus = false;
    config.linear_averaging = false;

    poker::CpuCfrSolver solver(game, config);
    solver.run_iterations(iterations);

    std::cout << "CPU iterations run: " << solver.stats().iterations_run << "\n";
    std::cout << "CPU root value P0:  " << solver.stats().last_root_value_p0 << "\n";
    std::cout << "\n";

    return solver.average_strategy();
}

#ifdef UNDER_THE_GUN_ENABLE_CUDA

std::vector<float> run_gpu_solver(
    const poker::Game& game,
    int iterations
) {
    std::cout << "Running GPU CFR for " << iterations << " iterations...\n";

    poker::GpuCfrConfig config;
    config.use_cfr_plus = false;
    config.linear_averaging = false;
    config.synchronize_each_iteration = true;

    poker::GpuCfrSolver solver(game, config);
    solver.run_iterations(iterations);

    std::cout << "GPU iterations run: " << solver.stats().iterations_run << "\n";
    std::cout << "GPU root value P0:  " << solver.stats().last_root_value_p0 << "\n";
    std::cout << "\n";

    return solver.average_strategy();
}

#endif

} // namespace

int main(int argc, char** argv) {
    try {
        const ProgramOptions options = parse_args(argc, argv);

        poker::Game game = poker::build_kuhn_game();

        print_game_summary(game);

        std::vector<float> average_strategy;

        if (options.use_gpu) {
#ifdef UNDER_THE_GUN_ENABLE_CUDA
            average_strategy = run_gpu_solver(game, options.iterations);
#else
            std::cerr
                << "This executable was built without CUDA support.\n"
                << "Rebuild with UNDER_THE_GUN_ENABLE_CUDA enabled, or run without --gpu.\n";

            return 1;
#endif
        } else {
            average_strategy = run_cpu_solver(game, options.iterations);
        }

        poker::ExploitabilityEvaluator evaluator(game);
        const poker::ExploitabilityResult exploitability =
            evaluator.exploitability(average_strategy);

        print_exploitability_report(exploitability);

        if (options.print_strategy) {
            print_strategy(game, average_strategy, "Average strategy");
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}