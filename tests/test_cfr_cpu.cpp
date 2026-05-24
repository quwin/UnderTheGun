#include "kuhn_builder.hpp"
#include "cfr_cpu.hpp"
#include "exploitability.hpp"

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float kFloatTol = 1e-5f;
constexpr double kDoubleTol = 1e-8;

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

int count_terminal_nodes(const poker::Game& game) {
    int count = 0;

    for (const poker::Node& node : game.nodes) {
        if (node.terminal) {
            ++count;
        }
    }

    return count;
}

int count_chance_children(const poker::Game& game) {
    const poker::Node& root = game.node(game.root);

    check(
        root.player == poker::Player::Chance,
        "Root node should be a chance node."
    );

    return static_cast<int>(root.children.size());
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

            if (p < -kFloatTol || p > 1.0f + kFloatTol) {
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

        check_near(sum, 1.0, 1e-4, msg.str());
    }
}

void test_kuhn_game_shape_for_cfr() {
    poker::Game game = poker::build_kuhn_game();

    check(game.root == 0, "Expected root id to be 0.");
    check(game.num_players == 2, "Expected two-player Kuhn.");
    check(game.num_nodes() > 0, "Game should contain nodes.");
    check(game.num_infosets() > 0, "Game should contain infosets.");
    check(game.num_q() > 0, "Game should contain infoset-action entries.");

    check(
        count_chance_children(game) == 6,
        "Standard two-player Kuhn should have 6 ordered private-card deals."
    );

    check(
        count_terminal_nodes(game) == 30,
        "Standard two-player Kuhn should have 30 terminal histories across all deals."
    );

    // With the rational-player-only infoset convention used in this project:
    //
    // P0 initial decision:        3 card infosets
    // P1 after P0 checks:        3 card infosets
    // P1 after P0 bets:          3 card infosets
    // P0 after check-bet:        3 card infosets
    //
    // Total = 12 rational-player infosets.
    //
    // Each has 2 legal actions, so num_q = 24.
    check(
        game.num_infosets() == 12,
        "Expected 12 rational-player infosets for compact two-player Kuhn."
    );

    check(
        game.num_q() == 24,
        "Expected 24 infoset-action entries for compact two-player Kuhn."
    );

    std::cout << "[pass] test_kuhn_game_shape_for_cfr\n";
}

void test_initial_current_strategy_is_uniform() {
    poker::Game game = poker::build_kuhn_game();
    poker::CpuCfrSolver solver(game);

    const std::vector<float> strategy = solver.current_strategy();

    check_strategy_distribution(game, strategy, "initial current strategy");

    for (const poker::InfoSet& infoset : game.infosets) {
        const float expected =
            1.0f / static_cast<float>(infoset.q_indices.size());

        for (int q : infoset.q_indices) {
            std::ostringstream msg;
            msg << "Initial strategy should be uniform at q=" << q;

            check_near(strategy[q], expected, 1e-6, msg.str());
        }
    }

    std::cout << "[pass] test_initial_current_strategy_is_uniform\n";
}

void test_initial_average_strategy_is_uniform() {
    poker::Game game = poker::build_kuhn_game();
    poker::CpuCfrSolver solver(game);

    const std::vector<float> average_strategy = solver.average_strategy();

    check_strategy_distribution(game, average_strategy, "initial average strategy");

    for (const poker::InfoSet& infoset : game.infosets) {
        const float expected =
            1.0f / static_cast<float>(infoset.q_indices.size());

        for (int q : infoset.q_indices) {
            std::ostringstream msg;
            msg << "Initial average strategy should fall back to uniform at q="
                << q;

            check_near(average_strategy[q], expected, 1e-6, msg.str());
        }
    }

    std::cout << "[pass] test_initial_average_strategy_is_uniform\n";
}

void test_one_iteration_keeps_state_finite() {
    poker::Game game = poker::build_kuhn_game();
    poker::CpuCfrSolver solver(game);

    solver.run_one_iteration();

    check(
        solver.stats().iterations_run == 1,
        "Expected exactly one iteration to have run."
    );

    check(
        std::isfinite(solver.stats().last_root_value_p0),
        "Root value should be finite after one iteration."
    );

    check_finite_vector(solver.regret_sum(), "regret_sum");
    check_finite_vector(solver.strategy_sum(), "strategy_sum");

    check_strategy_distribution(
        game,
        solver.current_strategy(),
        "current strategy after one iteration"
    );

    check_strategy_distribution(
        game,
        solver.average_strategy(),
        "average strategy after one iteration"
    );

    std::cout << "[pass] test_one_iteration_keeps_state_finite\n";
}

void test_many_iterations_produce_valid_strategy() {
    poker::Game game = poker::build_kuhn_game();

    poker::CfrConfig config;
    config.use_cfr_plus = false;
    config.linear_averaging = false;

    poker::CpuCfrSolver solver(game, config);

    solver.run_iterations(10000);

    check(
        solver.stats().iterations_run == 10000,
        "Expected 10000 iterations to have run."
    );

    check_finite_vector(solver.regret_sum(), "regret_sum after training");
    check_finite_vector(solver.strategy_sum(), "strategy_sum after training");

    check_strategy_distribution(
        game,
        solver.current_strategy(),
        "current strategy after training"
    );

    check_strategy_distribution(
        game,
        solver.average_strategy(),
        "average strategy after training"
    );

    std::cout << "[pass] test_many_iterations_produce_valid_strategy\n";
}

void test_cfr_reduces_exploitability() {
    poker::Game game = poker::build_kuhn_game();

    poker::CpuCfrSolver untrained_solver(game);
    const std::vector<float> initial_avg =
        untrained_solver.average_strategy();

    const poker::ExploitabilityResult initial_result =
        poker::compute_exploitability(game, initial_avg);

    poker::CfrConfig config;
    config.use_cfr_plus = false;
    config.linear_averaging = false;

    poker::CpuCfrSolver trained_solver(game, config);
    trained_solver.run_iterations(50000);

    const std::vector<float> trained_avg =
        trained_solver.average_strategy();

    const poker::ExploitabilityResult trained_result =
        poker::compute_exploitability(game, trained_avg);

    check(
        std::isfinite(initial_result.exploitability),
        "Initial exploitability should be finite."
    );

    check(
        std::isfinite(trained_result.exploitability),
        "Trained exploitability should be finite."
    );

    check(
        trained_result.exploitability < initial_result.exploitability,
        "CFR should reduce exploitability versus the initial uniform strategy."
    );

    // This threshold is intentionally loose. It should pass for a correct
    // vanilla CFR implementation without becoming brittle.
    check(
        trained_result.exploitability < 0.08,
        "Trained Kuhn strategy exploitability is higher than expected."
    );

    std::cout << "[info] initial exploitability: "
              << initial_result.exploitability << "\n";

    std::cout << "[info] trained exploitability: "
              << trained_result.exploitability << "\n";

    std::cout << "[pass] test_cfr_reduces_exploitability\n";
}

void test_trained_value_is_reasonable_for_kuhn() {
    poker::Game game = poker::build_kuhn_game();

    poker::CpuCfrSolver solver(game);
    solver.run_iterations(50000);

    const std::vector<float> avg = solver.average_strategy();

    const double value_p0 =
        poker::compute_expected_value_p0(game, avg);

    check(
        std::isfinite(value_p0),
        "Trained expected value should be finite."
    );

    // Under the common net-payoff convention for two-player Kuhn,
    // the first player's equilibrium value is around -1/18 = -0.0555...
    // Keep this loose because early CFR implementations may differ slightly
    // in averaging convention, update order, or payoff convention.
    check(
        value_p0 > -0.12 && value_p0 < 0.02,
        "Trained P0 value is outside a reasonable Kuhn range."
    );

    std::cout << "[info] trained P0 EV: " << value_p0 << "\n";
    std::cout << "[pass] test_trained_value_is_reasonable_for_kuhn\n";
}

void run_all_tests() {
    test_kuhn_game_shape_for_cfr();
    test_initial_current_strategy_is_uniform();
    test_initial_average_strategy_is_uniform();
    test_one_iteration_keeps_state_finite();
    test_many_iterations_produce_valid_strategy();
    test_cfr_reduces_exploitability();
    test_trained_value_is_reasonable_for_kuhn();
}

} // namespace

int main() {
    try {
        run_all_tests();
    } catch (const std::exception& e) {
        std::cerr << "[fail] " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "[pass] all CPU CFR tests passed\n";
    return EXIT_SUCCESS;
}
