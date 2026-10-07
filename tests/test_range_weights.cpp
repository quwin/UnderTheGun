#include "test_support.hpp"
#include "cfr_cpu.hpp"
#include "cfr_gpu.hpp"
#include <limits>

using namespace test_support;

namespace {
poker::holdem::HoldemSubgameConfig weighted_config() {
    auto config = tiny_config();
    config.board = poker::Board{{phevaluator::Card("Ah"), phevaluator::Card("7h"),
        phevaluator::Card("2c"), phevaluator::Card("Jd"), phevaluator::Card("4s")}};
    config.p0_range.clear(); config.p1_range.clear();
    config.p0_range.set_weight(poker::make_hand(phevaluator::Card("Kh"), phevaluator::Card("Qh")), 0.25f);
    config.p0_range.set_weight(poker::make_hand(phevaluator::Card("Ks"), phevaluator::Card("Kd")), 0.75f);
    config.p1_range.set_weight(poker::make_hand(phevaluator::Card("Qc"), phevaluator::Card("Qd")), 0.8f);
    config.p1_range.set_weight(poker::make_hand(phevaluator::Card("Th"), phevaluator::Card("9h")), 0.2f);
    config.p1_range.set_weight(poker::make_hand(phevaluator::Card("Kh"), phevaluator::Card("Qc")), 0.5f);
    // Board-blocked and zero-frequency hands must not enter the domain.
    config.p1_range.set_weight(poker::make_hand(phevaluator::Card("Ah"), phevaluator::Card("Ac")), 1.0f);
    config.p1_range.set_weight(poker::make_hand(phevaluator::Card("5s"), phevaluator::Card("6s")), 0.0f);
    config.terminal_mode = poker::TerminalMode::RecordComputed;
    return config;
}

void test_analytical_weighted_equity() {
    auto config = weighted_config();
    config.betting_abstraction.first_bet_sizes.clear();
    const auto game = poker::holdem::HoldemSubgameBuilder(config).build();
    check(game.hand_pairs.pair_count() == 5, "Exclude board blockers, overlapping private hands and zero frequencies.");
    for (int pair = 0; pair < game.hand_pairs.pair_count(); ++pair)
        check_near(game.hand_pairs.weight(pair),
            static_cast<double>(config.p0_range.weight(game.p0_hands.hands[game.hand_pairs.p0_index[pair]])) *
            config.p1_range.weight(game.p1_hands.hands[game.hand_pairs.p1_index[pair]]), 1e-12,
            "Retain range products for legal pairs.");
    // Only KhQh versus QcQd loses. Total legal mass=1.375, winning mass=1.175.
    const double expected = 1000.0 * 1.175 / 1.375;
    poker::TerminalValueProvider terminals;
    poker::PublicExploitabilityEvaluator evaluator(game, terminals);
    check_near(evaluator.expected_value_p0(uniform_strategy(game)), expected, 0.001,
               "Profile EV must normalize weighted compatible pairs.");
    check_near(evaluator.best_response(uniform_strategy(game), poker::Player::P0).value, expected, 0.001,
               "Default best response uses stored range frequencies.");
    poker::CpuCfrSolver cpu(game, terminals);
    cpu.run_one_iteration();
    check_near(cpu.stats().last_root_value_p0, expected, 0.001, "Default CPU training uses weighted pairs.");
    poker::UniformHandPairWeightProvider uniform;
    poker::PublicExploitabilityEvaluator overridden(game, terminals, uniform);
    check_near(overridden.expected_value_p0(uniform_strategy(game)), 800.0, 1e-6,
               "An explicit provider can override stored weights.");
    auto scaled_config = config;
    for (auto hand : config.p0_range.hands_with_positive_weight())
        scaled_config.p0_range.set_weight(hand, config.p0_range.weight(hand) * 0.1f);
    const auto scaled = poker::holdem::HoldemSubgameBuilder(scaled_config).build();
    poker::TerminalValueProvider scaled_terminals;
    poker::PublicExploitabilityEvaluator scaled_evaluator(scaled, scaled_terminals);
    check_near(scaled_evaluator.expected_value_p0(uniform_strategy(scaled)), expected, 0.001,
               "Scaling range frequencies preserves normalized EV.");
    auto manual = game;
    manual.hand_pairs.weights.clear();
    poker::TerminalValueProvider manual_terminals;
    poker::PublicExploitabilityEvaluator manual_evaluator(manual, manual_terminals);
    check_near(manual_evaluator.expected_value_p0(uniform_strategy(manual)), 800.0, 1e-6,
               "Unweighted manually constructed games retain uniform semantics.");
}

void test_weighted_learning() {
    for (int street : {5, 4}) {
        auto config = weighted_config();
        config.board.cards.resize(street);
        config.effective_stack = config.pot_size;
        auto game = poker::holdem::HoldemSubgameBuilder(config).build();
        // Exercise zero-probability pairs too, while preserving positive mass.
        game.hand_pairs.weights[0] = 0.0;
        poker::TerminalValueProvider terminals;
        poker::CpuCfrSolver cpu(game, terminals);
        cpu.run_one_iteration();
        const auto first_regrets = cpu.regret_sum();
        cpu.run_iterations(99);
        const auto cpu_avg = cpu.average_strategy();
        for (int chunk : {1, 3}) {
            poker::GpuCfrConfig gpu_config;
            gpu_config.evaluator_data_dir = UTG_TEST_EVALUATOR_DIR;
            gpu_config.pair_chunk_size = chunk;
            poker::GpuCfrSolver gpu(game, gpu_config);
            gpu.run_one_iteration();
            const auto regrets = gpu.regret_sum();
            for (std::size_t i = 0; i < regrets.size(); ++i)
                check_near(regrets[i] / game.hand_pairs.pair_count(), first_regrets[i], 0.005,
                           "Weighted GPU regrets must match normalized CPU regrets across chunks.");
            gpu.run_iterations(99);
            const auto gpu_avg = gpu.average_strategy();
            for (std::size_t i = 0; i < gpu_avg.size(); ++i)
                check_near(gpu_avg[i], cpu_avg[i], 0.002, "Weighted CPU/GPU average policies must agree.");
        }
    }
}

void test_invalid_weights() {
    const auto original = poker::holdem::HoldemSubgameBuilder(weighted_config()).build();
    for (int kind = 0; kind < 4; ++kind) {
        auto game = original;
        if (kind == 0) game.hand_pairs.weights.pop_back();
        if (kind == 1) game.hand_pairs.weights[0] = -1.0;
        if (kind == 2) game.hand_pairs.weights[0] = std::numeric_limits<double>::quiet_NaN();
        if (kind == 3) std::fill(game.hand_pairs.weights.begin(), game.hand_pairs.weights.end(), 0.0);
        bool rejected = false;
        try { game.validate(); } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "Reject malformed or zero-mass pair weights.");
    }
}
}

int main() { return run([] {
    test_analytical_weighted_equity();
    test_weighted_learning();
    test_invalid_weights();
}); }
