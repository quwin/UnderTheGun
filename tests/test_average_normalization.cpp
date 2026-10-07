#include "test_support.hpp"
#include "cfr_cpu.hpp"
#include "cfr_gpu.hpp"

using namespace test_support;

int main() { return run([] {
    auto source_config = tiny_config();
    source_config.betting_abstraction.first_bet_sizes.clear();
    const auto source = poker::holdem::HoldemSubgameBuilder(source_config).build();
    poker::Game game;
    game.starting_board = source.starting_board;
    game.set_hand_domains(source.p0_hands, source.p1_hands, source.hand_pairs);
    poker::PublicNode decision;
    decision.type = poker::PublicNodeType::Action; decision.player = poker::Player::P0;
    game.root = game.add_node(decision);
    for (int action = 0; action < 3; ++action) {
        const int child = game.add_node(poker::PublicNode{});
        game.add_action_child(game.root, child, game.add_action(poker::GameAction{action, 0}));
        game.terminal_records.push_back({poker::TerminalType::P0_Fold, 0, 1000, 0});
    }
    game.register_action_state(game.root, poker::Player::P0);
    game.validate();
    auto check_uniform = [&](const std::vector<float>& strategy) {
        check_strategy(game, strategy, 1e-7);
        for (float probability : strategy)
            check_near(probability, 1.0 / 3.0, 1e-7, "Repeated thirds must remain a normalized average policy.");
        poker::TerminalValueProvider terminals;
        poker::PublicExploitabilityEvaluator evaluator(game, terminals);
        check_near(evaluator.expected_value_p0(strategy), 0.0, 1e-7, "Average is accepted as a valid profile.");
    };
    poker::TerminalValueProvider terminals;
    poker::CpuCfrSolver cpu(game, terminals);
    check_uniform(cpu.average_strategy()); // Unvisited buckets still use uniform.
    cpu.run_iterations(100000);
    check_uniform(cpu.average_strategy());
    poker::GpuCfrConfig config;
    config.evaluator_data_dir = UTG_TEST_EVALUATOR_DIR;
    config.pair_chunk_size = 3;
    poker::GpuCfrSolver gpu(game, config);
    gpu.run_iterations(1000);
    check_uniform(gpu.average_strategy());
}); }
