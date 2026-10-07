#include "test_support.hpp"
#include "cfr_cpu.hpp"
#include "cfr_gpu.hpp"

using namespace test_support;

namespace {
poker::Game opposing_pairs(bool reverse, bool weighted) {
    poker::Game game;
    game.starting_board = poker::Board{{phevaluator::Card("Ah"), phevaluator::Card("7h"),
        phevaluator::Card("2c"), phevaluator::Card("Jd"), phevaluator::Card("4s")}};
    poker::HandPairTable pairs{{0, 0}, {0, 1}, weighted ? std::vector<double>{3.0, 1.0} : std::vector<double>{}};
    if (reverse) {
        std::reverse(pairs.p1_index.begin(), pairs.p1_index.end());
        std::reverse(pairs.weights.begin(), pairs.weights.end());
    }
    game.set_hand_domains(
        poker::HandDomain{{poker::make_hand(phevaluator::Card("Kh"), phevaluator::Card("Qh"))}},
        poker::HandDomain{{poker::make_hand(phevaluator::Card("Qc"), phevaluator::Card("Qd")),
                           poker::make_hand(phevaluator::Card("Th"), phevaluator::Card("9h"))}}, pairs);
    poker::PublicNode root;
    root.type = poker::PublicNodeType::Action; root.player = poker::Player::P0;
    game.root = game.add_node(root);
    for (int a = 0; a < 2; ++a) {
        const int child = game.add_node(poker::PublicNode{});
        game.add_action_child(game.root, child, game.add_action(poker::GameAction{a, 0}));
        game.terminal_records.push_back({a == 0 ? poker::TerminalType::Showdown : poker::TerminalType::P0_Fold,
                                         0, a == 0 ? 1000 : 0, a == 0 ? 500 : 0});
    }
    game.register_action_state(game.root, poker::Player::P0);
    game.validate();
    return game;
}

void test_iteration_aggregation() {
    for (bool weighted : {false, true}) {
        auto game = opposing_pairs(false, weighted), reversed = opposing_pairs(true, weighted);
        poker::TerminalValueProvider terminals, reversed_terminals;
        poker::CfrConfig config;
        config.use_cfr_plus = true; config.linear_averaging = true;
        poker::CpuCfrSolver cpu(game, terminals, config), other(reversed, reversed_terminals, config);
        cpu.run_one_iteration(); other.run_one_iteration();
        check_near(cpu.regret_sum()[0], 0.0, 1e-6, "Clip only after all pair contributions.");
        check_near(cpu.regret_sum()[1], weighted ? 125.0 : 0.0, 1e-6,
                   "Analytical aggregate regret must survive exactly one CFR+ clipping.");
        for (std::size_t i = 0; i < cpu.regret_sum().size(); ++i)
            check_near(other.regret_sum()[i], cpu.regret_sum()[i], 1e-6,
                       "CFR+ must be invariant to private-pair ordering.");
        cpu.run_iterations(99); other.run_iterations(99);
        for (std::size_t i = 0; i < cpu.average_strategy().size(); ++i)
            check_near(other.average_strategy()[i], cpu.average_strategy()[i], 1e-6,
                       "Average policies must be invariant to private-pair ordering.");
        for (int chunk : {1, 2}) {
            poker::GpuCfrConfig gpu_config;
            gpu_config.evaluator_data_dir = UTG_TEST_EVALUATOR_DIR;
            gpu_config.use_cfr_plus = true; gpu_config.linear_averaging = true;
            gpu_config.pair_chunk_size = chunk;
            poker::GpuCfrSolver gpu(game, gpu_config);
            gpu.run_iterations(100);
            const auto gpu_avg = gpu.average_strategy(), cpu_avg = cpu.average_strategy();
            for (std::size_t i = 0; i < gpu_avg.size(); ++i)
                check_near(gpu_avg[i], cpu_avg[i], 1e-5, "CPU/GPU CFR+ linear averages must agree.");
        }
    }
}
}

int main() { return run(test_iteration_aggregation); }
