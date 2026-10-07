#include "test_support.hpp"
#include "cfr_cpu.hpp"
#include "cfr_gpu.hpp"

using namespace test_support;

namespace {

void test_flattening(const poker::Game& game) {
    poker::FlatTerminalData terminals;
    const auto flat = poker::flatten_public_game_for_gpu(game, terminals);
    poker::flatten_terminal_data_for_gpu(game, terminals, poker::TerminalMode::ValuePrecomputed);
    const auto hands = poker::flatten_hand_data_for_gpu(game);
    check(flat.num_nodes == game.num_nodes() && flat.num_edges == game.num_edges() &&
              flat.num_action_states == game.num_action_states() && flat.root == game.root &&
              flat.tensor_entries == game.cfr_tensor_entries(),
          "Flattening must preserve game dimensions.");
    for (int id = 0; id < game.num_nodes(); ++id) {
        const auto& node = game.node(id);
        check(flat.player.at(id) == static_cast<int>(node.player) &&
                  flat.node_type.at(id) == static_cast<int>(node.type) &&
                  flat.action_state_index.at(id) == node.action_state_index,
              "Flattened node metadata mismatch.");
        // Terminal connectivity is represented by incoming edges and the
        // terminal lookup; only nonterminals carry parent/depth arrays.
        if (node.type != poker::PublicNodeType::Terminal)
            check(flat.parent.at(id) == node.parent && flat.depth.at(id) == node.depth,
                  "Flattened nonterminal connectivity mismatch.");
    }
    std::size_t action_edges = 0;
    std::size_t all_edges = 0;
    for (const auto& level : flat.level_edges) {
        for (int e = 0; e < level.size(); ++e) {
            const auto& parent = game.node(level.parent.at(e));
            bool found = false;
            for (int local = 0; local < parent.edge_count; ++local) {
                const auto& edge = game.edge(parent.first_edge + local);
                if (edge.child == level.child.at(e)) {
                    check(level.local_action.at(e) == edge.local_action &&
                              level.chance_prob.at(e) == edge.chance_prob,
                          "Flattened edge metadata mismatch.");
                    found = true;
                }
            }
            check(found, "Flattened edge is absent from game.");
            ++all_edges;
        }
    }
    for (const auto& state : game.action_states) action_edges += state.action_count;
    check(all_edges == game.edges.size() && flat.action_edge_parent.size() == action_edges,
          "Flattening must preserve all edges.");
    check(hands.p0_pair_index == game.hand_pairs.p0_index &&
              hands.p1_pair_index == game.hand_pairs.p1_index &&
              hands.hand_pair_count == game.hand_pairs.pair_count(),
          "Flattening must preserve private hand pairs.");
    check(terminals.terminal_value_p0 == game.terminal_value_p0,
          "Flattening must preserve terminal payoffs.");
    for (int t = 0; t < terminals.terminal_count(); ++t) {
        const int id = terminals.terminal_nodes.at(t);
        check(game.node(id).type == poker::PublicNodeType::Terminal &&
                  terminals.terminal_index_by_node.at(id) == t,
              "Terminal lookup must round-trip.");
    }
    std::cout << "[pass] Hold'em GPU flattening\n";
}

void test_cpu_gpu_agreement(const poker::Game& game, int iterations) {
    poker::TerminalValueProvider terminal_values;
    poker::CpuCfrSolver cpu(game, terminal_values);
    poker::GpuCfrConfig gpu_config;
    gpu_config.terminal_mode = poker::TerminalMode::RecordComputed;
    gpu_config.evaluator_data_dir = UTG_TEST_EVALUATOR_DIR;
    gpu_config.synchronize_each_iteration = true;
    // Exercise accumulation across chunks with multiple opponent hands.
    gpu_config.pair_chunk_size = 2;
    poker::GpuCfrSolver gpu(game, gpu_config);
    cpu.run_iterations(iterations);
    gpu.run_iterations(iterations);
    check(cpu.stats().iterations_run == iterations && gpu.stats().iterations_run == iterations,
          "Solver iteration count mismatch.");
    const auto cpu_avg = cpu.average_strategy();
    const auto gpu_avg = gpu.average_strategy();
    check_strategy(game, cpu_avg);
    check_strategy(game, gpu_avg);
    check_strategy(game, cpu.current_strategy());
    check_strategy(game, gpu.current_strategy());
    double diff = 0.0;
    for (std::size_t i = 0; i < cpu_avg.size(); ++i)
        diff = std::max(diff, std::abs(static_cast<double>(cpu_avg[i]) - gpu_avg[i]));
    check_near(diff, 0.0, 1e-3, "CPU/GPU average strategies must agree.");
    poker::PublicExploitabilityEvaluator evaluator(game, terminal_values);
    check_near(evaluator.expected_value_p0(cpu_avg), evaluator.expected_value_p0(gpu_avg),
               0.1, "CPU/GPU strategy EVs must agree.");
    std::cout << "[pass] CPU/GPU agreement after " << iterations << " iterations (max diff=" << diff << ")\n";
}

} // namespace

int main() {
    return run([] {
        auto config = tiny_config();
        config.terminal_mode = poker::TerminalMode::DebugComputed;
        const auto game = poker::holdem::HoldemSubgameBuilder(config).build();
        test_flattening(game);
        test_cpu_gpu_agreement(game, 1);
        test_cpu_gpu_agreement(game, 500);
    });
}
