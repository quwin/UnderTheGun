#pragma once

#include "game.hpp"
#include "gpu_state.hpp"

#include <cstdint>
#include <vector>

namespace poker {

// -----------------------------------------------------------------------------
// GPU CFR configuration / stats
// -----------------------------------------------------------------------------

struct GpuCfrConfig {
    int num_players = 2;

    // CUDA launch configuration.
    int threads_per_block = 256;

    // Vanilla CFR by default. If true, cumulative regrets are clipped at zero.
    bool use_cfr_plus = false;

    // If true, use iteration number as average-strategy weight.
    // Vanilla CFR uses weight 1.
    bool linear_averaging = false;

    // Useful while debugging CUDA errors. Expensive if enabled every iteration.
    bool synchronize_each_iteration = false;
};

struct GpuCfrStats {
    int iterations_run = 0;

    // Optional diagnostic. Fill this from d_u_p0[root] if desired.
    float last_root_value_p0 = 0.0f;
};

// -----------------------------------------------------------------------------
// Host-side flattened game representation
// -----------------------------------------------------------------------------

struct LevelEdges {
    // Host-side edge arrays for one depth layer.
    std::vector<int> parent;
    std::vector<int> child;

    // q index for real-player decision edges.
    // -1 for chance edges.
    std::vector<int> q_index;

    int size() const {
        return static_cast<int>(parent.size());
    }

    bool empty() const {
        return parent.empty();
    }
};

struct FlatGame {
    // Metadata of the overall game structure
    int num_nodes = 0;
    int num_infosets = 0;
    int num_q = 0;
    int num_players = 2;
    int max_depth = 0;
    int root = 0;

    // -------------------------------------------------------------------------
    // Per-node arrays, length = num_nodes.
    // -------------------------------------------------------------------------

    std::vector<int> parent;
    std::vector<int> depth;

    // Uses static_cast<int>(Player):
    //   Chance   = -1
    //   P0       = 0
    //   P1       = 1
    //   Terminal = 2
    std::vector<int> player;

    // Infoset id for real-player decision nodes, -1 otherwise.
    std::vector<int> infoset;

    // Terminal payoff for P0. Nonterminal entries should be 0.
    std::vector<float> terminal_u_p0;

    // Chance probability from parent to this node.
    // Non-chance entries should be 0.
    std::vector<float> chance_prob;

    // -------------------------------------------------------------------------
    // Per-q arrays, length = num_q.
    // q means one infoset-action pair.
    // -------------------------------------------------------------------------

    std::vector<int> q_infoset;
    std::vector<int> q_local_action;

    // -------------------------------------------------------------------------
    // Per-infoset arrays, length = num_infosets.
    // -------------------------------------------------------------------------

    std::vector<int> infoset_player;

    // q entries belonging to each infoset.
    // This assumes q entries for each infoset are contiguous, which the BuildContext/game.hpp design should guarantee.
    std::vector<int> infoset_q_begin;
    std::vector<int> infoset_q_count;

    // Nodes belonging to each infoset.
    //
    // Flattened CSR-style mapping:
    //   infoset h owns nodes:
    //     infoset_nodes[
    //       infoset_node_begin[h] ...
    //       infoset_node_begin[h] + infoset_node_count[h]
    //     )
    //
    // This is needed to compute per-infoset counterfactual reach correctly.
    std::vector<int> infoset_node_begin;
    std::vector<int> infoset_node_count;
    std::vector<int> infoset_nodes;

    // -------------------------------------------------------------------------
    // Edges grouped by parent depth.
    // level_edges[d] contains edges from depth d to depth d + 1.
    // -------------------------------------------------------------------------

    std::vector<LevelEdges> level_edges;

    // Convenience host-side decision edge list.
    // Contains only edges with q_index >= 0.
    std::vector<int> decision_edge_parent;
    std::vector<int> decision_edge_child;
    std::vector<int> decision_edge_q_index;

    bool valid_basic_shape() const {
        return num_nodes >= 0 &&
               num_infosets >= 0 &&
               num_q >= 0 &&
               static_cast<int>(parent.size()) == num_nodes &&
               static_cast<int>(depth.size()) == num_nodes &&
               static_cast<int>(player.size()) == num_nodes &&
               static_cast<int>(infoset.size()) == num_nodes &&
               static_cast<int>(terminal_u_p0.size()) == num_nodes &&
               static_cast<int>(chance_prob.size()) == num_nodes &&
               static_cast<int>(q_infoset.size()) == num_q &&
               static_cast<int>(q_local_action.size()) == num_q &&
               static_cast<int>(infoset_player.size()) == num_infosets &&
               static_cast<int>(infoset_q_begin.size()) == num_infosets &&
               static_cast<int>(infoset_q_count.size()) == num_infosets &&
               static_cast<int>(infoset_node_begin.size()) == num_infosets &&
               static_cast<int>(infoset_node_count.size()) == num_infosets;
    }
};

// Converts your recursive Game tree into the flat arrays used by the GPU solver.
FlatGame flatten_game_for_gpu(const Game& game);

// -----------------------------------------------------------------------------
// GPU CFR solver public API
// -----------------------------------------------------------------------------

class GpuCfrSolver {
public:
    explicit GpuCfrSolver(const Game& game);
    GpuCfrSolver(const Game& game, GpuCfrConfig config);

    explicit GpuCfrSolver(const FlatGame& flat_game);
    GpuCfrSolver(const FlatGame& flat_game, GpuCfrConfig config);

    ~GpuCfrSolver();

    GpuCfrSolver(const GpuCfrSolver&) = delete;
    GpuCfrSolver& operator=(const GpuCfrSolver&) = delete;

    GpuCfrSolver(GpuCfrSolver&& other) noexcept;
    GpuCfrSolver& operator=(GpuCfrSolver&& other) noexcept;

    void run_iterations(int iterations);
    void run_one_iteration();

    std::vector<float> current_strategy() const;
    std::vector<float> average_strategy() const;

    // Debug/testing helpers.
    // These are useful for one-iteration kernel diagnostics.
    void set_current_strategy_for_testing(
        const std::vector<float>& strategy
    );

    std::vector<float> debug_strategy_sum() const;

    std::vector<float> debug_strategy_weight_sum() const;

    const GpuCfrStats& stats() const {
        return stats_;
    }

    const FlatGame& flat_game() const {
        return flat_;
    }

private:
    GpuCfrConfig config_;
    GpuCfrStats stats_;
    FlatGame flat_;
    GpuState gpu_;

    // Allocation / initialization.
    void upload_static_game();
    void allocate_dynamic_buffers();
    void initialize_solver_state();
    void free_device_memory();

    // One CFR iteration pipeline.
    void compute_incoming_action_probabilities();
    void compute_expected_utilities();
    void compute_counterfactual_reach();
    void compute_infoset_reach();
    void compute_instantaneous_regrets();

    void compute_own_reach();
    void compute_own_infoset_reach();

    void update_regrets();
    void update_current_strategy();
    void accumulate_average_strategy();
    void normalize_average_strategy();

    int blocks_for(int n) const;
};

} // namespace poker