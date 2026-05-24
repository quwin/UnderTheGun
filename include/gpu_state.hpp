#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace poker {

// -----------------------------------------------------------------------------
// Device edge storage
// -----------------------------------------------------------------------------
//
// level_edges[d] stores all edges from depth d to depth d + 1.
// decision_edges stores only real-player decision edges, meaning q_index >= 0.

struct DeviceLevelEdges {
    int* d_parent = nullptr;
    int* d_child = nullptr;
    int* d_q_index = nullptr; // -1 for chance edges.

    int count = 0;
};

struct DeviceDecisionEdges {
    int* d_parent = nullptr;
    int* d_child = nullptr;
    int* d_q_index = nullptr; // always >= 0.

    int count = 0;
};

// -----------------------------------------------------------------------------
// Static device-side game data
// -----------------------------------------------------------------------------

struct DeviceGameData {
    int num_nodes = 0;
    int num_infosets = 0;
    int num_q = 0;
    int num_players = 2;
    int max_depth = 0;
    int root = 0;

    // Per-node arrays, length = num_nodes.
    int* d_parent = nullptr;
    int* d_depth = nullptr;

    // Uses static_cast<int>(Player):
    //   Chance  = -1
    //   P0      = 0
    //   P1      = 1
    //   Terminal= 2
    int* d_player = nullptr;

    // Infoset id for real-player decision nodes, -1 otherwise.
    int* d_infoset = nullptr;

    // Terminal payoff for P0. Nonterminal entries should be 0.
    // Length = num_nodes.
    float* d_terminal_u_p0 = nullptr;

    // Chance probability from parent to this node.
    // Non-chance entries should be 0.
    // Length = num_nodes.
    float* d_chance_prob = nullptr;

    // Per-q arrays, length = num_q.
    int* d_q_infoset = nullptr;
    int* d_q_local_action = nullptr;

    // Infoset metadata, length = num_infosets.
    int* d_infoset_player = nullptr;
    int* d_infoset_q_begin = nullptr;
    int* d_infoset_q_count = nullptr;

    // Infoset-to-node mapping.
    //
    // Needed to compute:
    //   infoset_reach[h] = sum counterfactual reach over nodes in infoset h.
    //
    // This is especially important because an infoset can contain multiple
    // game-tree nodes.
    int* d_infoset_node_begin = nullptr;
    int* d_infoset_node_count = nullptr;
    int* d_infoset_nodes = nullptr;
    int num_infoset_nodes = 0;

    // Level edge lists.
    std::vector<DeviceLevelEdges> level_edges;

    // Convenience list containing only player-decision edges.
    DeviceDecisionEdges decision_edges;
};

// -----------------------------------------------------------------------------
// Dynamic CFR device-side state
// -----------------------------------------------------------------------------

struct DeviceCfrState {
    // Current regret-matched strategy.
    // Length = num_q.
    float* d_sigma = nullptr;

    // Initial uniform strategy. Used as regret-matching fallback.
    // Length = num_q.
    float* d_sigma_init = nullptr;

    // Cumulative regret.
    // Length = num_q.
    float* d_regret_sum = nullptr;

    // Cumulative average-strategy numerator.
    // Length = num_q.
    float* d_strategy_sum = nullptr;

    // Normalized average strategy.
    // Length = num_q.
    float* d_avg_strategy = nullptr;

    // Average-strategy denominator per infoset.
    // Length = num_infosets.
    float* d_strategy_weight_sum = nullptr;
};

// -----------------------------------------------------------------------------
// Temporary per-iteration buffers
// -----------------------------------------------------------------------------

struct DeviceWorkBuffers {
    // Probability of taking the incoming edge into each node.
    // Length = num_nodes.
    float* d_incoming_prob = nullptr;

    // Expected utility for P0 at each node under current sigma.
    // Length = num_nodes.
    float* d_u_p0 = nullptr;

    // Counterfactual reach probabilities.
    // Length = num_nodes.
    float* d_reach_p0 = nullptr;
    float* d_reach_p1 = nullptr;
    // Aggregated counterfactual reach per infoset.
    // Length = num_infosets.
    float* d_counterfactual_infoset_reach = nullptr;

    // Own reach probabilities used only for average-strategy accumulation.
    //
    // These match the CPU solver's reach_p0/reach_p1 convention:
    // chance does not multiply these values; only the acting player's own
    // strategy choices multiply their own reach.
    //
    // Length = num_nodes.
    float* d_own_reach_p0 = nullptr;
    float* d_own_reach_p1 = nullptr;

    // Aggregated own reach per infoset.
    // For a P0 infoset, this is sum d_own_reach_p0[node] over nodes in the infoset.
    // For a P1 infoset, this is sum d_own_reach_p1[node] over nodes in the infoset.
    //
    // Length = num_infosets.
    float* d_own_infoset_reach = nullptr;

    // Instantaneous regret per q.
    // Length = num_q.
    float* d_inst_regret = nullptr;

    // Sum of positive cumulative regrets per infoset.
    // Length = num_infosets.
    float* d_positive_regret_sum = nullptr;

    // Optional scratch buffer for reductions/debugging.
    float* d_scratch = nullptr;
    std::size_t scratch_count = 0;
};

// -----------------------------------------------------------------------------
// Combined GPU state
// -----------------------------------------------------------------------------

struct GpuState {
    DeviceGameData game;
    DeviceCfrState cfr;
    DeviceWorkBuffers work;

    bool allocated = false;
};

// -----------------------------------------------------------------------------
// Allocation / cleanup interface
// -----------------------------------------------------------------------------
//
// Implement these in cfr_gpu.cpp or gpu_state.cpp/.cu, where cudaMalloc,
// cudaMemcpy, cudaMemset, and cudaFree are available.

void allocate_gpu_state(
    GpuState& state,
    int num_nodes,
    int num_infosets,
    int num_q,
    int num_players,
    int max_depth,
    int root
);

void free_gpu_state(GpuState& state);

void reset_work_buffers(GpuState& state);

} // namespace poker