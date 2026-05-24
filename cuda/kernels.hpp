#pragma once

#include "gpu_state.hpp"
#include <cuda_runtime_api.h>

namespace poker {

struct KernelLaunchConfig {
    int threads_per_block = 256;
};

int blocks_for(int n, int threads_per_block);

// -----------------------------------------------------------------------------
// Initialization / reset kernels
// -----------------------------------------------------------------------------

void launch_fill_float(
    const KernelLaunchConfig& config,
    float* d_values,
    float value,
    int count,
    cudaStream_t stream = nullptr
);

void launch_copy_float(
    const KernelLaunchConfig& config,
    const float* d_src,
    float* d_dst,
    int count,
    cudaStream_t stream = nullptr
);

void launch_initialize_uniform_strategy(
    const KernelLaunchConfig& config,
    const int* d_infoset_q_begin,
    const int* d_infoset_q_count,
    float* d_sigma,
    float* d_sigma_init,
    int num_infosets,
    cudaStream_t stream = nullptr
);

// -----------------------------------------------------------------------------
// CFR stage A: incoming action probabilities
// -----------------------------------------------------------------------------

void launch_compute_incoming_probabilities_level(
    const KernelLaunchConfig& config,
    const DeviceLevelEdges& edges,
    const int* d_player,
    const float* d_chance_prob,
    const float* d_sigma,
    float* d_incoming_prob,
    cudaStream_t stream = nullptr
);

// -----------------------------------------------------------------------------
// CFR stage B: expected utility, backward pass
// -----------------------------------------------------------------------------

void launch_initialize_terminal_utilities(
    const KernelLaunchConfig& config,
    const float* d_terminal_u_p0,
    float* d_u_p0,
    int num_nodes,
    cudaStream_t stream = nullptr
);

void launch_backward_utility_level(
    const KernelLaunchConfig& config,
    const DeviceLevelEdges& edges,
    const float* d_incoming_prob,
    const float* d_u_p0_read,
    float* d_u_p0_write,
    cudaStream_t stream = nullptr
);

// -----------------------------------------------------------------------------
// CFR stage C: counterfactual reach, forward pass
// -----------------------------------------------------------------------------

void launch_initialize_reach(
    const KernelLaunchConfig& config,
    float* d_reach_p0,
    float* d_reach_p1,
    int num_nodes,
    int root,
    cudaStream_t stream = nullptr
);

void launch_forward_reach_level(
    const KernelLaunchConfig& config,
    const DeviceLevelEdges& edges,
    const int* d_player,
    const float* d_incoming_prob,
    const float* d_reach_p0_read,
    const float* d_reach_p1_read,
    float* d_reach_p0_write,
    float* d_reach_p1_write,
    cudaStream_t stream = nullptr
);

void launch_compute_infoset_reach(
    const KernelLaunchConfig& config,
    const int* d_infoset_player,
    const int* d_infoset_node_begin,
    const int* d_infoset_node_count,
    const int* d_infoset_nodes,
    const float* d_reach_p0,
    const float* d_reach_p1,
    float* d_infoset_reach,
    int num_infosets,
    cudaStream_t stream = nullptr
);
void launch_forward_own_reach_level(
    const KernelLaunchConfig& config,
    const DeviceLevelEdges& edges,
    const int* d_player,
    const float* d_sigma,
    const float* d_own_reach_p0_read,
    const float* d_own_reach_p1_read,
    float* d_own_reach_p0_write,
    float* d_own_reach_p1_write,
    cudaStream_t stream = nullptr
);

void launch_compute_own_infoset_reach(
    const KernelLaunchConfig& config,
    const int* d_infoset_player,
    const int* d_infoset_node_begin,
    const int* d_infoset_node_count,
    const int* d_infoset_nodes,
    const float* d_own_reach_p0,
    const float* d_own_reach_p1,
    float* d_own_infoset_reach,
    int num_infosets,
    cudaStream_t stream = nullptr
);
// -----------------------------------------------------------------------------
// CFR stage D: regret computation
// -----------------------------------------------------------------------------

void launch_compute_instantaneous_regrets(
    const KernelLaunchConfig& config,
    const DeviceDecisionEdges& decision_edges,
    const int* d_player,
    const float* d_u_p0,
    const float* d_reach_p0,
    const float* d_reach_p1,
    float* d_inst_regret,
    cudaStream_t stream = nullptr
);

void launch_update_regrets(
    const KernelLaunchConfig& config,
    const float* d_inst_regret,
    float* d_regret_sum,
    int num_q,
    bool use_cfr_plus,
    cudaStream_t stream = nullptr
);

// -----------------------------------------------------------------------------
// CFR stage E: regret matching
// -----------------------------------------------------------------------------

void launch_clear_positive_regret_sums(
    const KernelLaunchConfig& config,
    float* d_positive_regret_sum,
    int num_infosets,
    cudaStream_t stream = nullptr
);

void launch_accumulate_positive_regret_sums(
    const KernelLaunchConfig& config,
    const float* d_regret_sum,
    const int* d_q_infoset,
    float* d_positive_regret_sum,
    int num_q,
    cudaStream_t stream = nullptr
);

void launch_regret_matching(
    const KernelLaunchConfig& config,
    const float* d_regret_sum,
    const float* d_positive_regret_sum,
    const float* d_sigma_init,
    const int* d_q_infoset,
    float* d_sigma,
    int num_q,
    cudaStream_t stream = nullptr
);

// -----------------------------------------------------------------------------
// CFR stage F: average strategy accumulation
// -----------------------------------------------------------------------------

void launch_accumulate_average_strategy(
    const KernelLaunchConfig& config,
    const int* d_q_infoset,
    const int* d_infoset_q_begin,
    const float* d_sigma,
    const float* d_own_infoset_reach,
    float* d_strategy_sum,
    float* d_strategy_weight_sum,
    int num_q,
    float iteration_weight,
    cudaStream_t stream = nullptr
);

void launch_normalize_average_strategy(
    const KernelLaunchConfig& config,
    const int* d_q_infoset,
    const float* d_strategy_sum,
    const float* d_strategy_weight_sum,
    const float* d_sigma_init,
    float* d_avg_strategy,
    int num_q,
    cudaStream_t stream = nullptr
);

} // namespace poker