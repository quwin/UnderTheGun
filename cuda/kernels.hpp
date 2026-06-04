#pragma once

#include "cfr_gpu.hpp"
#include <cstdint>
#include <cuda_runtime_api.h>

namespace poker {

// -----------------------------------------------------------------------------
// Kernel launch configuration
// -----------------------------------------------------------------------------

struct KernelLaunchConfig {
    int threads_per_block = 256;
};

int blocks_for(
    int n,
    int threads_per_block
);

int blocks_for_size(
    std::size_t n,
    int threads_per_block
);

// -----------------------------------------------------------------------------
// Generic utility kernels
// -----------------------------------------------------------------------------

void launch_fill_float(
    const KernelLaunchConfig& config,
    float* d_values,
    float value,
    std::size_t count,
    cudaStream_t stream = nullptr
);

void launch_copy_float(
    const KernelLaunchConfig& config,
    const float* d_src,
    float* d_dst,
    std::size_t count,
    cudaStream_t stream = nullptr
);

// Convenience overload for older call sites that still pass int.
void launch_copy_float(
    const KernelLaunchConfig& config,
    const float* d_src,
    float* d_dst,
    int count,
    cudaStream_t stream = nullptr
);

// -----------------------------------------------------------------------------
// Strategy initialization
// -----------------------------------------------------------------------------
//
// For each action state and bucket:
//
//   sigma[offset + bucket * action_count + a] = 1 / action_count
//   sigma_init[...] = same
//
// This replaces old:
//
//   launch_initialize_uniform_strategy(
//       d_infoset_q_begin,
//       d_infoset_q_count,
//       ...
//   )

void launch_initialize_public_uniform_strategy(
    const KernelLaunchConfig& config,
    const int* d_action_state_bucket_count,
    const int* d_action_state_action_count,
    const std::uint64_t* d_action_state_tensor_offset,
    float* d_sigma,
    float* d_sigma_init,
    int num_action_states,
    cudaStream_t stream = nullptr
);

// -----------------------------------------------------------------------------
// Terminal value loading
// -----------------------------------------------------------------------------
//
// HostPrecomputed terminal mode.
//
// Input:
//
//   d_terminal_index_by_node[node_id] = terminal index, or -1
//
//   d_terminal_value_p0[
//       terminal_index * hand_pair_count + pair_id
//   ]
//
// Output:
//
//   d_node_pair_value_p0[
//       node_id * hand_pair_count + pair_id
//   ]
//
// Non-terminal nodes should receive 0.

void launch_load_precomputed_terminal_pair_values(
    const KernelLaunchConfig& config,
    int num_nodes,
    int hand_pair_count,
    int terminal_count,
    const int* d_terminal_index_by_node,
    const float* d_terminal_value_p0,
    float* d_node_pair_value_p0,
    cudaStream_t stream = nullptr
);

void launch_compute_terminal_pair_values_from_records(
    const KernelLaunchConfig& config,

    int terminal_count,
    int hand_pair_count,

    const int* d_terminal_nodes,
    const int* d_terminal_type,
    const int* d_terminal_pot,
    const int* d_terminal_p0_committed,
    const unsigned char* d_terminal_board_cards,

    const int* d_p0_pair_index,
    const int* d_p1_pair_index,

    const unsigned char* d_p0_hand_card0,
    const unsigned char* d_p0_hand_card1,
    const unsigned char* d_p1_hand_card0,
    const unsigned char* d_p1_hand_card1,

    short* d_binaries_by_id,
    short* d_suitbit_by_id,
    short* d_flush,
    short* d_noflush7,
    unsigned char* d_suits,
    int* d_dp,

    float* d_node_pair_value_p0,

    cudaStream_t stream = nullptr
);

// -----------------------------------------------------------------------------
// Backward value pass
// -----------------------------------------------------------------------------
//
// Computes one public-tree depth layer.
//
// Input/output layout:
//
//   node_pair_value[
//       node_id * hand_pair_count + pair_id
//   ]
//
// For chance edges:
//
//   parent_value[pair] += chance_prob * child_value[pair]
//
// For action edges:
//
//   acting player determines bucket:
//      P0 -> p0_bucket_by_hand_index[p0_pair_index[pair]]
//      P1 -> p1_bucket_by_hand_index[p1_pair_index[pair]]
//
//   parent_value[pair] += sigma[state,bucket,local_action] * child_value[pair]
//
// The kernel should zero/overwrite parent entries for parents represented in
// this level before accumulating children.

void launch_public_backward_pair_value_level(
    const KernelLaunchConfig& config,
    const DevicePublicLevelEdges& edges,

    const int* d_node_type,
    const int* d_player,
    const int* d_action_state_index,

    const int* d_action_state_action_count,
    const std::uint64_t* d_action_state_tensor_offset,

    const int* d_p0_pair_index,
    const int* d_p1_pair_index,
    const int* d_p0_bucket_by_hand_index,
    const int* d_p1_bucket_by_hand_index,

    const float* d_sigma,

    // Read child values from here.
    const float* d_node_pair_value_read,

    // Write parent values here.
    float* d_node_pair_value_write,

    int hand_pair_count,
    cudaStream_t stream = nullptr
);

// -----------------------------------------------------------------------------
// Reach computation
// -----------------------------------------------------------------------------
//
// Computes per-action-state bucket reaches used for regret and average strategy.
//
// Outputs:
//
//   d_state_bucket_cf_reach[
//       action_state_bucket_offset[state] + bucket
//   ]
//
//   d_state_bucket_own_reach[
//       action_state_bucket_offset[state] + bucket
//   ]
//
// Exact implementation details can evolve, but for the validation version this
// kernel can compute reaches by iterating legal hand pairs and public-tree paths.
//
// Acting player bucket for pair:
//
//   P0 bucket:
//      d_p0_bucket_by_hand_index[d_p0_pair_index[pair]]
//
//   P1 bucket:
//      d_p1_bucket_by_hand_index[d_p1_pair_index[pair]]

    void launch_initialize_public_pair_reaches(
        const KernelLaunchConfig& config,
        int root,
        int hand_pair_count,
        float* d_node_pair_reach_p0,
        float* d_node_pair_reach_p1,
        float* d_node_pair_reach_chance,
        cudaStream_t stream = nullptr
    );

    void launch_public_forward_pair_reach_level(
        const KernelLaunchConfig& config,
        const DevicePublicLevelEdges& edges,

        const int* d_node_type,
        const int* d_player,
        const int* d_action_state_index,

        const int* d_action_state_action_count,
        const std::uint64_t* d_action_state_tensor_offset,

        const int* d_p0_pair_index,
        const int* d_p1_pair_index,
        const int* d_p0_bucket_by_hand_index,
        const int* d_p1_bucket_by_hand_index,

        const float* d_sigma,

        float* d_node_pair_reach_p0,
        float* d_node_pair_reach_p1,
        float* d_node_pair_reach_chance,

        int hand_pair_count,
        cudaStream_t stream = nullptr
    );

    void launch_public_aggregate_state_bucket_reaches(
        const KernelLaunchConfig& config,

        const int* d_action_state_node,
        const int* d_action_state_player,
        const int* d_action_state_bucket_count,
        const std::uint64_t* d_action_state_bucket_offset,

        int hand_pair_count,
        const int* d_p0_pair_index,
        const int* d_p1_pair_index,
        const int* d_p0_bucket_by_hand_index,
        const int* d_p1_bucket_by_hand_index,

        const float* d_node_pair_reach_p0,
        const float* d_node_pair_reach_p1,
        const float* d_node_pair_reach_chance,

        float* d_state_bucket_cf_reach,
        float* d_state_bucket_own_reach,

        int num_action_states,
        cudaStream_t stream = nullptr
    );

// -----------------------------------------------------------------------------
// Action value extraction
// -----------------------------------------------------------------------------
//
// Converts child node-pair values into action-state bucket action values.
//
// Output:
//
//   d_action_value_p0[
//       tensor_offset[state] + bucket * action_count + local_action
//   ]
//
// Because many hand pairs can map into the same bucket, the kernel should
// aggregate or average appropriately. For exact-domain mode, each bucket is one
// exact hand index, but there are still multiple opponent hands per bucket.

    void launch_public_compute_action_values_from_pair_values(
        const KernelLaunchConfig& config,

        const DevicePublicActionEdges& action_edges,

        const int* d_action_state_player,
        const int* d_action_state_bucket_count,
        const int* d_action_state_action_count,
        const std::uint64_t* d_action_state_tensor_offset,
        const std::uint64_t* d_action_state_bucket_offset,

        int hand_pair_count,
        const int* d_p0_pair_index,
        const int* d_p1_pair_index,
        const int* d_p0_bucket_by_hand_index,
        const int* d_p1_bucket_by_hand_index,

        const float* d_node_pair_value_p0,

        const float* d_node_pair_reach_p0,
        const float* d_node_pair_reach_p1,
        const float* d_node_pair_reach_chance,
        const float* d_state_bucket_cf_reach,

        float* d_action_value_p0,

        cudaStream_t stream = nullptr
    );

// -----------------------------------------------------------------------------
// Regret update
// -----------------------------------------------------------------------------
//
// For each action state and bucket:
//
//   state_value = sum_a sigma[a] * action_value[a]
//
// For P0 action states:
//
//   regret_delta[a] = cf_reach * (action_value_p0[a] - state_value_p0)
//
// For P1 action states, utility is from P1 perspective, so:
//
//   regret_delta[a] = cf_reach * (
//       -action_value_p0[a] - (-state_value_p0)
//   )
//
// CFR+:
//
//   regret_sum[idx] = max(0, regret_sum[idx] + regret_delta)
//
// Vanilla CFR:
//
//   regret_sum[idx] += regret_delta
//
// Also writes:
//
//   d_state_bucket_value_p0[state_bucket_index] = state_value_p0

void launch_public_update_regrets(
    const KernelLaunchConfig& config,

    const int* d_action_state_player,
    const int* d_action_state_bucket_count,
    const int* d_action_state_action_count,
    const std::uint64_t* d_action_state_tensor_offset,
    const std::uint64_t* d_action_state_bucket_offset,

    const float* d_action_value_p0,
    float* d_state_bucket_value_p0,
    const float* d_state_bucket_cf_reach,

    const float* d_sigma,
    float* d_regret_sum,

    int num_action_states,
    bool use_cfr_plus,

    cudaStream_t stream = nullptr
);

// -----------------------------------------------------------------------------
// Average strategy accumulation
// -----------------------------------------------------------------------------
//
// For each action-state bucket/action:
//
//   strategy_sum[idx] += own_reach[state_bucket] * iteration_weight * sigma[idx]
//
// and:
//
//   strategy_weight_sum[state_bucket] += own_reach[state_bucket] * iteration_weight
//
// Note:
//   The implementation should add the denominator once per bucket, not once per
//   action.

void launch_public_accumulate_average_strategy(
    const KernelLaunchConfig& config,

    const int* d_action_state_bucket_count,
    const int* d_action_state_action_count,
    const std::uint64_t* d_action_state_tensor_offset,
    const std::uint64_t* d_action_state_bucket_offset,

    const float* d_state_bucket_own_reach,
    const float* d_sigma,

    float* d_strategy_sum,
    float* d_strategy_weight_sum,

    int num_action_states,
    float iteration_weight,

    cudaStream_t stream = nullptr
);

// -----------------------------------------------------------------------------
// Average strategy normalization
// -----------------------------------------------------------------------------
//
// For each action-state bucket:
//
//   if weight_sum > 0:
//       avg_strategy[a] = strategy_sum[a] / weight_sum
//   else:
//       avg_strategy[a] = uniform

void launch_public_normalize_average_strategy(
    const KernelLaunchConfig& config,

    const int* d_action_state_bucket_count,
    const int* d_action_state_action_count,
    const std::uint64_t* d_action_state_tensor_offset,
    const std::uint64_t* d_action_state_bucket_offset,

    const float* d_strategy_sum,
    const float* d_strategy_weight_sum,

    float* d_avg_strategy,

    int num_action_states,

    cudaStream_t stream = nullptr
);

// -----------------------------------------------------------------------------
// Regret matching
// -----------------------------------------------------------------------------
//
// For each action-state bucket:
//
//   positive_sum = sum_a max(regret_sum[a], 0)
//
//   if positive_sum > 0:
//       sigma[a] = max(regret_sum[a], 0) / positive_sum
//   else:
//       sigma[a] = sigma_init[a]
//
// `sigma_init` should normally be uniform.

void launch_public_regret_matching(
    const KernelLaunchConfig& config,

    const int* d_action_state_bucket_count,
    const int* d_action_state_action_count,
    const std::uint64_t* d_action_state_tensor_offset,

    const float* d_regret_sum,
    float* d_sigma,
    const float* d_sigma_init,

    int num_action_states,

    cudaStream_t stream = nullptr
);

} // namespace poker