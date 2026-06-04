// kernels.cu
//
// Validation-first CUDA kernels for the public-tree / hand-aware CFR layout.
//
// Strategy/regret tensor:
//
//   tensor_index =
//       action_state_tensor_offset[state]
//     + bucket * action_state_action_count[state]
//     + local_action
//
// State-bucket tensor:
//
//   state_bucket_index =
//       action_state_bucket_offset[state]
//     + bucket
//
// Player convention:
//
//   Player::Chance   = -1
//   Player::P0       = 0
//   Player::P1       = 1
//   Player::Terminal = 2
//
// PublicNodeType convention:
//
//   Action   = 0
//   Chance   = 1
//   Terminal = 2

#include "kernels.hpp"
#include "../external/CUDA-Poker-Calculator/src/cuda/evaluator7.cu"
#include "../external/CUDA-Poker-Calculator/src/cuda/hash.cu"

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace poker {
namespace {

constexpr int kPlayerP0 = 0;
constexpr int kPlayerP1 = 1;

constexpr int kNodeAction = 0;
constexpr int kNodeChance = 1;

extern "C" __device__ int evaluate_7cards(
    int a,
    int b,
    int c,
    int d,
    int e,
    int f,
    int g,
    short* binaries_by_id,
    short* suitbit_by_id,
    short* flush,
    short* noflush7,
    unsigned char* suits,
    int* dp
);

__device__ __forceinline__ std::uint64_t tensor_index_device(
    const std::uint64_t* offset,
    const int* action_count,
    int state,
    int bucket,
    int local_action
) {
    return offset[state] +
           static_cast<std::uint64_t>(bucket) *
               static_cast<std::uint64_t>(action_count[state]) +
           static_cast<std::uint64_t>(local_action);
}

__device__ __forceinline__ int pair_bucket_for_player(
    int player,
    int pair_id,
    const int* p0_pair_index,
    const int* p1_pair_index,
    const int* p0_bucket_by_hand_index,
    const int* p1_bucket_by_hand_index
) {
    if (player == kPlayerP0) {
        const int h = p0_pair_index[pair_id];
        return p0_bucket_by_hand_index[h];
    }

    if (player == kPlayerP1) {
        const int h = p1_pair_index[pair_id];
        return p1_bucket_by_hand_index[h];
    }

    return -1;
}

__global__ void fill_float_kernel(
    float* values,
    float value,
    std::size_t count
) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < count) {
        values[i] = value;
    }
}

__global__ void copy_float_kernel(
    const float* src,
    float* dst,
    std::size_t count
) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < count) {
        dst[i] = src[i];
    }
}

__global__ void initialize_public_uniform_strategy_kernel(
    const int* bucket_count,
    const int* action_count,
    const std::uint64_t* tensor_offset,
    float* sigma,
    float* sigma_init,
    int num_action_states
) {
    const int state = blockIdx.x;

    if (state >= num_action_states) {
        return;
    }

    const int buckets = bucket_count[state];
    const int actions = action_count[state];

    if (buckets <= 0 || actions <= 0) {
        return;
    }

    const float uniform = 1.0f / static_cast<float>(actions);

    for (int bucket = threadIdx.x; bucket < buckets; bucket += blockDim.x) {
        const std::uint64_t base =
            tensor_offset[state] +
            static_cast<std::uint64_t>(bucket) *
                static_cast<std::uint64_t>(actions);

        for (int a = 0; a < actions; ++a) {
            sigma[base + static_cast<std::uint64_t>(a)] = uniform;
            sigma_init[base + static_cast<std::uint64_t>(a)] = uniform;
        }
    }
}

__global__ void load_precomputed_terminal_pair_values_kernel(
    int num_nodes,
    int hand_pair_count,
    const int* terminal_index_by_node,
    const float* terminal_value_p0,
    float* node_pair_value_p0
) {
    const std::size_t total =
        static_cast<std::size_t>(num_nodes) *
        static_cast<std::size_t>(hand_pair_count);

    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= total) {
        return;
    }

    const int node =
        static_cast<int>(i / static_cast<std::size_t>(hand_pair_count));

    const int pair =
        static_cast<int>(i % static_cast<std::size_t>(hand_pair_count));

    const int terminal_index = terminal_index_by_node[node];

    if (terminal_index < 0) {
        node_pair_value_p0[i] = 0.0f;
        return;
    }

    const std::size_t terminal_idx =
        static_cast<std::size_t>(terminal_index) *
            static_cast<std::size_t>(hand_pair_count) +
        static_cast<std::size_t>(pair);

    node_pair_value_p0[i] = terminal_value_p0[terminal_idx];
}

// This is validation-first and assumes all outgoing edges for a parent are
// contiguous in edges.parent. That matches your Game's contiguous edge ranges.
__global__ void public_backward_pair_value_level_kernel(
    const int edge_count,
    const int* edge_parent,
    const int* edge_child,
    const int* edge_local_action,
    const float* edge_chance_prob,

    const int* node_type,
    const int* player,
    const int* action_state_index,

    const int* action_state_action_count,
    const std::uint64_t* action_state_tensor_offset,

    const int* p0_pair_index,
    const int* p1_pair_index,
    const int* p0_bucket_by_hand_index,
    const int* p1_bucket_by_hand_index,

    const float* sigma,

    const float* node_pair_value_read,
    float* node_pair_value_write,

    int hand_pair_count
) {
    const std::size_t total =
        static_cast<std::size_t>(edge_count) *
        static_cast<std::size_t>(hand_pair_count);

    const std::size_t linear =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (linear >= total) {
        return;
    }

    const int edge_id =
        static_cast<int>(linear / static_cast<std::size_t>(hand_pair_count));

    const int pair_id =
        static_cast<int>(linear % static_cast<std::size_t>(hand_pair_count));

    const int parent = edge_parent[edge_id];

    // Only the first edge for each parent computes the full parent value.
    // This avoids atomics and avoids zero/add races.
    if (edge_id > 0 && edge_parent[edge_id - 1] == parent) {
        return;
    }

    const int ptype = node_type[parent];

    float value = 0.0f;

    int e = edge_id;

    while (e < edge_count && edge_parent[e] == parent) {
        const int child = edge_child[e];

        const float child_value =
            node_pair_value_read[
                static_cast<std::size_t>(child) *
                    static_cast<std::size_t>(hand_pair_count) +
                static_cast<std::size_t>(pair_id)
            ];

        if (ptype == kNodeChance) {
            value += edge_chance_prob[e] * child_value;
        } else if (ptype == kNodeAction) {
            const int state = action_state_index[parent];
            const int pl = player[parent];

            const int bucket =
                pair_bucket_for_player(
                    pl,
                    pair_id,
                    p0_pair_index,
                    p1_pair_index,
                    p0_bucket_by_hand_index,
                    p1_bucket_by_hand_index
                );

            const int local_action = edge_local_action[e];

            const std::uint64_t idx =
                tensor_index_device(
                    action_state_tensor_offset,
                    action_state_action_count,
                    state,
                    bucket,
                    local_action
                );

            value += sigma[idx] * child_value;
        }

        ++e;
    }

    node_pair_value_write[
        static_cast<std::size_t>(parent) *
            static_cast<std::size_t>(hand_pair_count) +
        static_cast<std::size_t>(pair_id)
    ] = value;
}

__global__ void public_update_regrets_kernel(
    const int* action_state_player,
    const int* action_state_bucket_count,
    const int* action_state_action_count,
    const std::uint64_t* action_state_tensor_offset,
    const std::uint64_t* action_state_bucket_offset,

    const float* action_value_p0,
    float* state_bucket_value_p0,
    const float* state_bucket_cf_reach,

    const float* sigma,
    float* regret_sum,

    int num_action_states,
    bool use_cfr_plus
) {
    const int state = blockIdx.x;

    if (state >= num_action_states) {
        return;
    }

    const int buckets = action_state_bucket_count[state];
    const int actions = action_state_action_count[state];
    const int player = action_state_player[state];

    for (int bucket = threadIdx.x; bucket < buckets; bucket += blockDim.x) {
        float node_value_p0 = 0.0f;

        const std::uint64_t base =
            action_state_tensor_offset[state] +
            static_cast<std::uint64_t>(bucket) *
                static_cast<std::uint64_t>(actions);

        for (int a = 0; a < actions; ++a) {
            const std::uint64_t idx = base + static_cast<std::uint64_t>(a);
            node_value_p0 += sigma[idx] * action_value_p0[idx];
        }

        const std::uint64_t state_bucket =
            action_state_bucket_offset[state] +
            static_cast<std::uint64_t>(bucket);

        state_bucket_value_p0[state_bucket] = node_value_p0;

        const float cf_reach = state_bucket_cf_reach[state_bucket];

        for (int a = 0; a < actions; ++a) {
            const std::uint64_t idx = base + static_cast<std::uint64_t>(a);

            float regret_delta = 0.0f;

            if (player == kPlayerP0) {
                regret_delta =
                    cf_reach * (action_value_p0[idx] - node_value_p0);
            } else if (player == kPlayerP1) {
                regret_delta =
                    cf_reach * ((-action_value_p0[idx]) - (-node_value_p0));
            }

            const float updated = regret_sum[idx] + regret_delta;

            regret_sum[idx] =
                use_cfr_plus && updated < 0.0f
                    ? 0.0f
                    : updated;
        }
    }
}

__global__ void public_accumulate_average_strategy_kernel(
    const int* bucket_count,
    const int* action_count,
    const std::uint64_t* tensor_offset,
    const std::uint64_t* bucket_offset,

    const float* state_bucket_own_reach,
    const float* sigma,

    float* strategy_sum,
    float* strategy_weight_sum,

    int num_action_states,
    float iteration_weight
) {
    const int state = blockIdx.x;

    if (state >= num_action_states) {
        return;
    }

    const int buckets = bucket_count[state];
    const int actions = action_count[state];

    for (int bucket = threadIdx.x; bucket < buckets; bucket += blockDim.x) {
        const std::uint64_t sb =
            bucket_offset[state] + static_cast<std::uint64_t>(bucket);

        const float weight =
            state_bucket_own_reach[sb] * iteration_weight;

        strategy_weight_sum[sb] += weight;

        const std::uint64_t base =
            tensor_offset[state] +
            static_cast<std::uint64_t>(bucket) *
                static_cast<std::uint64_t>(actions);

        for (int a = 0; a < actions; ++a) {
            const std::uint64_t idx = base + static_cast<std::uint64_t>(a);
            strategy_sum[idx] += weight * sigma[idx];
        }
    }
}

__global__ void public_normalize_average_strategy_kernel(
    const int* bucket_count,
    const int* action_count,
    const std::uint64_t* tensor_offset,
    const std::uint64_t* bucket_offset,

    const float* strategy_sum,
    const float* strategy_weight_sum,

    float* avg_strategy,

    int num_action_states
) {
    const int state = blockIdx.x;

    if (state >= num_action_states) {
        return;
    }

    const int buckets = bucket_count[state];
    const int actions = action_count[state];

    for (int bucket = threadIdx.x; bucket < buckets; bucket += blockDim.x) {
        const std::uint64_t sb =
            bucket_offset[state] + static_cast<std::uint64_t>(bucket);

        const float denom = strategy_weight_sum[sb];
        const float uniform = 1.0f / static_cast<float>(actions);

        const std::uint64_t base =
            tensor_offset[state] +
            static_cast<std::uint64_t>(bucket) *
                static_cast<std::uint64_t>(actions);

        for (int a = 0; a < actions; ++a) {
            const std::uint64_t idx = base + static_cast<std::uint64_t>(a);

            avg_strategy[idx] =
                denom > 0.0f
                    ? strategy_sum[idx] / denom
                    : uniform;
        }
    }
}

__global__ void public_regret_matching_kernel(
    const int* bucket_count,
    const int* action_count,
    const std::uint64_t* tensor_offset,

    const float* regret_sum,
    float* sigma,
    const float* sigma_init,

    int num_action_states
) {
    const int state = blockIdx.x;

    if (state >= num_action_states) {
        return;
    }

    const int buckets = bucket_count[state];
    const int actions = action_count[state];

    for (int bucket = threadIdx.x; bucket < buckets; bucket += blockDim.x) {
        const std::uint64_t base =
            tensor_offset[state] +
            static_cast<std::uint64_t>(bucket) *
                static_cast<std::uint64_t>(actions);

        float positive_sum = 0.0f;

        for (int a = 0; a < actions; ++a) {
            const float r = regret_sum[base + static_cast<std::uint64_t>(a)];

            if (r > 0.0f) {
                positive_sum += r;
            }
        }

        if (positive_sum > 0.0f) {
            for (int a = 0; a < actions; ++a) {
                const std::uint64_t idx = base + static_cast<std::uint64_t>(a);
                const float r = regret_sum[idx];
                sigma[idx] = r > 0.0f ? r / positive_sum : 0.0f;
            }
        } else {
            for (int a = 0; a < actions; ++a) {
                const std::uint64_t idx = base + static_cast<std::uint64_t>(a);
                sigma[idx] = sigma_init[idx];
            }
        }
    }
}
__device__ float terminal_win_utility(
int pot,
int p0_committed
) {
    return static_cast<float>(pot - p0_committed);
}

__device__ float terminal_loss_utility(
    int p0_committed
) {
    return -static_cast<float>(p0_committed);
}

__device__ float terminal_tie_utility(
    int pot,
    int p0_committed
) {
    return 0.5f * static_cast<float>(pot) -
           static_cast<float>(p0_committed);
}

__global__ void compute_terminal_pair_values_from_records_kernel(
    int terminal_count,
    int hand_pair_count,

    const int* __restrict__ terminal_nodes,
    const int* __restrict__ terminal_type,
    const int* __restrict__ pot,
    const int* __restrict__ p0_committed,
    const unsigned char* __restrict__ terminal_board_cards,

    const int* __restrict__ p0_pair_index,
    const int* __restrict__ p1_pair_index,

    const unsigned char* __restrict__ p0_hand_card0,
    const unsigned char* __restrict__ p0_hand_card1,
    const unsigned char* __restrict__ p1_hand_card0,
    const unsigned char* __restrict__ p1_hand_card1,

    short* binaries_by_id,
    short* suitbit_by_id,
    short* flush,
    short* noflush7,
    unsigned char* suits,
    int* dp,

    float* __restrict__ node_pair_value_p0
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = terminal_count * hand_pair_count;
    if (idx >= total) {
        return;
    }
    const int terminal_index = idx / hand_pair_count;
    const int pair_id = idx - terminal_index * hand_pair_count;
    const int node_id = terminal_nodes[terminal_index];
    const int type = terminal_type[terminal_index];
    const int terminal_pot = pot[terminal_index];
    const int terminal_p0_committed = p0_committed[terminal_index];

    float value = 0.0f;

    if (type == static_cast<int>(TerminalType::P0_Fold)) {
        value = terminal_loss_utility(terminal_p0_committed);
    } else if (type == static_cast<int>(TerminalType::P1_Fold)) {
        value = terminal_win_utility(terminal_pot, terminal_p0_committed);
    } else if (type == static_cast<int>(TerminalType::Showdown)) {
        const int p0_i = p0_pair_index[pair_id];
        const int p1_i = p1_pair_index[pair_id];
        const int t = idx / hand_pair_count;

        int p0a = p0_hand_card0[p0_i];
        int p0b = p0_hand_card1[p0_i];
        int p1a = p1_hand_card0[p1_i];
        int p1b = p1_hand_card1[p1_i];
        int b0 = terminal_board_cards[t * 5 + 0];
        int b1 = terminal_board_cards[t * 5 + 1];
        int b2 = terminal_board_cards[t * 5 + 2];
        int b3 = terminal_board_cards[t * 5 + 3];
        int b4 = terminal_board_cards[t * 5 + 4];

        const int p0_rank = evaluate_7cards(
            p0a, p0b, b0, b1, b2, b3, b4,
            binaries_by_id,
            suitbit_by_id,
            flush,
            noflush7,
            suits,
            dp
        );

        const int p1_rank = evaluate_7cards(
            p1a, p1b, b0, b1, b2, b3, b4,
            binaries_by_id,
            suitbit_by_id,
            flush,
            noflush7,
            suits,
            dp
        );
        // Smaller rank is stronger.
        if (p0_rank < p1_rank) {
            value = terminal_win_utility(terminal_pot, terminal_p0_committed);
        } else if (p1_rank < p0_rank) {
            value = terminal_loss_utility(terminal_p0_committed);
        } else {
            value = terminal_tie_utility(terminal_pot, terminal_p0_committed);
        }
    } else if (type == static_cast<int>(TerminalType::AllIn)) {
        // TODO:
        value = 0;
    }

    node_pair_value_p0[
        static_cast<std::size_t>(node_id) *
        static_cast<std::size_t>(hand_pair_count) +
        static_cast<std::size_t>(pair_id)
    ] = value;
}
} // namespace

// -----------------------------------------------------------------------------
// Launch helpers
// -----------------------------------------------------------------------------

int blocks_for(
    int n,
    int threads_per_block
) {
    if (n <= 0) {
        return 0;
    }

    return (n + threads_per_block - 1) / threads_per_block;
}

int blocks_for_size(
    std::size_t n,
    int threads_per_block
) {
    if (n == 0) {
        return 0;
    }

    return static_cast<int>(
        (n + static_cast<std::size_t>(threads_per_block) - 1) /
        static_cast<std::size_t>(threads_per_block)
    );
}

// -----------------------------------------------------------------------------
// Generic utility kernels
// -----------------------------------------------------------------------------

void launch_fill_float(
    const KernelLaunchConfig& config,
    float* d_values,
    float value,
    std::size_t count,
    cudaStream_t stream
) {
    if (count == 0) {
        return;
    }

    fill_float_kernel<<<
        blocks_for_size(count, config.threads_per_block),
        config.threads_per_block,
        0,
        stream
    >>>(
        d_values,
        value,
        count
    );
}

void launch_copy_float(
    const KernelLaunchConfig& config,
    const float* d_src,
    float* d_dst,
    std::size_t count,
    cudaStream_t stream
) {
    if (count == 0) {
        return;
    }

    copy_float_kernel<<<
        blocks_for_size(count, config.threads_per_block),
        config.threads_per_block,
        0,
        stream
    >>>(
        d_src,
        d_dst,
        count
    );
}

void launch_copy_float(
    const KernelLaunchConfig& config,
    const float* d_src,
    float* d_dst,
    int count,
    cudaStream_t stream
) {
    if (count <= 0) {
        return;
    }

    launch_copy_float(
        config,
        d_src,
        d_dst,
        static_cast<std::size_t>(count),
        stream
    );
}

// -----------------------------------------------------------------------------
// Strategy initialization
// -----------------------------------------------------------------------------

void launch_initialize_public_uniform_strategy(
    const KernelLaunchConfig& config,
    const int* d_action_state_bucket_count,
    const int* d_action_state_action_count,
    const std::uint64_t* d_action_state_tensor_offset,
    float* d_sigma,
    float* d_sigma_init,
    int num_action_states,
    cudaStream_t stream
) {
    if (num_action_states <= 0) {
        return;
    }

    initialize_public_uniform_strategy_kernel<<<
        num_action_states,
        config.threads_per_block,
        0,
        stream
    >>>(
        d_action_state_bucket_count,
        d_action_state_action_count,
        d_action_state_tensor_offset,
        d_sigma,
        d_sigma_init,
        num_action_states
    );
}

// -----------------------------------------------------------------------------
// Terminal value loading
// -----------------------------------------------------------------------------

void launch_load_precomputed_terminal_pair_values(
    const KernelLaunchConfig& config,
    int num_nodes,
    int hand_pair_count,
    int /*terminal_count*/,
    const int* d_terminal_index_by_node,
    const float* d_terminal_value_p0,
    float* d_node_pair_value_p0,
    cudaStream_t stream
) {
    if (num_nodes <= 0 || hand_pair_count <= 0) {
        return;
    }

    const std::size_t count =
        static_cast<std::size_t>(num_nodes) *
        static_cast<std::size_t>(hand_pair_count);

    load_precomputed_terminal_pair_values_kernel<<<
        blocks_for_size(count, config.threads_per_block),
        config.threads_per_block,
        0,
        stream
    >>>(
        num_nodes,
        hand_pair_count,
        d_terminal_index_by_node,
        d_terminal_value_p0,
        d_node_pair_value_p0
    );
}

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

        cudaStream_t stream
    ) {
    const int total = terminal_count * hand_pair_count;

    if (total <= 0) {
        return;
    }

    compute_terminal_pair_values_from_records_kernel<<<
        blocks_for(total, config.threads_per_block),
        config.threads_per_block,
        0,
        stream
    >>>(
        terminal_count,
        hand_pair_count,

        d_terminal_nodes,
        d_terminal_type,
        d_terminal_pot,
        d_terminal_p0_committed,
        d_terminal_board_cards,

        d_p0_pair_index,
        d_p1_pair_index,

        d_p0_hand_card0,
        d_p0_hand_card1,
        d_p1_hand_card0,
        d_p1_hand_card1,

        d_binaries_by_id,
        d_suitbit_by_id,
        d_flush,
        d_noflush7,
        d_suits,
        d_dp,

        d_node_pair_value_p0
    );
}
// -----------------------------------------------------------------------------
// Backward value pass
// -----------------------------------------------------------------------------

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

    const float* d_node_pair_value_read,
    float* d_node_pair_value_write,

    int hand_pair_count,
    cudaStream_t stream
) {
    if (edges.count <= 0 || hand_pair_count <= 0) {
        return;
    }

    const std::size_t count =
        static_cast<std::size_t>(edges.count) *
        static_cast<std::size_t>(hand_pair_count);

    public_backward_pair_value_level_kernel<<<
        blocks_for_size(count, config.threads_per_block),
        config.threads_per_block,
        0,
        stream
    >>>(
        edges.count,
        edges.d_parent,
        edges.d_child,
        edges.d_local_action,
        edges.d_chance_prob,

        d_node_type,
        d_player,
        d_action_state_index,

        d_action_state_action_count,
        d_action_state_tensor_offset,

        d_p0_pair_index,
        d_p1_pair_index,
        d_p0_bucket_by_hand_index,
        d_p1_bucket_by_hand_index,

        d_sigma,

        d_node_pair_value_read,
        d_node_pair_value_write,

        hand_pair_count
    );
}

// -----------------------------------------------------------------------------
// Reach computation
// -----------------------------------------------------------------------------

    __global__ void initialize_public_pair_reaches_kernel(
        int root,
        int hand_pair_count,
        float* reach_p0,
        float* reach_p1,
        float* reach_chance
    ) {
    const int pair =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (pair >= hand_pair_count) {
        return;
    }

    const std::size_t idx =
        static_cast<std::size_t>(root) *
            static_cast<std::size_t>(hand_pair_count) +
        static_cast<std::size_t>(pair);

    reach_p0[idx] = 1.0f;
    reach_p1[idx] = 1.0f;
    reach_chance[idx] = 1.0f;
}

    void launch_initialize_public_pair_reaches(
    const KernelLaunchConfig& config,
    int root,
    int hand_pair_count,
    float* d_node_pair_reach_p0,
    float* d_node_pair_reach_p1,
    float* d_node_pair_reach_chance,
    cudaStream_t stream
) {
    if (hand_pair_count <= 0) {
        return;
    }

    initialize_public_pair_reaches_kernel<<<
        blocks_for(hand_pair_count, config.threads_per_block),
        config.threads_per_block,
        0,
        stream
    >>>(
        root,
        hand_pair_count,
        d_node_pair_reach_p0,
        d_node_pair_reach_p1,
        d_node_pair_reach_chance
    );
}
    __global__ void public_forward_pair_reach_level_kernel(
    int edge_count,
    const int* edge_parent,
    const int* edge_child,
    const int* edge_local_action,
    const float* edge_chance_prob,

    const int* node_type,
    const int* player,
    const int* action_state_index,

    const int* action_state_action_count,
    const std::uint64_t* action_state_tensor_offset,

    const int* p0_pair_index,
    const int* p1_pair_index,
    const int* p0_bucket_by_hand_index,
    const int* p1_bucket_by_hand_index,

    const float* sigma,

    float* reach_p0,
    float* reach_p1,
    float* reach_chance,

    int hand_pair_count
) {
    const std::size_t total =
        static_cast<std::size_t>(edge_count) *
        static_cast<std::size_t>(hand_pair_count);

    const std::size_t linear =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (linear >= total) {
        return;
    }

    const int edge =
        static_cast<int>(linear / static_cast<std::size_t>(hand_pair_count));

    const int pair =
        static_cast<int>(linear % static_cast<std::size_t>(hand_pair_count));

    const int parent = edge_parent[edge];
    const int child = edge_child[edge];

    const std::size_t parent_idx =
        static_cast<std::size_t>(parent) *
            static_cast<std::size_t>(hand_pair_count) +
        static_cast<std::size_t>(pair);

    const std::size_t child_idx =
        static_cast<std::size_t>(child) *
            static_cast<std::size_t>(hand_pair_count) +
        static_cast<std::size_t>(pair);

    float r0 = reach_p0[parent_idx];
    float r1 = reach_p1[parent_idx];
    float rc = reach_chance[parent_idx];

    const int ptype = node_type[parent];

    if (ptype == kNodeChance) {
        rc *= edge_chance_prob[edge];
    } else if (ptype == kNodeAction) {
        const int acting_player = player[parent];
        const int state = action_state_index[parent];
        const int local_action = edge_local_action[edge];

        int bucket = -1;

        if (acting_player == kPlayerP0) {
            const int h = p0_pair_index[pair];
            bucket = p0_bucket_by_hand_index[h];
        } else if (acting_player == kPlayerP1) {
            const int h = p1_pair_index[pair];
            bucket = p1_bucket_by_hand_index[h];
        } else {
            return;
        }

        const std::uint64_t idx =
            tensor_index_device(
                action_state_tensor_offset,
                action_state_action_count,
                state,
                bucket,
                local_action
            );

        const float action_prob = sigma[idx];

        if (acting_player == kPlayerP0) {
            r0 *= action_prob;
        } else {
            r1 *= action_prob;
        }
    } else {
        return;
    }

    // Tree invariant: each child has one parent, so this is a unique write.
    reach_p0[child_idx] = r0;
    reach_p1[child_idx] = r1;
    reach_chance[child_idx] = rc;
}
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
    cudaStream_t stream
) {
    if (edges.count <= 0 || hand_pair_count <= 0) {
        return;
    }

    const std::size_t count =
        static_cast<std::size_t>(edges.count) *
        static_cast<std::size_t>(hand_pair_count);

    public_forward_pair_reach_level_kernel<<<
        blocks_for_size(count, config.threads_per_block),
        config.threads_per_block,
        0,
        stream
    >>>(
        edges.count,
        edges.d_parent,
        edges.d_child,
        edges.d_local_action,
        edges.d_chance_prob,

        d_node_type,
        d_player,
        d_action_state_index,

        d_action_state_action_count,
        d_action_state_tensor_offset,

        d_p0_pair_index,
        d_p1_pair_index,
        d_p0_bucket_by_hand_index,
        d_p1_bucket_by_hand_index,

        d_sigma,

        d_node_pair_reach_p0,
        d_node_pair_reach_p1,
        d_node_pair_reach_chance,

        hand_pair_count
    );
}
    __global__ void public_aggregate_state_bucket_reaches_kernel(
    int num_action_states,

    const int* action_state_node,
    const int* action_state_player,
    const int* action_state_bucket_count,
    const std::uint64_t* action_state_bucket_offset,

    int hand_pair_count,
    const int* p0_pair_index,
    const int* p1_pair_index,
    const int* p0_bucket_by_hand_index,
    const int* p1_bucket_by_hand_index,

    const float* reach_p0,
    const float* reach_p1,
    const float* reach_chance,

    float* state_bucket_cf_reach,
    float* state_bucket_own_reach
) {
    const int state = blockIdx.x;

    if (state >= num_action_states) {
        return;
    }

    const int node = action_state_node[state];
    const int player = action_state_player[state];
    const int buckets = action_state_bucket_count[state];

    for (int bucket = threadIdx.x; bucket < buckets; bucket += blockDim.x) {
        double cf_sum = 0.0;
        double own_sum = 0.0;

        for (int pair = 0; pair < hand_pair_count; ++pair) {
            int pair_bucket = -1;

            if (player == kPlayerP0) {
                const int h = p0_pair_index[pair];
                pair_bucket = p0_bucket_by_hand_index[h];
            } else if (player == kPlayerP1) {
                const int h = p1_pair_index[pair];
                pair_bucket = p1_bucket_by_hand_index[h];
            } else {
                continue;
            }

            if (pair_bucket != bucket) {
                continue;
            }

            const std::size_t node_pair =
                static_cast<std::size_t>(node) *
                    static_cast<std::size_t>(hand_pair_count) +
                static_cast<std::size_t>(pair);

            const double r0 =
                static_cast<double>(reach_p0[node_pair]);

            const double r1 =
                static_cast<double>(reach_p1[node_pair]);

            const double rc =
                static_cast<double>(reach_chance[node_pair]);

            if (player == kPlayerP0) {
                cf_sum += rc * r1;
                own_sum += rc * r0;
            } else {
                cf_sum += rc * r0;
                own_sum += rc * r1;
            }
        }

        const std::uint64_t sb =
            action_state_bucket_offset[state] +
            static_cast<std::uint64_t>(bucket);

        state_bucket_cf_reach[sb] = static_cast<float>(cf_sum);
        state_bucket_own_reach[sb] = static_cast<float>(own_sum);
    }
}
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
    cudaStream_t stream
) {
    if (num_action_states <= 0 || hand_pair_count <= 0) {
        return;
    }

    public_aggregate_state_bucket_reaches_kernel<<<
        num_action_states,
        config.threads_per_block,
        0,
        stream
    >>>(
        num_action_states,

        d_action_state_node,
        d_action_state_player,
        d_action_state_bucket_count,
        d_action_state_bucket_offset,

        hand_pair_count,
        d_p0_pair_index,
        d_p1_pair_index,
        d_p0_bucket_by_hand_index,
        d_p1_bucket_by_hand_index,

        d_node_pair_reach_p0,
        d_node_pair_reach_p1,
        d_node_pair_reach_chance,

        d_state_bucket_cf_reach,
        d_state_bucket_own_reach
    );
}


// -----------------------------------------------------------------------------
// Action value extraction
// -----------------------------------------------------------------------------

__global__ void public_compute_action_values_from_pair_values_kernel(
    int action_edge_count,

    const int* action_edge_parent,
    const int* action_edge_child,
    const int* action_edge_state,
    const int* action_edge_local_action,

    const int* action_state_player,
    const int* action_state_bucket_count,
    const int* action_state_action_count,
    const std::uint64_t* action_state_tensor_offset,
    const std::uint64_t* action_state_bucket_offset,

    int hand_pair_count,
    const int* p0_pair_index,
    const int* p1_pair_index,
    const int* p0_bucket_by_hand_index,
    const int* p1_bucket_by_hand_index,

    const float* node_pair_value_p0,

    const float* node_pair_reach_p0,
    const float* node_pair_reach_p1,
    const float* node_pair_reach_chance,
    const float* state_bucket_cf_reach,

    float* action_value_p0
) {
    const int action_edge = blockIdx.x;

    if (action_edge >= action_edge_count) {
        return;
    }

    const int parent = action_edge_parent[action_edge];
    const int child = action_edge_child[action_edge];
    const int state = action_edge_state[action_edge];
    const int local_action = action_edge_local_action[action_edge];

    const int player = action_state_player[state];
    const int bucket_count = action_state_bucket_count[state];
    const int action_count = action_state_action_count[state];

    if (player != kPlayerP0 && player != kPlayerP1) {
        return;
    }

    if (local_action < 0 || local_action >= action_count) {
        return;
    }

    for (int bucket = threadIdx.x;
         bucket < bucket_count;
         bucket += blockDim.x) {
        const std::uint64_t state_bucket_index =
            action_state_bucket_offset[state] +
            static_cast<std::uint64_t>(bucket);

        const float cf_reach =
            state_bucket_cf_reach[state_bucket_index];

        double weighted_sum = 0.0;

        if (cf_reach > 0.0f) {
            for (int pair_id = 0;
                 pair_id < hand_pair_count;
                 ++pair_id) {
                int pair_bucket = -1;

                if (player == kPlayerP0) {
                    const int p0_hand_index =
                        p0_pair_index[pair_id];

                    pair_bucket =
                        p0_bucket_by_hand_index[p0_hand_index];
                } else {
                    const int p1_hand_index =
                        p1_pair_index[pair_id];

                    pair_bucket =
                        p1_bucket_by_hand_index[p1_hand_index];
                }

                if (pair_bucket != bucket) {
                    continue;
                }

                const std::size_t parent_pair_index =
                    static_cast<std::size_t>(parent) *
                        static_cast<std::size_t>(hand_pair_count) +
                    static_cast<std::size_t>(pair_id);

                const std::size_t child_pair_index =
                    static_cast<std::size_t>(child) *
                        static_cast<std::size_t>(hand_pair_count) +
                    static_cast<std::size_t>(pair_id);

                const float chance_reach =
                    node_pair_reach_chance[parent_pair_index];

                const float opponent_reach =
                    player == kPlayerP0
                        ? node_pair_reach_p1[parent_pair_index]
                        : node_pair_reach_p0[parent_pair_index];

                const float counterfactual_weight =
                    chance_reach * opponent_reach;

                weighted_sum +=
                    static_cast<double>(counterfactual_weight) *
                    static_cast<double>(node_pair_value_p0[child_pair_index]);
            }
        }

        const float value =
            cf_reach > 0.0f
                ? static_cast<float>(
                    weighted_sum / static_cast<double>(cf_reach)
                  )
                : 0.0f;

        const std::uint64_t tensor_index =
            action_state_tensor_offset[state] +
            static_cast<std::uint64_t>(bucket) *
                static_cast<std::uint64_t>(action_count) +
            static_cast<std::uint64_t>(local_action);

        action_value_p0[tensor_index] = value;
    }
}

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

    cudaStream_t stream
) {
    if (action_edges.count <= 0 || hand_pair_count <= 0) {
        return;
    }

    public_compute_action_values_from_pair_values_kernel<<<
        action_edges.count,
        config.threads_per_block,
        0,
        stream
    >>>(
        action_edges.count,

        action_edges.d_parent,
        action_edges.d_child,
        action_edges.d_action_state,
        action_edges.d_local_action,

        d_action_state_player,
        d_action_state_bucket_count,
        d_action_state_action_count,
        d_action_state_tensor_offset,
        d_action_state_bucket_offset,

        hand_pair_count,
        d_p0_pair_index,
        d_p1_pair_index,
        d_p0_bucket_by_hand_index,
        d_p1_bucket_by_hand_index,

        d_node_pair_value_p0,

        d_node_pair_reach_p0,
        d_node_pair_reach_p1,
        d_node_pair_reach_chance,
        d_state_bucket_cf_reach,

        d_action_value_p0
    );
}

// -----------------------------------------------------------------------------
// Regret update
// -----------------------------------------------------------------------------

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

    cudaStream_t stream
) {
    if (num_action_states <= 0) {
        return;
    }

    public_update_regrets_kernel<<<
        num_action_states,
        config.threads_per_block,
        0,
        stream
    >>>(
        d_action_state_player,
        d_action_state_bucket_count,
        d_action_state_action_count,
        d_action_state_tensor_offset,
        d_action_state_bucket_offset,

        d_action_value_p0,
        d_state_bucket_value_p0,
        d_state_bucket_cf_reach,

        d_sigma,
        d_regret_sum,

        num_action_states,
        use_cfr_plus
    );
}

// -----------------------------------------------------------------------------
// Average strategy accumulation
// -----------------------------------------------------------------------------

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

    cudaStream_t stream
) {
    if (num_action_states <= 0) {
        return;
    }

    public_accumulate_average_strategy_kernel<<<
        num_action_states,
        config.threads_per_block,
        0,
        stream
    >>>(
        d_action_state_bucket_count,
        d_action_state_action_count,
        d_action_state_tensor_offset,
        d_action_state_bucket_offset,

        d_state_bucket_own_reach,
        d_sigma,

        d_strategy_sum,
        d_strategy_weight_sum,

        num_action_states,
        iteration_weight
    );
}

// -----------------------------------------------------------------------------
// Average strategy normalization
// -----------------------------------------------------------------------------

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

    cudaStream_t stream
) {
    if (num_action_states <= 0) {
        return;
    }

    public_normalize_average_strategy_kernel<<<
        num_action_states,
        config.threads_per_block,
        0,
        stream
    >>>(
        d_action_state_bucket_count,
        d_action_state_action_count,
        d_action_state_tensor_offset,
        d_action_state_bucket_offset,

        d_strategy_sum,
        d_strategy_weight_sum,

        d_avg_strategy,

        num_action_states
    );
}

// -----------------------------------------------------------------------------
// Regret matching
// -----------------------------------------------------------------------------

void launch_public_regret_matching(
    const KernelLaunchConfig& config,

    const int* d_action_state_bucket_count,
    const int* d_action_state_action_count,
    const std::uint64_t* d_action_state_tensor_offset,

    const float* d_regret_sum,
    float* d_sigma,
    const float* d_sigma_init,

    int num_action_states,

    cudaStream_t stream
) {
    if (num_action_states <= 0) {
        return;
    }

    public_regret_matching_kernel<<<
        num_action_states,
        config.threads_per_block,
        0,
        stream
    >>>(
        d_action_state_bucket_count,
        d_action_state_action_count,
        d_action_state_tensor_offset,

        d_regret_sum,
        d_sigma,
        d_sigma_init,

        num_action_states
    );
}

} // namespace poker