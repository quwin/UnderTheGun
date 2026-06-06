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

extern "C" __device__ int cuda_evaluate_7cards(
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
__device__ unsigned long long card_bit_device(int card) {
    return 1ULL << static_cast<unsigned long long>(card);
}

__device__ bool card_is_dead_device(
    unsigned long long dead_mask,
    int card
) {
    return (dead_mask & card_bit_device(card)) != 0ULL;
}
__device__ float terminal_showdown_utility(
    int pot,
    int p0_committed,

    int p0a,
    int p0b,
    int p1a,
    int p1b,

    int b0,
    int b1,
    int b2,
    int b3,
    int b4,

    short* d_binaries_by_id,
    short* d_suitbit_by_id,
    short* d_flush,
    short* d_noflush7,
    unsigned char* d_suits,
    int* d_dp
) {
    const int p0_rank = cuda_evaluate_7cards(
        p0a,
        p0b,
        b0,
        b1,
        b2,
        b3,
        b4,
        d_binaries_by_id,
        d_suitbit_by_id,
        d_flush,
        d_noflush7,
        d_suits,
        d_dp
    );
    const int p1_rank = cuda_evaluate_7cards(
        p1a,
        p1b,
        b0,
        b1,
        b2,
        b3,
        b4,
        d_binaries_by_id,
        d_suitbit_by_id,
        d_flush,
        d_noflush7,
        d_suits,
        d_dp
    );
    // Smaller rank is stronger.
    if (p0_rank < p1_rank) {
        return terminal_win_utility(pot, p0_committed);
    }

    if (p1_rank < p0_rank) {
        return terminal_loss_utility(p0_committed);
    }

    return terminal_tie_utility(pot, p0_committed);
}
__device__ float terminal_all_in_runout_utility(
    int pot,
    int p0_committed,

    int p0a,
    int p0b,
    int p1a,
    int p1b,

    int b0,
    int b1,
    int b2,
    int b3,
    int b4,

    short* d_binaries_by_id,
    short* d_suitbit_by_id,
    short* d_flush,
    short* d_noflush7,
    unsigned char* d_suits,
    int* d_dp
) {
    constexpr int kMissingCard = 52;
    unsigned long long dead_mask = 0ULL;
    dead_mask |= card_bit_device(p0a);
    dead_mask |= card_bit_device(p0b);
    dead_mask |= card_bit_device(p1a);
    dead_mask |= card_bit_device(p1b);
    dead_mask |= card_bit_device(b0);
    dead_mask |= card_bit_device(b1);
    dead_mask |= card_bit_device(b2);
    const bool has_turn = b3 != kMissingCard;
    const bool has_river = b4 != kMissingCard;
    if (has_turn) {
        dead_mask |= card_bit_device(b3);
    }
    if (has_river) {
        dead_mask |= card_bit_device(b4);
    }
    // River board: direct showdown.
    if (has_turn && has_river) {
        return terminal_showdown_utility(
            pot,
            p0_committed,
            p0a,
            p0b,
            p1a,
            p1b,
            b0,
            b1,
            b2,
            b3,
            b4,
            d_binaries_by_id,
            d_suitbit_by_id,
            d_flush,
            d_noflush7,
            d_suits,
            d_dp
        );
    }
    float value_sum = 0.0f;
    int runout_count = 0;
    // Turn board: enumerate one river card.
    if (has_turn && !has_river) {
        for (int river = 0; river < 52; ++river) {
            if (card_is_dead_device(dead_mask, river)) {
                continue;
            }
            value_sum += terminal_showdown_utility(
                pot,
                p0_committed,
                p0a,
                p0b,
                p1a,
                p1b,
                b0,
                b1,
                b2,
                b3,
                river,
                d_binaries_by_id,
                d_suitbit_by_id,
                d_flush,
                d_noflush7,
                d_suits,
                d_dp
            );
            ++runout_count;
        }
        return runout_count > 0 ? value_sum / static_cast<float>(runout_count) : 0.0f;
    }
    // Flop board: enumerate unordered turn-river combinations.
    if (!has_turn && !has_river) {
        for (int turn = 0; turn < 52; ++turn) {
            if (card_is_dead_device(dead_mask, turn)) {
                continue;
            }
            const unsigned long long dead_with_turn = dead_mask | card_bit_device(turn);
            for (int river = turn + 1; river < 52; ++river) {
                if (card_is_dead_device(dead_with_turn, river)) {
                    continue;
                }
                value_sum += terminal_showdown_utility(
                    pot,
                    p0_committed,
                    p0a,
                    p0b,
                    p1a,
                    p1b,
                    b0,
                    b1,
                    b2,
                    turn,
                    river,
                    d_binaries_by_id,
                    d_suitbit_by_id,
                    d_flush,
                    d_noflush7,
                    d_suits,
                    d_dp
                );
                ++runout_count;
            }
        }
        return runout_count > 0 ? value_sum / static_cast<float>(runout_count): 0.0f;
    }
    return 0.0f;
}
__global__ void compute_terminal_pair_values_from_records_chunk_kernel(
    int terminal_count,
    int pair_start,
    int active_pair_count,
    int pair_chunk_size,

    const int* __restrict__ d_terminal_nodes,
    const int* __restrict__ d_terminal_type,
    const int* __restrict__ d_terminal_pot,
    const int* __restrict__ d_terminal_p0_committed,
    const unsigned char* __restrict__ d_terminal_board_cards,

    const int* __restrict__ d_p0_pair_index,
    const int* __restrict__ d_p1_pair_index,

    const unsigned char* __restrict__ d_p0_hand_card0,
    const unsigned char* __restrict__ d_p0_hand_card1,
    const unsigned char* __restrict__ d_p1_hand_card0,
    const unsigned char* __restrict__ d_p1_hand_card1,

    short* d_binaries_by_id,
    short* d_suitbit_by_id,
    short* d_flush,
    short* d_noflush7,
    unsigned char* d_suits,
    int* d_dp,

    float* __restrict__ d_node_pair_value_p0
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = terminal_count * active_pair_count;
    if (idx >= total) {
        return;
    }
    const int terminal_index = idx / active_pair_count;
    const int local_pair = idx - terminal_index * active_pair_count;
    const int global_pair = pair_start + local_pair;
    const int node_id = d_terminal_nodes[terminal_index];
    const int type = d_terminal_type[terminal_index];
    const int pot = d_terminal_pot[terminal_index];
    const int p0_committed = d_terminal_p0_committed[terminal_index];

    float value = 0.0f;

    if (type == static_cast<int>(TerminalType::P0_Fold)) {
        value = terminal_loss_utility(p0_committed);
    } else if (type == static_cast<int>(TerminalType::P1_Fold)) {
        value = terminal_win_utility(pot, p0_committed);
    } else if (type == static_cast<int>(TerminalType::Showdown)) {
        const int p0_i = d_p0_pair_index[global_pair];
        const int p1_i = d_p1_pair_index[global_pair];
        const int p0a = d_p0_hand_card0[p0_i];
        const int p0b = d_p0_hand_card1[p0_i];
        const int p1a = d_p1_hand_card0[p1_i];
        const int p1b = d_p1_hand_card1[p1_i];
        const int bo = terminal_index * 5;
        value = terminal_showdown_utility(
            pot,
            p0_committed,
            p0a,
            p0b,
            p1a,
            p1b,
            d_terminal_board_cards[bo + 0],
            d_terminal_board_cards[bo + 1],
            d_terminal_board_cards[bo + 2],
            d_terminal_board_cards[bo + 3],
            d_terminal_board_cards[bo + 4],
            d_binaries_by_id,
            d_suitbit_by_id,
            d_flush,
            d_noflush7,
            d_suits,
            d_dp
        );
    } else if (type == static_cast<int>(TerminalType::AllIn)) {
        const int p0_i = d_p0_pair_index[global_pair];
        const int p1_i = d_p1_pair_index[global_pair];
        const int p0a = d_p0_hand_card0[p0_i];
        const int p0b = d_p0_hand_card1[p0_i];
        const int p1a = d_p1_hand_card0[p1_i];
        const int p1b = d_p1_hand_card1[p1_i];
        const int bo = terminal_index * 5;
        value = terminal_all_in_runout_utility(
            pot,
            p0_committed,
            p0a,
            p0b,
            p1a,
            p1b,
            d_terminal_board_cards[bo + 0],
            d_terminal_board_cards[bo + 1],
            d_terminal_board_cards[bo + 2],
            d_terminal_board_cards[bo + 3],
            d_terminal_board_cards[bo + 4],
            d_binaries_by_id,
            d_suitbit_by_id,
            d_flush,
            d_noflush7,
            d_suits,
            d_dp
        );
    }
    d_node_pair_value_p0[static_cast<std::size_t>(node_id) * static_cast<std::size_t>(pair_chunk_size) +static_cast<std::size_t>(local_pair)] = value;
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

void launch_compute_terminal_pair_values_from_records_chunk(
    const KernelLaunchConfig& config,

    int terminal_count,
    int pair_start,
    int active_pair_count,
    int pair_chunk_size,

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
    if (terminal_count <= 0 || active_pair_count <= 0) {
        return;
    }

    if (pair_chunk_size <= 0 || active_pair_count > pair_chunk_size) {
        return;
    }

    const std::size_t total = static_cast<std::size_t>(terminal_count) * static_cast<std::size_t>(active_pair_count);

    compute_terminal_pair_values_from_records_chunk_kernel<<<
        blocks_for_size(total, config.threads_per_block),
        config.threads_per_block,
        0,
        stream
    >>>(
        terminal_count,
        pair_start,
        active_pair_count,
        pair_chunk_size,

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
__global__ void public_backward_pair_value_level_chunk_kernel(
    int edge_count,

    const int* __restrict__ d_edge_parent,
    const int* __restrict__ d_edge_child,
    const int* __restrict__ d_edge_local_action,
    const float* __restrict__ d_edge_chance_prob,

    const int* __restrict__ d_node_type,
    const int* __restrict__ d_player,
    const int* __restrict__ d_action_state_index,

    const int* __restrict__ d_action_state_action_count,
    const std::uint64_t* __restrict__ d_action_state_tensor_offset,

    const int* __restrict__ d_p0_pair_index,
    const int* __restrict__ d_p1_pair_index,
    const int* __restrict__ d_p0_bucket_by_hand_index,
    const int* __restrict__ d_p1_bucket_by_hand_index,

    const float* __restrict__ d_sigma,

    const float* __restrict__ d_node_pair_value_read,
    float* __restrict__ d_node_pair_value_write,

    int pair_start,
    int active_pair_count,
    int pair_chunk_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = edge_count * active_pair_count;

    if (idx >= total) {
        return;
    }

    const int edge_i = idx / active_pair_count;
    const int local_pair = idx - edge_i * active_pair_count;
    const int global_pair = pair_start + local_pair;

    const int parent = d_edge_parent[edge_i];
    const int child = d_edge_child[edge_i];

    const int parent_type = d_node_type[parent];
    const int parent_player = d_player[parent];

    if (parent_type == static_cast<int>(PublicNodeType::Terminal)) {
        return;
    }

    float edge_weight = 1.0f;

    if (parent_player == static_cast<int>(Player::Chance)) {
        edge_weight = d_edge_chance_prob[edge_i];
    } else if (
        parent_player == static_cast<int>(Player::P0) ||
        parent_player == static_cast<int>(Player::P1)
    ) {
        const int state = d_action_state_index[parent];

        if (state < 0) {
            return;
        }

        const int action_count =
            d_action_state_action_count[state];

        const int local_action =
            d_edge_local_action[edge_i];

        if (local_action < 0 || local_action >= action_count) {
            return;
        }

        int bucket = -1;

        if (parent_player == static_cast<int>(Player::P0)) {
            const int p0_index = d_p0_pair_index[global_pair];
            bucket = d_p0_bucket_by_hand_index[p0_index];
        } else {
            const int p1_index = d_p1_pair_index[global_pair];
            bucket = d_p1_bucket_by_hand_index[p1_index];
        }

        if (bucket < 0) {
            return;
        }

        const std::uint64_t sigma_idx =
            d_action_state_tensor_offset[state] +
            static_cast<std::uint64_t>(bucket) *
            static_cast<std::uint64_t>(action_count) +
            static_cast<std::uint64_t>(local_action);

        edge_weight = d_sigma[sigma_idx];
    } else {
        return;
    }

    const std::size_t parent_idx =
        static_cast<std::size_t>(parent) *
        static_cast<std::size_t>(pair_chunk_size) +
        static_cast<std::size_t>(local_pair);

    const std::size_t child_idx =
        static_cast<std::size_t>(child) *
        static_cast<std::size_t>(pair_chunk_size) +
        static_cast<std::size_t>(local_pair);

    atomicAdd(
        &d_node_pair_value_write[parent_idx],
        edge_weight * d_node_pair_value_read[child_idx]
    );
}
// -----------------------------------------------------------------------------
// Backward value pass
// -----------------------------------------------------------------------------

    void launch_public_backward_pair_value_level_chunk(
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

        int pair_start,
        int active_pair_count,
        int pair_chunk_size,

        cudaStream_t stream
    ) {
    if (edges.count <= 0 || active_pair_count <= 0) {
        return;
    }

    if (pair_start < 0 ||
        pair_chunk_size <= 0 ||
        active_pair_count > pair_chunk_size) {
        return;
        }

    const int total = edges.count * active_pair_count;

    public_backward_pair_value_level_chunk_kernel<<<
        blocks_for(total, config.threads_per_block),
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

        pair_start,
        active_pair_count,
        pair_chunk_size
    );
}

// -----------------------------------------------------------------------------
// Reach computation
// -----------------------------------------------------------------------------

__global__ void initialize_public_pair_reaches_chunk_kernel(
    int root,
    int active_pair_count,
    int pair_chunk_size,

    float* __restrict__ d_node_pair_reach_p0,
    float* __restrict__ d_node_pair_reach_p1,
    float* __restrict__ d_node_pair_reach_chance
) {
    const int local_pair =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (local_pair >= active_pair_count) {
        return;
    }

    const std::size_t idx =
        static_cast<std::size_t>(root) *
        static_cast<std::size_t>(pair_chunk_size) +
        static_cast<std::size_t>(local_pair);

    d_node_pair_reach_p0[idx] = 1.0f;
    d_node_pair_reach_p1[idx] = 1.0f;
    d_node_pair_reach_chance[idx] = 1.0f;
}

    void launch_initialize_public_pair_reaches_chunk(
        const KernelLaunchConfig& config,

        int root,
        int active_pair_count,
        int pair_chunk_size,

        float* d_node_pair_reach_p0,
        float* d_node_pair_reach_p1,
        float* d_node_pair_reach_chance,

        cudaStream_t stream
    ) {
    if (active_pair_count <= 0) {
        return;
    }

    if (root < 0 || pair_chunk_size <= 0 ||
        active_pair_count > pair_chunk_size) {
        return;
        }

    initialize_public_pair_reaches_chunk_kernel<<<
        blocks_for(active_pair_count, config.threads_per_block),
        config.threads_per_block,
        0,
        stream
    >>>(
        root,
        active_pair_count,
        pair_chunk_size,

        d_node_pair_reach_p0,
        d_node_pair_reach_p1,
        d_node_pair_reach_chance
    );
}
__global__ void public_forward_pair_reach_level_chunk_kernel(
    int edge_count,

    const int* __restrict__ d_edge_parent,
    const int* __restrict__ d_edge_child,
    const int* __restrict__ d_edge_local_action,
    const float* __restrict__ d_edge_chance_prob,

    const int* __restrict__ d_node_type,
    const int* __restrict__ d_player,
    const int* __restrict__ d_action_state_index,

    const int* __restrict__ d_action_state_action_count,
    const std::uint64_t* __restrict__ d_action_state_tensor_offset,

    const int* __restrict__ d_p0_pair_index,
    const int* __restrict__ d_p1_pair_index,
    const int* __restrict__ d_p0_bucket_by_hand_index,
    const int* __restrict__ d_p1_bucket_by_hand_index,

    const float* __restrict__ d_sigma,

    float* __restrict__ d_node_pair_reach_p0,
    float* __restrict__ d_node_pair_reach_p1,
    float* __restrict__ d_node_pair_reach_chance,

    int pair_start,
    int active_pair_count,
    int pair_chunk_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = edge_count * active_pair_count;

    if (idx >= total) {
        return;
    }

    const int edge_i = idx / active_pair_count;
    const int local_pair = idx - edge_i * active_pair_count;
    const int global_pair = pair_start + local_pair;

    const int parent = d_edge_parent[edge_i];
    const int child = d_edge_child[edge_i];

    const int parent_type = d_node_type[parent];
    const int parent_player = d_player[parent];

    if (parent_type == static_cast<int>(PublicNodeType::Terminal)) {
        return;
    }

    const std::size_t parent_idx =
        static_cast<std::size_t>(parent) *
        static_cast<std::size_t>(pair_chunk_size) +
        static_cast<std::size_t>(local_pair);

    const std::size_t child_idx =
        static_cast<std::size_t>(child) *
        static_cast<std::size_t>(pair_chunk_size) +
        static_cast<std::size_t>(local_pair);

    const float parent_reach_p0 =
        d_node_pair_reach_p0[parent_idx];

    const float parent_reach_p1 =
        d_node_pair_reach_p1[parent_idx];

    const float parent_reach_chance =
        d_node_pair_reach_chance[parent_idx];

    if (parent_player == static_cast<int>(Player::Chance)) {
        const float chance_prob = d_edge_chance_prob[edge_i];

        d_node_pair_reach_p0[child_idx] =
            parent_reach_p0;

        d_node_pair_reach_p1[child_idx] =
            parent_reach_p1;

        d_node_pair_reach_chance[child_idx] =
            parent_reach_chance * chance_prob;

        return;
    }

    if (parent_player != static_cast<int>(Player::P0) &&
        parent_player != static_cast<int>(Player::P1)) {
        return;
    }

    const int state = d_action_state_index[parent];

    if (state < 0) {
        return;
    }

    const int action_count =
        d_action_state_action_count[state];

    const int local_action =
        d_edge_local_action[edge_i];

    if (local_action < 0 || local_action >= action_count) {
        return;
    }

    int bucket = -1;

    if (parent_player == static_cast<int>(Player::P0)) {
        const int p0_index = d_p0_pair_index[global_pair];
        bucket = d_p0_bucket_by_hand_index[p0_index];
    } else {
        const int p1_index = d_p1_pair_index[global_pair];
        bucket = d_p1_bucket_by_hand_index[p1_index];
    }

    if (bucket < 0) {
        return;
    }

    const std::uint64_t sigma_idx =
        d_action_state_tensor_offset[state] +
        static_cast<std::uint64_t>(bucket) *
        static_cast<std::uint64_t>(action_count) +
        static_cast<std::uint64_t>(local_action);

    const float action_prob = d_sigma[sigma_idx];

    if (parent_player == static_cast<int>(Player::P0)) {
        d_node_pair_reach_p0[child_idx] =
            parent_reach_p0 * action_prob;

        d_node_pair_reach_p1[child_idx] =
            parent_reach_p1;

        d_node_pair_reach_chance[child_idx] =
            parent_reach_chance;
    } else {
        d_node_pair_reach_p0[child_idx] =
            parent_reach_p0;

        d_node_pair_reach_p1[child_idx] =
            parent_reach_p1 * action_prob;

        d_node_pair_reach_chance[child_idx] =
            parent_reach_chance;
    }
}
void launch_public_forward_pair_reach_level_chunk(
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

    int pair_start,
    int active_pair_count,
    int pair_chunk_size,

    cudaStream_t stream
) {
    if (edges.count <= 0 || active_pair_count <= 0) {
        return;
    }

    if (pair_start < 0 ||
        pair_chunk_size <= 0 ||
        active_pair_count > pair_chunk_size) {
        return;
        }

    const int total = edges.count * active_pair_count;

    public_forward_pair_reach_level_chunk_kernel<<<
        blocks_for(total, config.threads_per_block),
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

        pair_start,
        active_pair_count,
        pair_chunk_size
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

__global__ void aggregate_state_bucket_reaches_chunk_kernel(
    int num_action_states,

    const int* __restrict__ d_action_state_node,
    const int* __restrict__ d_action_state_player,
    const int* __restrict__ d_action_state_bucket_count,
    const std::uint64_t* __restrict__ d_action_state_bucket_offset,

    int pair_start,
    int active_pair_count,
    int pair_chunk_size,

    const int* __restrict__ d_p0_pair_index,
    const int* __restrict__ d_p1_pair_index,
    const int* __restrict__ d_p0_bucket_by_hand_index,
    const int* __restrict__ d_p1_bucket_by_hand_index,

    const float* __restrict__ d_node_pair_reach_p0,
    const float* __restrict__ d_node_pair_reach_p1,
    const float* __restrict__ d_node_pair_reach_chance,

    float* __restrict__ d_state_bucket_own_reach
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_action_states * active_pair_count;

    if (idx >= total) {
        return;
    }

    const int state = idx / active_pair_count;
    const int local_pair = idx - state * active_pair_count;
    const int global_pair = pair_start + local_pair;

    const int node_id = d_action_state_node[state];
    const int player = d_action_state_player[state];
    const int bucket_count = d_action_state_bucket_count[state];

    if (node_id < 0 || bucket_count <= 0) {
        return;
    }

    int bucket = -1;

    const int p0_index = d_p0_pair_index[global_pair];
    const int p1_index = d_p1_pair_index[global_pair];

    if (player == static_cast<int>(Player::P0)) {
        bucket = d_p0_bucket_by_hand_index[p0_index];
    } else if (player == static_cast<int>(Player::P1)) {
        bucket = d_p1_bucket_by_hand_index[p1_index];
    } else {
        return;
    }

    if (bucket < 0 || bucket >= bucket_count) {
        return;
    }

    const std::size_t node_pair_idx =
        static_cast<std::size_t>(node_id) *
        static_cast<std::size_t>(pair_chunk_size) +
        static_cast<std::size_t>(local_pair);

    const float reach_p0 = d_node_pair_reach_p0[node_pair_idx];

    const float reach_p1 = d_node_pair_reach_p1[node_pair_idx];

    float own_reach = 0.0f;

    if (player == static_cast<int>(Player::P0)) {
        // Own reach is used for average-strategy weighting.
        own_reach = reach_p0;
    } else {
        // Own reach is used for average-strategy weighting.
        own_reach = reach_p1;
    }

    const std::uint64_t state_bucket_idx =
        d_action_state_bucket_offset[state] +
        static_cast<std::uint64_t>(bucket);

    atomicAdd(
        &d_state_bucket_own_reach[state_bucket_idx],
        own_reach
    );
}
    void launch_public_aggregate_state_bucket_reaches_chunk(
        const KernelLaunchConfig& config,

        const int* d_action_state_node,
        const int* d_action_state_player,
        const int* d_action_state_bucket_count,
        const std::uint64_t* d_action_state_bucket_offset,

        int pair_start,
        int active_pair_count,
        int pair_chunk_size,

        const int* d_p0_pair_index,
        const int* d_p1_pair_index,
        const int* d_p0_bucket_by_hand_index,
        const int* d_p1_bucket_by_hand_index,

        const float* d_node_pair_reach_p0,
        const float* d_node_pair_reach_p1,
        const float* d_node_pair_reach_chance,

        float* d_state_bucket_own_reach,

        int num_action_states,

        cudaStream_t stream
    ) {
    if (num_action_states <= 0 || active_pair_count <= 0) {
        return;
    }

    if (pair_start < 0 ||
        pair_chunk_size <= 0 ||
        active_pair_count > pair_chunk_size) {
        return;
        }

    const int total = num_action_states * active_pair_count;

    aggregate_state_bucket_reaches_chunk_kernel<<<
        blocks_for(total, config.threads_per_block),
        config.threads_per_block,
        0,
        stream
    >>>(
        num_action_states,

        d_action_state_node,
        d_action_state_player,
        d_action_state_bucket_count,
        d_action_state_bucket_offset,

        pair_start,
        active_pair_count,
        pair_chunk_size,

        d_p0_pair_index,
        d_p1_pair_index,
        d_p0_bucket_by_hand_index,
        d_p1_bucket_by_hand_index,

        d_node_pair_reach_p0,
        d_node_pair_reach_p1,
        d_node_pair_reach_chance,

        d_state_bucket_own_reach
    );
}
// -----------------------------------------------------------------------------
// Action value extraction
// -----------------------------------------------------------------------------

__global__ void compute_action_values_from_pair_values_chunk_kernel(
    int action_edge_count,

    const int* __restrict__ d_action_edge_parent,
    const int* __restrict__ d_action_edge_child,
    const int* __restrict__ d_action_edge_state,
    const int* __restrict__ d_action_edge_local_action,

    const int* __restrict__ d_action_state_player,
    const int* __restrict__ d_action_state_bucket_count,
    const int* __restrict__ d_action_state_action_count,
    const std::uint64_t* __restrict__ d_action_state_tensor_offset,
    const std::uint64_t* __restrict__ d_action_state_bucket_offset,

    const int* __restrict__ d_p0_pair_index,
    const int* __restrict__ d_p1_pair_index,
    const int* __restrict__ d_p0_bucket_by_hand_index,
    const int* __restrict__ d_p1_bucket_by_hand_index,

    const float* __restrict__ d_node_pair_value_p0,
    const float* __restrict__ d_node_pair_reach_p0,
    const float* __restrict__ d_node_pair_reach_p1,
    const float* __restrict__ d_node_pair_reach_chance,

    int pair_start,
    int active_pair_count,
    int pair_chunk_size,

    float* __restrict__ d_action_value_p0,
    float* __restrict__ d_state_bucket_value_p0
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = action_edge_count * active_pair_count;

    if (idx >= total) {
        return;
    }

    const int edge_i = idx / active_pair_count;
    const int local_pair = idx - edge_i * active_pair_count;
    const int global_pair = pair_start + local_pair;

    const int parent = d_action_edge_parent[edge_i];
    const int child = d_action_edge_child[edge_i];
    const int state = d_action_edge_state[edge_i];
    const int local_action = d_action_edge_local_action[edge_i];

    const int acting_player = d_action_state_player[state];
    const int action_count = d_action_state_action_count[state];

    int bucket = -1;
    float cf_reach = 0.0f;

    const std::size_t parent_pair_idx =
        static_cast<std::size_t>(parent) *
        static_cast<std::size_t>(pair_chunk_size) +
        static_cast<std::size_t>(local_pair);

    const std::size_t child_pair_idx =
        static_cast<std::size_t>(child) *
        static_cast<std::size_t>(pair_chunk_size) +
        static_cast<std::size_t>(local_pair);

    const float reach_p0 = d_node_pair_reach_p0[parent_pair_idx];
    const float reach_p1 = d_node_pair_reach_p1[parent_pair_idx];
    const float reach_chance = d_node_pair_reach_chance[parent_pair_idx];

    if (acting_player == static_cast<int>(Player::P0)) {
        const int p0_i = d_p0_pair_index[global_pair];
        bucket = d_p0_bucket_by_hand_index[p0_i];

        // Counterfactual reach for P0 regret:
        // opponent reach * chance reach.
        cf_reach = reach_p1 * reach_chance;
    } else if (acting_player == static_cast<int>(Player::P1)) {
        const int p1_i = d_p1_pair_index[global_pair];
        bucket = d_p1_bucket_by_hand_index[p1_i];

        // Counterfactual reach for P1 regret:
        // opponent reach * chance reach.
        cf_reach = reach_p0 * reach_chance;
    } else {
        return;
    }

    if (bucket < 0 || bucket >= d_action_state_bucket_count[state]) {
        return;
    }

    const std::uint64_t tensor_idx =
        d_action_state_tensor_offset[state] +
        static_cast<std::uint64_t>(bucket) *
        static_cast<std::uint64_t>(action_count) +
        static_cast<std::uint64_t>(local_action);

    const std::uint64_t state_bucket_idx =
        d_action_state_bucket_offset[state] +
        static_cast<std::uint64_t>(bucket);

    const float parent_value_p0 =
        d_node_pair_value_p0[parent_pair_idx];

    const float child_value_p0 =
        d_node_pair_value_p0[child_pair_idx];

    // Store P0-perspective weighted values.
    //
    // For P0 regrets:
    //   regret = action_value_p0 - state_value_p0
    //
    // For P1 regrets:
    //   regret from P1 perspective is:
    //      (-action_value_p0) - (-state_value_p0)
    //    = state_value_p0 - action_value_p0
    atomicAdd(
        &d_action_value_p0[tensor_idx],
        cf_reach * child_value_p0
    );

    atomicAdd(
        &d_state_bucket_value_p0[state_bucket_idx],
        cf_reach * parent_value_p0
    );
}

void launch_public_compute_action_values_from_pair_values_chunk(
    const KernelLaunchConfig& config,

    const DevicePublicActionEdges& action_edges,

    const int* d_action_state_player,
    const int* d_action_state_bucket_count,
    const int* d_action_state_action_count,
    const std::uint64_t* d_action_state_tensor_offset,
    const std::uint64_t* d_action_state_bucket_offset,

    const int* d_p0_pair_index,
    const int* d_p1_pair_index,
    const int* d_p0_bucket_by_hand_index,
    const int* d_p1_bucket_by_hand_index,

    const float* d_node_pair_value_p0,
    const float* d_node_pair_reach_p0,
    const float* d_node_pair_reach_p1,
    const float* d_node_pair_reach_chance,

    int pair_start,
    int active_pair_count,
    int pair_chunk_size,

    float* d_action_value_p0,
    float* d_state_bucket_value_p0,

    cudaStream_t stream
) {
    if (action_edges.count <= 0 || active_pair_count <= 0) {
        return;
    }

    const int total = action_edges.count * active_pair_count;

    compute_action_values_from_pair_values_chunk_kernel<<<
        blocks_for(total, config.threads_per_block),
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

        d_p0_pair_index,
        d_p1_pair_index,
        d_p0_bucket_by_hand_index,
        d_p1_bucket_by_hand_index,

        d_node_pair_value_p0,
        d_node_pair_reach_p0,
        d_node_pair_reach_p1,
        d_node_pair_reach_chance,

        pair_start,
        active_pair_count,
        pair_chunk_size,

        d_action_value_p0,
        d_state_bucket_value_p0
    );
}

// -----------------------------------------------------------------------------
// Regret update
// -----------------------------------------------------------------------------
__global__ void apply_regret_deltas_kernel(
    float* d_regret_sum,
    const float* d_regret_delta,
    std::size_t tensor_entries,
    bool use_cfr_plus
) {
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) *
        static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);

    if (idx >= tensor_entries) {
        return;
    }

    float updated = d_regret_sum[idx] + d_regret_delta[idx];

    if (use_cfr_plus && updated < 0.0f) {
        updated = 0.0f;
    }

    d_regret_sum[idx] = updated;
}
void launch_apply_regret_deltas(
    const KernelLaunchConfig& config,
    float* d_regret_sum,
    const float* d_regret_delta,
    std::size_t tensor_entries,
    bool use_cfr_plus,
    cudaStream_t stream
) {
    if (tensor_entries == 0) {
        return;
    }

    apply_regret_deltas_kernel<<<
        blocks_for_size(tensor_entries, config.threads_per_block),
        config.threads_per_block,
        0,
        stream
    >>>(
        d_regret_sum,
        d_regret_delta,
        tensor_entries,
        use_cfr_plus
    );
}
__global__ void accumulate_regret_deltas_for_chunk_kernel(
    int action_edge_count,

    const int* __restrict__ d_action_edge_parent,
    const int* __restrict__ d_action_edge_child,
    const int* __restrict__ d_action_edge_state,
    const int* __restrict__ d_action_edge_local_action,

    const int* __restrict__ d_action_state_player,
    const int* __restrict__ d_action_state_bucket_count,
    const int* __restrict__ d_action_state_action_count,
    const std::uint64_t* __restrict__ d_action_state_tensor_offset,

    const int* __restrict__ d_p0_pair_index,
    const int* __restrict__ d_p1_pair_index,
    const int* __restrict__ d_p0_bucket_by_hand_index,
    const int* __restrict__ d_p1_bucket_by_hand_index,

    const float* __restrict__ d_node_pair_value_p0,
    const float* __restrict__ d_node_pair_reach_p0,
    const float* __restrict__ d_node_pair_reach_p1,
    const float* __restrict__ d_node_pair_reach_chance,

    int pair_start,
    int active_pair_count,
    int pair_chunk_size,

    float* __restrict__ d_regret_delta
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = action_edge_count * active_pair_count;

    if (idx >= total) {
        return;
    }

    const int edge_i = idx / active_pair_count;
    const int local_pair = idx - edge_i * active_pair_count;
    const int global_pair = pair_start + local_pair;

    const int parent = d_action_edge_parent[edge_i];
    const int child = d_action_edge_child[edge_i];
    const int state = d_action_edge_state[edge_i];
    const int local_action = d_action_edge_local_action[edge_i];

    const int acting_player = d_action_state_player[state];
    const int action_count = d_action_state_action_count[state];

    int bucket = -1;
    float cf_reach = 0.0f;

    const std::size_t parent_pair_idx =
        static_cast<std::size_t>(parent) *
        static_cast<std::size_t>(pair_chunk_size) +
        static_cast<std::size_t>(local_pair);

    const std::size_t child_pair_idx =
        static_cast<std::size_t>(child) *
        static_cast<std::size_t>(pair_chunk_size) +
        static_cast<std::size_t>(local_pair);

    const float reach_p0 =
        d_node_pair_reach_p0[parent_pair_idx];

    const float reach_p1 =
        d_node_pair_reach_p1[parent_pair_idx];

    const float reach_chance =
        d_node_pair_reach_chance[parent_pair_idx];

    if (acting_player == static_cast<int>(Player::P0)) {
        const int p0_i = d_p0_pair_index[global_pair];
        bucket = d_p0_bucket_by_hand_index[p0_i];

        // P0 counterfactual reach excludes P0's own reach.
        cf_reach = reach_p1 * reach_chance;
    } else if (acting_player == static_cast<int>(Player::P1)) {
        const int p1_i = d_p1_pair_index[global_pair];
        bucket = d_p1_bucket_by_hand_index[p1_i];

        // P1 counterfactual reach excludes P1's own reach.
        cf_reach = reach_p0 * reach_chance;
    } else {
        return;
    }

    if (bucket < 0 ||
        bucket >= d_action_state_bucket_count[state]) {
        return;
    }

    const float parent_value_p0 =
        d_node_pair_value_p0[parent_pair_idx];

    const float child_value_p0 =
        d_node_pair_value_p0[child_pair_idx];

    float regret_delta = 0.0f;

    if (acting_player == static_cast<int>(Player::P0)) {
        regret_delta =
            cf_reach * (child_value_p0 - parent_value_p0);
    } else {
        // Values are stored from P0 perspective.
        // P1's utility is negative P0 utility.
        regret_delta =
            cf_reach * (parent_value_p0 - child_value_p0);
    }

    const std::uint64_t tensor_idx =
        d_action_state_tensor_offset[state] +
        static_cast<std::uint64_t>(bucket) *
        static_cast<std::uint64_t>(action_count) +
        static_cast<std::uint64_t>(local_action);

    atomicAdd(
        &d_regret_delta[tensor_idx],
        regret_delta
    );
}

void launch_public_accumulate_regret_deltas_for_chunk(
    const KernelLaunchConfig& config,

    const DevicePublicActionEdges& action_edges,

    const int* d_action_state_player,
    const int* d_action_state_bucket_count,
    const int* d_action_state_action_count,
    const std::uint64_t* d_action_state_tensor_offset,

    const int* d_p0_pair_index,
    const int* d_p1_pair_index,
    const int* d_p0_bucket_by_hand_index,
    const int* d_p1_bucket_by_hand_index,

    const float* d_node_pair_value_p0,
    const float* d_node_pair_reach_p0,
    const float* d_node_pair_reach_p1,
    const float* d_node_pair_reach_chance,

    int pair_start,
    int active_pair_count,
    int pair_chunk_size,

    float* d_regret_delta,

    cudaStream_t stream
) {
    if (action_edges.count <= 0 || active_pair_count <= 0) {
        return;
    }

    const int total = action_edges.count * active_pair_count;

    accumulate_regret_deltas_for_chunk_kernel<<<
        blocks_for(total, config.threads_per_block),
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

        d_p0_pair_index,
        d_p1_pair_index,
        d_p0_bucket_by_hand_index,
        d_p1_bucket_by_hand_index,

        d_node_pair_value_p0,
        d_node_pair_reach_p0,
        d_node_pair_reach_p1,
        d_node_pair_reach_chance,

        pair_start,
        active_pair_count,
        pair_chunk_size,

        d_regret_delta
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