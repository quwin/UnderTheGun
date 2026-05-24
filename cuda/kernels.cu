#include "kernels.hpp"
#include <cuda_runtime_api.h>
#include <iostream>

namespace poker {

namespace {

constexpr float kEpsilon = 1e-12f;
constexpr int kPlayerP0 = 0;
constexpr int kPlayerP1 = 1;

__global__ void fill_float_kernel(float* values, float value, int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        values[idx] = value;
    }
}

__global__ void copy_float_kernel(const float* src, float* dst, int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = src[idx];
    }
}

__global__ void initialize_uniform_strategy_kernel(
    const int* infoset_q_begin,
    const int* infoset_q_count,
    float* sigma,
    float* sigma_init,
    int num_infosets
) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= num_infosets) {
        return;
    }

    const int begin = infoset_q_begin[h];
    const int count = infoset_q_count[h];
    if (count <= 0) {
        return;
    }

    const float p = 1.0f / static_cast<float>(count);
    for (int offset = 0; offset < count; ++offset) {
        const int q = begin + offset;
        sigma[q] = p;
        sigma_init[q] = p;
    }
}

__global__ void compute_incoming_probabilities_level_kernel(
    DeviceLevelEdges edges,
    const int* /* player */,
    const float* chance_prob,
    const float* sigma,
    float* incoming_prob
) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= edges.count) {
        return;
    }

    const int child = edges.d_child[e];
    const int q = edges.d_q_index[e];

    incoming_prob[child] = (q >= 0) ? sigma[q] : chance_prob[child];
}

__global__ void initialize_terminal_utilities_kernel(
    const float* terminal_u_p0,
    float* u_p0,
    int num_nodes
) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_nodes) {
        u_p0[v] = terminal_u_p0[v];
    }
}

__global__ void backward_utility_level_kernel(
    DeviceLevelEdges edges,
    const float* incoming_prob,
    const float* u_read,
    float* u_write
) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= edges.count) {
        return;
    }

    const int parent = edges.d_parent[e];
    const int child = edges.d_child[e];
    const float contribution = incoming_prob[child] * u_read[child];

    atomicAdd(&u_write[parent], contribution);
}

__global__ void initialize_reach_kernel(
    float* reach_p0,
    float* reach_p1,
    int num_nodes,
    int root
) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes) {
        return;
    }

    reach_p0[v] = (v == root) ? 1.0f : 0.0f;
    reach_p1[v] = (v == root) ? 1.0f : 0.0f;
}

__global__ void forward_reach_level_kernel(
    DeviceLevelEdges edges,
    const int* player,
    const float* incoming_prob,
    const float* reach_p0_read,
    const float* reach_p1_read,
    float* reach_p0_write,
    float* reach_p1_write
) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= edges.count) {
        return;
    }

    const int parent = edges.d_parent[e];
    const int child = edges.d_child[e];
    const int parent_player = player[parent];
    const float edge_prob = incoming_prob[child];

    const float factor_p0 = (parent_player == kPlayerP0) ? 1.0f : edge_prob;
    const float factor_p1 = (parent_player == kPlayerP1) ? 1.0f : edge_prob;

    reach_p0_write[child] = reach_p0_read[parent] * factor_p0;
    reach_p1_write[child] = reach_p1_read[parent] * factor_p1;
}

__global__ void compute_infoset_reach_kernel(
    const int* infoset_player,
    const int* infoset_node_begin,
    const int* infoset_node_count,
    const int* infoset_nodes,
    const float* reach_p0,
    const float* reach_p1,
    float* infoset_reach,
    int num_infosets
) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= num_infosets) {
        return;
    }

    const int owner = infoset_player[h];
    const int begin = infoset_node_begin[h];
    const int count = infoset_node_count[h];

    float sum = 0.0f;
    for (int i = 0; i < count; ++i) {
        const int node = infoset_nodes[begin + i];
        sum += (owner == kPlayerP0) ? reach_p0[node] : reach_p1[node];
    }

    infoset_reach[h] = sum;
}

__global__ void compute_instantaneous_regrets_kernel(
    DeviceDecisionEdges decision_edges,
    const int* player,
    const float* u_p0,
    const float* reach_p0,
    const float* reach_p1,
    float* inst_regret
) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= decision_edges.count) {
        return;
    }

    const int parent = decision_edges.d_parent[e];
    const int child = decision_edges.d_child[e];
    const int q = decision_edges.d_q_index[e];
    const int acting_player = player[parent];

    float regret = 0.0f;
    if (acting_player == kPlayerP0) {
        // use reach excluding P0 = opponent/chance reach
        regret = reach_p0[parent] * (u_p0[child] - u_p0[parent]);
    } else if (acting_player == kPlayerP1) {
        // use reach excluding P1 = opponent/chance reach
        regret = reach_p1[parent] * (u_p0[parent] - u_p0[child]);
    }

    atomicAdd(&inst_regret[q], regret);
}

__global__ void update_regrets_kernel(
    const float* inst_regret,
    float* regret_sum,
    int num_q,
    bool use_cfr_plus
) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= num_q) {
        return;
    }

    float updated = regret_sum[q] + inst_regret[q];
    if (use_cfr_plus && updated < 0.0f) {
        updated = 0.0f;
    }

    regret_sum[q] = updated;
}

__global__ void accumulate_positive_regret_sums_kernel(
    const float* regret_sum,
    const int* q_infoset,
    float* positive_regret_sum,
    int num_q
) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= num_q) {
        return;
    }

    const float positive = regret_sum[q] > 0.0f ? regret_sum[q] : 0.0f;
    if (positive > 0.0f) {
        atomicAdd(&positive_regret_sum[q_infoset[q]], positive);
    }
}

__global__ void regret_matching_kernel(
    const float* regret_sum,
    const float* positive_regret_sum,
    const float* sigma_init,
    const int* q_infoset,
    float* sigma,
    int num_q
) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= num_q) {
        return;
    }

    const int h = q_infoset[q];
    const float denom = positive_regret_sum[h];
    const float positive = regret_sum[q] > 0.0f ? regret_sum[q] : 0.0f;

    sigma[q] = (denom > kEpsilon) ? (positive / denom) : sigma_init[q];
}

__global__ void accumulate_average_strategy_kernel(
    const int* q_infoset,
    const int* infoset_q_begin,
    const float* sigma,
    const float* own_infoset_reach,
    float* strategy_sum,
    float* strategy_weight_sum,
    int num_q,
    float iteration_weight
) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;

    if (q >= num_q) {
        return;
    }

    const int h = q_infoset[q];
    const float weight = iteration_weight * own_infoset_reach[h];

    strategy_sum[q] += weight * sigma[q];

    // Add denominator once per infoset, not once per action.
    if (q == infoset_q_begin[h]) {
        strategy_weight_sum[h] += weight;
    }
}

__global__ void normalize_average_strategy_kernel(
    const int* q_infoset,
    const float* strategy_sum,
    const float* /* strategy_weight_sum */,
    const float* sigma_init,
    float* avg_strategy,
    int num_q
) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= num_q) {
        return;
    }

    const int h = q_infoset[q];
    float denom = 0.0f;

    for (int r = 0; r < num_q; ++r) {
        if (q_infoset[r] == h) {
            denom += strategy_sum[r];
        }
    }

    avg_strategy[q] = (denom > kEpsilon)
        ? strategy_sum[q] / denom
        : sigma_init[q];
}

} // namespace

int blocks_for(int n, int threads_per_block) {
    if (n <= 0) {
        return 1;
    }

    return (n + threads_per_block - 1) / threads_per_block;
}

void launch_fill_float(
    const KernelLaunchConfig& config,
    float* d_values,
    float value,
    int count,
    cudaStream_t stream
) {
    if (count <= 0) {
        return;
    }

    fill_float_kernel<<<blocks_for(count, config.threads_per_block),
                        config.threads_per_block,
                        0,
                        stream>>>(d_values, value, count);
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

    copy_float_kernel<<<blocks_for(count, config.threads_per_block),
                        config.threads_per_block,
                        0,
                        stream>>>(d_src, d_dst, count);
}

void launch_initialize_uniform_strategy(
    const KernelLaunchConfig& config,
    const int* d_infoset_q_begin,
    const int* d_infoset_q_count,
    float* d_sigma,
    float* d_sigma_init,
    int num_infosets,
    cudaStream_t stream
) {
    if (num_infosets <= 0) {
        return;
    }

    initialize_uniform_strategy_kernel<<<
    blocks_for(num_infosets, config.threads_per_block),
    config.threads_per_block,
    0,
    stream>>>(
        d_infoset_q_begin,
        d_infoset_q_count,
        d_sigma,
        d_sigma_init,
        num_infosets
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("initialize_uniform_strategy_kernel launch failed: %s\n",cudaGetErrorString(err));
    }
}

void launch_compute_incoming_probabilities_level(
    const KernelLaunchConfig& config,
    const DeviceLevelEdges& edges,
    const int* d_player,
    const float* d_chance_prob,
    const float* d_sigma,
    float* d_incoming_prob,
    cudaStream_t stream
) {
    if (edges.count <= 0) {
        return;
    }

    compute_incoming_probabilities_level_kernel<<<blocks_for(edges.count, config.threads_per_block),
                                                  config.threads_per_block,
                                                  0,
                                                  stream>>>(
        edges,
        d_player,
        d_chance_prob,
        d_sigma,
        d_incoming_prob
    );
}

void launch_initialize_terminal_utilities(
    const KernelLaunchConfig& config,
    const float* d_terminal_u_p0,
    float* d_u_p0,
    int num_nodes,
    cudaStream_t stream
) {
    if (num_nodes <= 0) {
        return;
    }

    initialize_terminal_utilities_kernel<<<blocks_for(num_nodes, config.threads_per_block),
                                           config.threads_per_block,
                                           0,
                                           stream>>>(
        d_terminal_u_p0,
        d_u_p0,
        num_nodes
    );
}

void launch_backward_utility_level(
    const KernelLaunchConfig& config,
    const DeviceLevelEdges& edges,
    const float* d_incoming_prob,
    const float* d_u_p0_read,
    float* d_u_p0_write,
    cudaStream_t stream
) {
    if (edges.count <= 0) {
        return;
    }

    backward_utility_level_kernel<<<blocks_for(edges.count, config.threads_per_block),
                                    config.threads_per_block,
                                    0,
                                    stream>>>(
        edges,
        d_incoming_prob,
        d_u_p0_read,
        d_u_p0_write
    );
}

void launch_initialize_reach(
    const KernelLaunchConfig& config,
    float* d_reach_p0,
    float* d_reach_p1,
    int num_nodes,
    int root,
    cudaStream_t stream
) {
    if (num_nodes <= 0) {
        return;
    }

    initialize_reach_kernel<<<blocks_for(num_nodes, config.threads_per_block),
                              config.threads_per_block,
                              0,
                              stream>>>(
        d_reach_p0,
        d_reach_p1,
        num_nodes,
        root
    );
}

void launch_forward_reach_level(
    const KernelLaunchConfig& config,
    const DeviceLevelEdges& edges,
    const int* d_player,
    const float* d_incoming_prob,
    const float* d_reach_p0_read,
    const float* d_reach_p1_read,
    float* d_reach_p0_write,
    float* d_reach_p1_write,
    cudaStream_t stream
) {
    if (edges.count <= 0) {
        return;
    }

    forward_reach_level_kernel<<<blocks_for(edges.count, config.threads_per_block),
                                 config.threads_per_block,
                                 0,
                                 stream>>>(
        edges,
        d_player,
        d_incoming_prob,
        d_reach_p0_read,
        d_reach_p1_read,
        d_reach_p0_write,
        d_reach_p1_write
    );
}

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
    cudaStream_t stream
) {
    if (num_infosets <= 0) {
        return;
    }

    compute_infoset_reach_kernel<<<blocks_for(num_infosets, config.threads_per_block),
                                   config.threads_per_block,
                                   0,
                                   stream>>>(
        d_infoset_player,
        d_infoset_node_begin,
        d_infoset_node_count,
        d_infoset_nodes,
        d_reach_p0,
        d_reach_p1,
        d_infoset_reach,
        num_infosets
    );
}

void launch_compute_instantaneous_regrets(
    const KernelLaunchConfig& config,
    const DeviceDecisionEdges& decision_edges,
    const int* d_player,
    const float* d_u_p0,
    const float* d_reach_p0,
    const float* d_reach_p1,
    float* d_inst_regret,
    cudaStream_t stream
) {
    if (decision_edges.count <= 0) {
        return;
    }

    compute_instantaneous_regrets_kernel<<<blocks_for(decision_edges.count, config.threads_per_block),
                                           config.threads_per_block,
                                           0,
                                           stream>>>(
        decision_edges,
        d_player,
        d_u_p0,
        d_reach_p0,
        d_reach_p1,
        d_inst_regret
    );
}

void launch_update_regrets(
    const KernelLaunchConfig& config,
    const float* d_inst_regret,
    float* d_regret_sum,
    int num_q,
    bool use_cfr_plus,
    cudaStream_t stream
) {
    if (num_q <= 0) {
        return;
    }

    update_regrets_kernel<<<blocks_for(num_q, config.threads_per_block),
                            config.threads_per_block,
                            0,
                            stream>>>(
        d_inst_regret,
        d_regret_sum,
        num_q,
        use_cfr_plus
    );
}

void launch_clear_positive_regret_sums(
    const KernelLaunchConfig& config,
    float* d_positive_regret_sum,
    int num_infosets,
    cudaStream_t stream
) {
    launch_fill_float(
        config,
        d_positive_regret_sum,
        0.0f,
        num_infosets,
        stream
    );
}

__global__ void forward_own_reach_level_kernel(
    const int* parent,
    const int* child,
    const int* q_index,
    int edge_count,
    const int* player,
    const float* sigma,
    const float* own_reach_p0_read,
    const float* own_reach_p1_read,
    float* own_reach_p0_write,
    float* own_reach_p1_write
) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;

    if (e >= edge_count) {
        return;
    }

    const int p = parent[e];
    const int c = child[e];
    const int q = q_index[e];
    const int parent_player = player[p];

    const float parent_own_p0 = own_reach_p0_read[p];
    const float parent_own_p1 = own_reach_p1_read[p];

    // Chance node.
    //
    // CPU parity rule:
    // chance does NOT multiply reach_p0/reach_p1.
    if (parent_player == -1) {
        own_reach_p0_write[c] = parent_own_p0;
        own_reach_p1_write[c] = parent_own_p1;
        return;
    }

    // P0 decision node.
    if (parent_player == 0) {
        own_reach_p0_write[c] = parent_own_p0 * sigma[q];
        own_reach_p1_write[c] = parent_own_p1;
        return;
    }

    // P1 decision node.
    if (parent_player == 1) {
        own_reach_p0_write[c] = parent_own_p0;
        own_reach_p1_write[c] = parent_own_p1 * sigma[q];
        return;
    }
}

void launch_forward_own_reach_level(
    const KernelLaunchConfig& config,
    const DeviceLevelEdges& edges,
    const int* d_player,
    const float* d_sigma,
    const float* d_own_reach_p0_read,
    const float* d_own_reach_p1_read,
    float* d_own_reach_p0_write,
    float* d_own_reach_p1_write,
    cudaStream_t stream
) {
    if (edges.count <= 0) {
        return;
    }

    forward_own_reach_level_kernel<<<
        blocks_for(edges.count, config.threads_per_block),
        config.threads_per_block,
        0,
        stream
    >>>(
        edges.d_parent,
        edges.d_child,
        edges.d_q_index,
        edges.count,
        d_player,
        d_sigma,
        d_own_reach_p0_read,
        d_own_reach_p1_read,
        d_own_reach_p0_write,
        d_own_reach_p1_write
    );
}

__global__ void compute_own_infoset_reach_kernel(
    const int* infoset_player,
    const int* infoset_node_begin,
    const int* infoset_node_count,
    const int* infoset_nodes,
    const float* own_reach_p0,
    const float* own_reach_p1,
    float* own_infoset_reach,
    int num_infosets
) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;

    if (h >= num_infosets) {
        return;
    }

    const int owner = infoset_player[h];
    const int begin = infoset_node_begin[h];
    const int count = infoset_node_count[h];

    float sum = 0.0f;

    for (int i = 0; i < count; ++i) {
        const int node = infoset_nodes[begin + i];

        if (owner == 0) {
            sum += own_reach_p0[node];
        } else if (owner == 1) {
            sum += own_reach_p1[node];
        }
    }

    own_infoset_reach[h] = sum;
}

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
        cudaStream_t stream
    ) {
    if (num_infosets <= 0) {
        return;
    }

    compute_own_infoset_reach_kernel<<<
        blocks_for(num_infosets, config.threads_per_block),
        config.threads_per_block,
        0,
        stream
    >>>(
        d_infoset_player,
        d_infoset_node_begin,
        d_infoset_node_count,
        d_infoset_nodes,
        d_own_reach_p0,
        d_own_reach_p1,
        d_own_infoset_reach,
        num_infosets
    );
}

void launch_accumulate_positive_regret_sums(
    const KernelLaunchConfig& config,
    const float* d_regret_sum,
    const int* d_q_infoset,
    float* d_positive_regret_sum,
    int num_q,
    cudaStream_t stream
) {
    if (num_q <= 0) {
        return;
    }

    accumulate_positive_regret_sums_kernel<<<blocks_for(num_q, config.threads_per_block),
                                             config.threads_per_block,
                                             0,
                                             stream>>>(
        d_regret_sum,
        d_q_infoset,
        d_positive_regret_sum,
        num_q
    );
}

void launch_regret_matching(
    const KernelLaunchConfig& config,
    const float* d_regret_sum,
    const float* d_positive_regret_sum,
    const float* d_sigma_init,
    const int* d_q_infoset,
    float* d_sigma,
    int num_q,
    cudaStream_t stream
) {
    if (num_q <= 0) {
        return;
    }

    regret_matching_kernel<<<blocks_for(num_q, config.threads_per_block),
                             config.threads_per_block,
                             0,
                             stream>>>(
        d_regret_sum,
        d_positive_regret_sum,
        d_sigma_init,
        d_q_infoset,
        d_sigma,
        num_q
    );
}

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
        cudaStream_t stream
    ) {
    if (num_q <= 0) {
        return;
    }

    accumulate_average_strategy_kernel<<<
        blocks_for(num_q, config.threads_per_block),
        config.threads_per_block,
        0,
        stream
    >>>(
        d_q_infoset,
        d_infoset_q_begin,
        d_sigma,
        d_own_infoset_reach,
        d_strategy_sum,
        d_strategy_weight_sum,
        num_q,
        iteration_weight
    );
}

void launch_normalize_average_strategy(
    const KernelLaunchConfig& config,
    const int* d_q_infoset,
    const float* d_strategy_sum,
    const float* d_strategy_weight_sum,
    const float* d_sigma_init,
    float* d_avg_strategy,
    int num_q,
    cudaStream_t stream
) {
    if (num_q <= 0) {
        return;
    }

    normalize_average_strategy_kernel<<<blocks_for(num_q, config.threads_per_block),
                                        config.threads_per_block,
                                        0,
                                        stream>>>(
        d_q_infoset,
        d_strategy_sum,
        d_strategy_weight_sum,
        d_sigma_init,
        d_avg_strategy,
        num_q
    );
}
} // namespace poker