// cfr_gpu.cpp
//
// Host-side CUDA implementation of CFR for the flattened explicit game tree.
// This file owns GPU memory, launches kernels, and exposes the same public API
// as CpuCfrSolver.
//
// Major invariants:
//   - flat_ is immutable after construction.
//   - gpu_.game arrays are immutable after upload.
//   - gpu_.cfr arrays persist across CFR iterations.
//   - gpu_.work arrays are temporary per-iteration scratch.
//   - average strategy is accumulated before sigma is updated.
//   - regret reach and average-strategy reach must not be confused.

#include "cfr_gpu.hpp"
#include "kernels.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace poker {
namespace {

// ------------------------------
// CUDA utility helpers
// ------------------------------

// Checks if a given CUDA operation results in an error, and if so throws that error with a given [message]
void check_cuda(const cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(result));
    }
}

template <typename T>
void cuda_alloc_copy(T** dst, const std::vector<T>& src) {
    if (dst == nullptr) {
        throw std::invalid_argument("cuda_alloc_copy received null dst.");
    }
    if (src.empty()) {
        return;
    }
    *dst = nullptr;

    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(dst),sizeof(T) * src.size()),
    "cudaMalloc failed in cuda_alloc_copy"
    );

    try {
        check_cuda(
            cudaMemcpy(*dst,src.data(),sizeof(T) * src.size(),cudaMemcpyHostToDevice),
            "cudaMemcpy failed in cuda_alloc_copy"
        );
    } catch (...) {
        cudaFree(*dst);
        *dst = nullptr;
        throw;
    }
}

    template <typename T>
void cuda_alloc_zero(T** dst, const int count) {
    if (dst == nullptr) {
        throw std::invalid_argument("cuda_alloc_zero received null dst.");
    }
    if (count <= 0) {
        return;
    }
    *dst = nullptr;

    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(dst),sizeof(T) * static_cast<std::size_t>(count)),
        "cudaMalloc failed in cuda_alloc_zero"
    );

    try {
        check_cuda(
            cudaMemset(*dst,0,sizeof(T) * static_cast<std::size_t>(count)),
            "cudaMemset failed in cuda_alloc_zero"
        );
    } catch (...) {
        cudaFree(*dst);
        *dst = nullptr;
        throw;
    }
}

template <typename T>
void cuda_free_ptr(T*& ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

int player_to_int(Player player) {
    return static_cast<int>(player);
}

bool is_real_player_int(int player) {
    return player == player_to_int(Player::P0) || player == player_to_int(Player::P1);
}

} // namespace

// ------------------------------
// Flattening
// ------------------------------

FlatGame flatten_game_for_gpu(const Game& game) {
    FlatGame flat;

    // Copy metadata from game into flat
    flat.num_nodes = game.num_nodes();
    flat.num_infosets = game.num_infosets();
    flat.num_q = game.num_q();

    flat.num_players = game.num_players;
    flat.max_depth = game.max_depth;
    flat.root = game.root;

    if (flat.num_nodes <= 0) {
        throw std::runtime_error("Cannot flatten an empty game.");
    }

    if (flat.root < 0 || flat.root >= flat.num_nodes) {
        throw std::runtime_error("Game root is out of range.");
    }

    if (flat.num_infosets < 0 || flat.num_q < 0) {
        throw std::runtime_error("Game has invalid infoset or q counts.");
    }

    if (flat.max_depth < 0) {
        throw std::runtime_error("Game has invalid max_depth.");
    }

    // Allocate each per-node data into an array, indexed by the node ID
    flat.parent.assign(flat.num_nodes, -1); // Sets flat.parent as a vector of size flat.num_nodes filled with "-1"
    flat.depth.assign(flat.num_nodes, 0);
    flat.player.assign(flat.num_nodes, static_cast<int>(Player::Terminal));
    flat.infoset.assign(flat.num_nodes, -1);
    flat.terminal_u_p0.assign(flat.num_nodes, 0.0f);
    flat.chance_prob.assign(flat.num_nodes, 0.0f);

    // level_edges[d] stores edges whose parent is at depth d.
    //
    // Since a child has depth parent.depth + 1, useful edge depths are:
    //   0, 1, ..., max_depth - 1
    //
    // Keeping max_depth + 1 entries is harmless and matches the existing
    // project shape.
    flat.level_edges.clear();
    flat.level_edges.resize(static_cast<std::size_t>(flat.max_depth + 1));

    // Temporary structure used to build the CSR-style infoset-node mapping.
    //
    // infoset_nodes[h] contains all decision-node ids that belong to infoset h.
    std::vector<std::vector<int>> infoset_nodes(static_cast<std::size_t>(flat.num_infosets));

    // Flatten the nodes and edges
    for (const Node& node : game.nodes) {
        if (node.id < 0 || node.id >= flat.num_nodes) {
            throw std::runtime_error("Node id is out of range.");
        }
        if (node.depth < 0 || node.depth > flat.max_depth) {
            throw std::runtime_error("Node depth is out of range.");
        }
        if (node.id == flat.root && node.parent != -1) {
            throw std::runtime_error("Root node should have parent -1.");
        }
        if (node.id != flat.root) {
            if (node.parent < 0 || node.parent >= flat.num_nodes) {
                throw std::runtime_error("Non-root node has invalid parent.");
            }
        }
        if (node.terminal && !node.children.empty()) {
            throw std::runtime_error("Terminal node should not have children.");
        }
        const int node_id = node.id;
        // Record which tree nodes belong to which infoset.
        // Multiple tree nodes can be within the same infoset.
        // The GPU kernels are unable to determine the correct infoset from just the node
        if (is_real_player(node.player)) {
            if (node.infoset < 0 || node.infoset >= flat.num_infosets) {
                throw std::runtime_error(
                    "Real-player node has invalid infoset."
                );
            }
            // Push the node_id into the corresponding infoset array
            infoset_nodes[static_cast<std::size_t>(node.infoset)]
                .push_back(node_id);
        } else {
            if (node.infoset != -1) {
                throw std::runtime_error(
                    "Chance or terminal node should not have an infoset."
                );
            }
        }
        flat.parent[node_id] = node.parent;
        flat.depth[node_id] = node.depth;
        flat.player[node_id] = static_cast<int>(node.player);
        flat.infoset[node_id] = node.infoset;
        flat.terminal_u_p0[node_id] = node.terminal ? node.utility_p0 : 0.0f;
        flat.chance_prob[node_id] = node.chance_prob;

        // Flatten outgoing edges.
        for (int local_action = 0; local_action < static_cast<int>(node.children.size()); ++local_action) {
            const int child_id = node.children[local_action];
            const Node& child = game.node(child_id);

            if (child_id < 0 || child_id >= flat.num_nodes) {
                throw std::runtime_error("Child id is out of range.");
            }
            if (child.parent != node.id) {
                throw std::runtime_error(
                    "Parent/child pointer mismatch while flattening."
                );
            }
            if (child.depth != node.depth + 1) {
                throw std::runtime_error(
                    "Child depth should be parent depth + 1."
                );
            }
            if (node.depth >= flat.max_depth) {
                throw std::runtime_error(
                    "Node at max_depth should not have outgoing edges."
                );
            }

            int q_index = -1;
            // If this edge leaves a real-player decision node, it corresponds
            // to one infoset-action entry q.
            if (is_real_player(node.player)) {
                if (node.infoset < 0 || node.infoset >= flat.num_infosets) {
                    throw std::runtime_error(
                        "Decision node has invalid infoset."
                    );
                }
                const InfoSet& infoset = game.infoset(node.infoset);
                if (local_action >= static_cast<int>(infoset.q_indices.size())) {
                    throw std::runtime_error(
                        "Decision-node children and infoset q_indices are misaligned."
                    );
                }

                q_index = infoset.q_indices[local_action];

                if (q_index < 0 || q_index >= flat.num_q) {
                    throw std::runtime_error(
                        "Decision edge has q index out of range."
                    );
                }

                // Decision-edge list contains only player-action edges.
                // It is used by the regret kernel.
                flat.decision_edge_parent.push_back(node.id);
                flat.decision_edge_child.push_back(child.id);
                flat.decision_edge_q_index.push_back(q_index);
            }

            // Level-edge list contains all edges: chance and player edges.
            // It is used by tree dynamic programming kernels.
            auto&[parents, children, q_indexes] = flat.level_edges[static_cast<std::size_t>(node.depth)];

            parents.push_back(node.id);
            children.push_back(child.id);
            q_indexes.push_back(q_index);
        }
    }

    // Flatten q-entry metadata
    // q = one infoset-action pair

    flat.q_infoset.assign(flat.num_q, -1);
    flat.q_local_action.assign(flat.num_q, -1);

    for (const InfoSetAction& q_entry : game.q_entries) {
        if (q_entry.q < 0 || q_entry.q >= flat.num_q) {
            throw std::runtime_error("q_entry.q is out of range.");
        }
        if (q_entry.infoset < 0 || q_entry.infoset >= flat.num_infosets) {
            throw std::runtime_error("q_entry.infoset is out of range.");
        }
        if (q_entry.local_action < 0) {
            throw std::runtime_error("q_entry.local_action is invalid.");
        }

        flat.q_infoset[q_entry.q] = q_entry.infoset;
        flat.q_local_action[q_entry.q] = q_entry.local_action;
    }
    // Double checks that all the q entries were initialized, in case game.q_entries.size() < flat.num_q
    for (int q = 0; q < flat.num_q; ++q) {
        if (flat.q_infoset[q] < 0) {
            throw std::runtime_error("Some q entries were not initialized.");
        }

        if (flat.q_local_action[q] < 0) {
            throw std::runtime_error(
                "Some q local-action entries were not initialized."
            );
        }
    }

    // Flatten infoset metadata.
    flat.infoset_player.assign(flat.num_infosets, static_cast<int>(Player::Terminal));
    flat.infoset_q_begin.assign(flat.num_infosets, 0);
    flat.infoset_q_count.assign(flat.num_infosets, 0);

    flat.infoset_node_begin.assign(flat.num_infosets, 0);
    flat.infoset_node_count.assign(flat.num_infosets, 0);

    flat.infoset_nodes.clear();

    for (const InfoSet& infoset : game.infosets) {
        if (infoset.id < 0 || infoset.id >= flat.num_infosets) {
            throw std::runtime_error("Infoset id is out of range.");
        }
        if (!is_real_player(infoset.player)) {
            throw std::runtime_error("Infoset owner must be P0 or P1.");
        }
        if (infoset.actions.empty()) {
            throw std::runtime_error("Infoset has no actions.");
        }
        if (infoset.q_indices.size() != infoset.actions.size()) {
            throw std::runtime_error(
                "Infoset actions and q_indices have different sizes."
            );
        }

        const int infoset_id = infoset.id;

        flat.infoset_player[infoset_id] = static_cast<int>(infoset.player);

        // Assumes q entries belonging to an infoset are contiguous.
        // That assumption is used by kernels that iterate:
        //
        //   q_begin[h] ... q_begin[h] + q_count[h]
        //
        const int q_begin = infoset.q_indices.front();
        const int q_count = static_cast<int>(infoset.q_indices.size());

        if (q_begin < 0 || q_begin >= flat.num_q) {
            throw std::runtime_error("Infoset q_begin is out of range.");
        }

        for (int local = 0; local < q_count; ++local) {
            const int q = infoset.q_indices[local];

            if (q < 0 || q >= flat.num_q) {
                throw std::runtime_error("Infoset q index is out of range.");
            }
            if (q != q_begin + local) {
                throw std::runtime_error(
                    "Infoset q indices must be contiguous for GPU flattening."
                );
            }
            if (flat.q_infoset[q] != infoset_id) {
                throw std::runtime_error(
                    "q_infoset does not match owning infoset."
                );
            }
            if (flat.q_local_action[q] != local) {
                throw std::runtime_error(
                    "q_local_action does not match infoset action order."
                );
            }
        }

        flat.infoset_q_begin[infoset_id] = q_begin;
        flat.infoset_q_count[infoset_id] = q_count;

        // Build CSR-style infoset -> nodes mapping.
        flat.infoset_node_begin[infoset_id] = static_cast<int>(flat.infoset_nodes.size());

        for (int node_id : infoset_nodes[static_cast<std::size_t>(infoset_id)]) {
            if (node_id < 0 || node_id >= flat.num_nodes) {
                throw std::runtime_error(
                    "Infoset-node mapping contains invalid node id."
                );
            }
            if (flat.infoset[node_id] != infoset_id) {
                throw std::runtime_error(
                    "Infoset-node mapping does not match node infoset."
                );
            }
            if (!is_real_player_int(flat.player[node_id])) {
                throw std::runtime_error(
                    "Infoset-node mapping contains non-player node."
                );
            }

            flat.infoset_nodes.push_back(node_id);
        }

        flat.infoset_node_count[infoset_id] = static_cast<int>(infoset_nodes[static_cast<std::size_t>(infoset_id)].size());

        if (flat.infoset_node_count[infoset_id] == 0) {
            throw std::runtime_error(
                "Infoset has no corresponding decision nodes."
            );
        }
    }

    // Validate the output structure.

    if (flat.decision_edge_parent.size() != flat.decision_edge_child.size() ||
        flat.decision_edge_parent.size() != flat.decision_edge_q_index.size()) {
        throw std::runtime_error("Decision-edge arrays have mismatched sizes.");
    }
    for (const LevelEdges& edges : flat.level_edges) {
        if (edges.parent.size() != edges.child.size() ||
            edges.parent.size() != edges.q_index.size()) {
            throw std::runtime_error("Level-edge arrays have mismatched sizes.");
        }
    }
    if (!flat.valid_basic_shape()) {
        throw std::runtime_error("FlatGame has invalid basic shape.");
    }

    return flat;
}

// ------------------------------
// Construction / ownership
// ------------------------------

GpuCfrSolver::GpuCfrSolver(const Game& game)
    : GpuCfrSolver(flatten_game_for_gpu(game), GpuCfrConfig{}) {}

GpuCfrSolver::GpuCfrSolver(const Game& game, GpuCfrConfig config)
    : GpuCfrSolver(flatten_game_for_gpu(game), config) {}

GpuCfrSolver::GpuCfrSolver(const FlatGame& flat_game)
    : GpuCfrSolver(flat_game, GpuCfrConfig{}) {}

GpuCfrSolver::GpuCfrSolver(const FlatGame& flat_game, GpuCfrConfig config)
        : config_(config),
          stats_{},
          flat_(flat_game),
          gpu_{} {
    if (!flat_.valid_basic_shape()) {
        throw std::invalid_argument("FlatGame has invalid basic shape.");
    }
    if (flat_.num_nodes <= 0) {
        throw std::invalid_argument("FlatGame must contain at least one node.");
    }
    if (flat_.root < 0 || flat_.root >= flat_.num_nodes) {
        throw std::invalid_argument("FlatGame root is out of range.");
    }
    if (flat_.num_infosets <= 0) {
        throw std::invalid_argument("FlatGame must contain at least one infoset.");
    }
    if (flat_.num_q <= 0) {
        throw std::invalid_argument("FlatGame must contain at least one q entry.");
    }
    if (flat_.max_depth < 0) {
        throw std::invalid_argument("FlatGame max_depth is invalid.");
    }
    if (config_.threads_per_block <= 0) {
        throw std::invalid_argument("GpuCfrConfig threads_per_block must be positive.");
    }
    try {
        upload_static_game();
        allocate_dynamic_buffers();
        initialize_solver_state();
    } catch (...) {
        free_device_memory();
        throw;
    }
}

GpuCfrSolver::~GpuCfrSolver() {
    free_device_memory();
}
GpuCfrSolver::GpuCfrSolver(GpuCfrSolver&& other) noexcept
        : config_(other.config_),
          stats_(other.stats_),
          flat_(std::move(other.flat_)),
          gpu_(std::move(other.gpu_)) {
    // Prevent the moved-from object's destructor from freeing memory now owned
    // by this object.
    other.gpu_ = GpuState{};
    other.stats_ = GpuCfrStats{};
}
GpuCfrSolver& GpuCfrSolver::operator=(GpuCfrSolver&& other) noexcept {
    if (this != &other) {
        // Release any CUDA memory currently owned by this object.
        free_device_memory();

        config_ = other.config_;
        stats_ = other.stats_;
        flat_ = std::move(other.flat_);
        gpu_ = std::move(other.gpu_);

        // Prevent double-free when the moved-from object is destroyed.
        other.gpu_ = GpuState{};
        other.stats_ = GpuCfrStats{};
    }

    return *this;
}

// ------------------------------
// Public run API
// ------------------------------

void GpuCfrSolver::run_iterations(const int iterations) {
    if (iterations < 0) {
        throw std::runtime_error("Cannot run negative iterations");
    }
    for (int i = 0; i < iterations; i++) {
        run_one_iteration();
    }
}

void GpuCfrSolver::run_one_iteration() {

    compute_incoming_action_probabilities();
    compute_expected_utilities();

    compute_counterfactual_reach();
    compute_infoset_reach();

    compute_own_reach();
    compute_own_infoset_reach();

    compute_instantaneous_regrets();

    accumulate_average_strategy();
    normalize_average_strategy();
    // -------------------------------------------------------------------------
    // Stage G: update cumulative regret and compute next iteration's strategy.
    // -------------------------------------------------------------------------
    update_regrets();
    update_current_strategy();
    // -------------------------------------------------------------------------
    // Stage H: complete iteration bookkeeping.
    // -------------------------------------------------------------------------
    ++stats_.iterations_run;
    // -------------------------------------------------------------------------
    // Error handling / synchronization.
    //
    // During debugging, synchronize every iteration to catch the exact failing
    // stage. For performance, only check launch errors here.
    // -------------------------------------------------------------------------
    if (config_.synchronize_each_iteration) {
        check_cuda(
            cudaDeviceSynchronize(),
            "cudaDeviceSynchronize failed after GPU CFR iteration"
        );
    } else {
        check_cuda(
            cudaGetLastError(),
            "CUDA kernel launch failed during GPU CFR iteration"
        );
    }
}

// ------------------------------
// Strategy accessors
// ------------------------------

std::vector<float> GpuCfrSolver::current_strategy() const {
    std::vector<float> result(static_cast<std::size_t>(flat_.num_q), 0.0f);
    if (flat_.num_q == 0) {
        return result;
    }
    if (gpu_.cfr.d_sigma == nullptr) {
        throw std::runtime_error("GPU current strategy buffer is not allocated.");
    }
    //   copy d_sigma to host vector
    check_cuda(
        cudaMemcpy(result.data(), gpu_.cfr.d_sigma,sizeof(float) * result.size(),cudaMemcpyDeviceToHost),
        "Failed to copy current strategy from device to host"
    );
    return result;
}

std::vector<float> GpuCfrSolver::average_strategy() const {
    std::vector<float> result(static_cast<std::size_t>(flat_.num_q), 0.0f);

    if (flat_.num_q == 0) {
        return result;
    }
    if (gpu_.cfr.d_avg_strategy == nullptr) {
        throw std::runtime_error("GPU average strategy buffer is not allocated.");
    }
    // copy d_avg_strategy to host vector
    check_cuda(
        cudaMemcpy(result.data(),gpu_.cfr.d_avg_strategy,sizeof(float) * result.size(),cudaMemcpyDeviceToHost),
        "Failed to copy average strategy from device to host"
    );

    return result;
}

void GpuCfrSolver::set_current_strategy_for_testing(
        const std::vector<float>& strategy
    ) {
    if (static_cast<int>(strategy.size()) != flat_.num_q) {
        throw std::invalid_argument(
            "Testing strategy size does not match flat_.num_q."
        );
    }

    if (gpu_.cfr.d_sigma == nullptr) {
        throw std::runtime_error("GPU sigma buffer is not allocated.");
    }

    check_cuda(
        cudaMemcpy(
            gpu_.cfr.d_sigma,
            strategy.data(),
            sizeof(float) * strategy.size(),
            cudaMemcpyHostToDevice
        ),
        "Failed to copy testing strategy to device sigma"
    );
}

std::vector<float> GpuCfrSolver::debug_strategy_sum() const {
    std::vector<float> result(static_cast<std::size_t>(flat_.num_q), 0.0f);

    if (gpu_.cfr.d_strategy_sum == nullptr) {
        throw std::runtime_error("GPU strategy_sum buffer is not allocated.");
    }

    check_cuda(
        cudaMemcpy(
            result.data(),
            gpu_.cfr.d_strategy_sum,
            sizeof(float) * result.size(),
            cudaMemcpyDeviceToHost
        ),
        "Failed to copy strategy_sum from device"
    );

    return result;
}

std::vector<float> GpuCfrSolver::debug_strategy_weight_sum() const {
    std::vector<float> result(
        static_cast<std::size_t>(flat_.num_infosets),
        0.0f
    );

    if (gpu_.cfr.d_strategy_weight_sum == nullptr) {
        throw std::runtime_error(
            "GPU strategy_weight_sum buffer is not allocated."
        );
    }

    check_cuda(
        cudaMemcpy(
            result.data(),
            gpu_.cfr.d_strategy_weight_sum,
            sizeof(float) * result.size(),
            cudaMemcpyDeviceToHost
        ),
        "Failed to copy strategy_weight_sum from device"
    );

    return result;
}

// ------------------------------
// Memory upload/allocation
// ------------------------------

void GpuCfrSolver::upload_static_game() {
    // -------------------------------------------------------------------------
    // Copy scalar metadata.
    // -------------------------------------------------------------------------

    gpu_.game.num_nodes = flat_.num_nodes;
    gpu_.game.num_infosets = flat_.num_infosets;
    gpu_.game.num_q = flat_.num_q;
    gpu_.game.num_players = flat_.num_players;
    gpu_.game.max_depth = flat_.max_depth;
    gpu_.game.root = flat_.root;

    // -------------------------------------------------------------------------
    // Upload static per-node arrays.
    //
    // Length: num_nodes
    // -------------------------------------------------------------------------

    cuda_alloc_copy(&gpu_.game.d_parent, flat_.parent);
    cuda_alloc_copy(&gpu_.game.d_depth, flat_.depth);
    cuda_alloc_copy(&gpu_.game.d_player, flat_.player);
    cuda_alloc_copy(&gpu_.game.d_infoset, flat_.infoset);
    cuda_alloc_copy(&gpu_.game.d_terminal_u_p0, flat_.terminal_u_p0);
    cuda_alloc_copy(&gpu_.game.d_chance_prob, flat_.chance_prob);

    // -------------------------------------------------------------------------
    // Upload static per-q arrays.
    //
    // Length: num_q
    // -------------------------------------------------------------------------

    cuda_alloc_copy(&gpu_.game.d_q_infoset, flat_.q_infoset);
    cuda_alloc_copy(&gpu_.game.d_q_local_action, flat_.q_local_action);

    // -------------------------------------------------------------------------
    // Upload static per-infoset metadata.
    //
    // Length: num_infosets
    // -------------------------------------------------------------------------

    cuda_alloc_copy(&gpu_.game.d_infoset_player, flat_.infoset_player);
    cuda_alloc_copy(&gpu_.game.d_infoset_q_begin, flat_.infoset_q_begin);
    cuda_alloc_copy(&gpu_.game.d_infoset_q_count, flat_.infoset_q_count);

    // -------------------------------------------------------------------------
    // Upload infoset -> nodes CSR mapping.
    //
    // These arrays are needed because a single infoset can contain multiple
    // game-tree nodes.
    // -------------------------------------------------------------------------

    cuda_alloc_copy(
        &gpu_.game.d_infoset_node_begin,
        flat_.infoset_node_begin
    );

    cuda_alloc_copy(
        &gpu_.game.d_infoset_node_count,
        flat_.infoset_node_count
    );

    cuda_alloc_copy(
        &gpu_.game.d_infoset_nodes,
        flat_.infoset_nodes
    );

    gpu_.game.num_infoset_nodes =
        static_cast<int>(flat_.infoset_nodes.size());

    // -------------------------------------------------------------------------
    // Upload level edge lists.
    //
    // level_edges[d] contains all edges whose parent is at depth d.
    // These are used by the dynamic-programming passes:
    //   - incoming edge probability
    //   - backward utility pass
    //   - forward reach pass
    // -------------------------------------------------------------------------

    gpu_.game.level_edges.resize(flat_.level_edges.size());

    for (std::size_t depth = 0; depth < flat_.level_edges.size(); ++depth) {
        const LevelEdges& host_edges = flat_.level_edges[depth];
        DeviceLevelEdges& device_edges = gpu_.game.level_edges[depth];

        device_edges.count = host_edges.size();

        cuda_alloc_copy(&device_edges.d_parent, host_edges.parent);
        cuda_alloc_copy(&device_edges.d_child, host_edges.child);
        cuda_alloc_copy(&device_edges.d_q_index, host_edges.q_index);
    }

    // -------------------------------------------------------------------------
    // Upload decision-edge list.
    //
    // decision_edges contains only real-player action edges, meaning q_index >= 0.
    // These are used by the regret kernel.
    // -------------------------------------------------------------------------

    gpu_.game.decision_edges.count =
        static_cast<int>(flat_.decision_edge_q_index.size());

    cuda_alloc_copy(
        &gpu_.game.decision_edges.d_parent,
        flat_.decision_edge_parent
    );

    cuda_alloc_copy(
        &gpu_.game.decision_edges.d_child,
        flat_.decision_edge_child
    );

    cuda_alloc_copy(
        &gpu_.game.decision_edges.d_q_index,
        flat_.decision_edge_q_index
    );

    gpu_.allocated = true;
}

    void GpuCfrSolver::allocate_dynamic_buffers() {
    // -------------------------------------------------------------------------
    // Persistent CFR state.
    //
    // These arrays live across all CFR iterations.
    // -------------------------------------------------------------------------

    cuda_alloc_zero(&gpu_.cfr.d_sigma, flat_.num_q);
    cuda_alloc_zero(&gpu_.cfr.d_sigma_init, flat_.num_q);
    cuda_alloc_zero(&gpu_.cfr.d_regret_sum, flat_.num_q);
    cuda_alloc_zero(&gpu_.cfr.d_strategy_sum, flat_.num_q);
    cuda_alloc_zero(&gpu_.cfr.d_avg_strategy, flat_.num_q);

    // One denominator per infoset for normalized average strategy.
    cuda_alloc_zero(
        &gpu_.cfr.d_strategy_weight_sum,
        flat_.num_infosets
    );

    // -------------------------------------------------------------------------
    // Per-iteration work buffers.
    //
    // These arrays are overwritten during each CFR iteration.
    // -------------------------------------------------------------------------

    cuda_alloc_zero(&gpu_.work.d_incoming_prob, flat_.num_nodes);
    cuda_alloc_zero(&gpu_.work.d_u_p0, flat_.num_nodes);

    // Current implementation uses two reach arrays.
    //
    // Be very clear in kernels what these mean. I recommend treating them as:
    //   d_reach_p0 = reach excluding P0's own actions
    //   d_reach_p1 = reach excluding P1's own actions
    cuda_alloc_zero(&gpu_.work.d_reach_p0, flat_.num_nodes);
    cuda_alloc_zero(&gpu_.work.d_reach_p1, flat_.num_nodes);
    cuda_alloc_zero(&gpu_.work.d_counterfactual_infoset_reach, flat_.num_infosets);

    cuda_alloc_zero(&gpu_.work.d_own_reach_p0, flat_.num_nodes);
    cuda_alloc_zero(&gpu_.work.d_own_reach_p1, flat_.num_nodes);
    cuda_alloc_zero(&gpu_.work.d_own_infoset_reach, flat_.num_infosets);

    // Aggregated reach per infoset.
    cuda_alloc_zero(&gpu_.work.d_counterfactual_infoset_reach, flat_.num_infosets);

    // One instantaneous regret per q.
    cuda_alloc_zero(&gpu_.work.d_inst_regret, flat_.num_q);

    // One positive-regret sum per infoset.
    cuda_alloc_zero(
        &gpu_.work.d_positive_regret_sum,
        flat_.num_infosets
    );

    // Optional scratch field from gpu_state.hpp.
    gpu_.work.d_scratch = nullptr;
    gpu_.work.scratch_count = 0;
}

    void GpuCfrSolver::initialize_solver_state() {
    KernelLaunchConfig launch{config_.threads_per_block};

    // -------------------------------------------------------------------------
    // Initialize current strategy and fallback strategy.
    //
    // For each infoset h:
    //   sigma[q]      = 1 / number_of_actions(h)
    //   sigma_init[q] = 1 / number_of_actions(h)
    //
    // sigma_init is used when regret matching has no positive regrets.
    // -------------------------------------------------------------------------

    launch_initialize_uniform_strategy(
        launch,
        gpu_.game.d_infoset_q_begin,
        gpu_.game.d_infoset_q_count,
        gpu_.cfr.d_sigma,
        gpu_.cfr.d_sigma_init,
        flat_.num_infosets
    );

    // -------------------------------------------------------------------------
    // Initial average strategy should also be uniform.
    //
    // strategy_sum and strategy_weight_sum remain zero until iterations run.
    // d_avg_strategy is just the public normalized average-strategy view.
    // -------------------------------------------------------------------------

    launch_copy_float(
        launch,
        gpu_.cfr.d_sigma,
        gpu_.cfr.d_avg_strategy,
        flat_.num_q
    );

    check_cuda(
        cudaDeviceSynchronize(),
        "GPU solver initialization failed"
    );
}

void GpuCfrSolver::free_device_memory() {
    // -------------------------------------------------------------------------
    // Static per-node arrays.
    // -------------------------------------------------------------------------

    cuda_free_ptr(gpu_.game.d_parent);
    cuda_free_ptr(gpu_.game.d_depth);
    cuda_free_ptr(gpu_.game.d_player);
    cuda_free_ptr(gpu_.game.d_infoset);
    cuda_free_ptr(gpu_.game.d_terminal_u_p0);
    cuda_free_ptr(gpu_.game.d_chance_prob);

    // -------------------------------------------------------------------------
    // Static per-q arrays.
    // -------------------------------------------------------------------------

    cuda_free_ptr(gpu_.game.d_q_infoset);
    cuda_free_ptr(gpu_.game.d_q_local_action);

    // -------------------------------------------------------------------------
    // Static per-infoset arrays.
    // -------------------------------------------------------------------------

    cuda_free_ptr(gpu_.game.d_infoset_player);
    cuda_free_ptr(gpu_.game.d_infoset_q_begin);
    cuda_free_ptr(gpu_.game.d_infoset_q_count);
    cuda_free_ptr(gpu_.game.d_infoset_node_begin);
    cuda_free_ptr(gpu_.game.d_infoset_node_count);
    cuda_free_ptr(gpu_.game.d_infoset_nodes);

    gpu_.game.num_infoset_nodes = 0;

    // -------------------------------------------------------------------------
    // Static level-edge arrays.
    // -------------------------------------------------------------------------

    for (DeviceLevelEdges& edges : gpu_.game.level_edges) {
        cuda_free_ptr(edges.d_parent);
        cuda_free_ptr(edges.d_child);
        cuda_free_ptr(edges.d_q_index);
        edges.count = 0;
    }

    gpu_.game.level_edges.clear();

    // -------------------------------------------------------------------------
    // Static decision-edge arrays.
    // -------------------------------------------------------------------------

    cuda_free_ptr(gpu_.game.decision_edges.d_parent);
    cuda_free_ptr(gpu_.game.decision_edges.d_child);
    cuda_free_ptr(gpu_.game.decision_edges.d_q_index);
    gpu_.game.decision_edges.count = 0;

    // -------------------------------------------------------------------------
    // Persistent CFR arrays.
    // -------------------------------------------------------------------------

    cuda_free_ptr(gpu_.cfr.d_sigma);
    cuda_free_ptr(gpu_.cfr.d_sigma_init);
    cuda_free_ptr(gpu_.cfr.d_regret_sum);
    cuda_free_ptr(gpu_.cfr.d_strategy_sum);
    cuda_free_ptr(gpu_.cfr.d_avg_strategy);
    cuda_free_ptr(gpu_.cfr.d_strategy_weight_sum);

    // -------------------------------------------------------------------------
    // Per-iteration work buffers.
    // -------------------------------------------------------------------------

    cuda_free_ptr(gpu_.work.d_incoming_prob);
    cuda_free_ptr(gpu_.work.d_u_p0);
    cuda_free_ptr(gpu_.work.d_reach_p0);
    cuda_free_ptr(gpu_.work.d_reach_p1);
    cuda_free_ptr(gpu_.work.d_counterfactual_infoset_reach);

    cuda_free_ptr(gpu_.work.d_own_reach_p0);
    cuda_free_ptr(gpu_.work.d_own_reach_p1);
    cuda_free_ptr(gpu_.work.d_own_infoset_reach);

    cuda_free_ptr(gpu_.work.d_inst_regret);
    cuda_free_ptr(gpu_.work.d_positive_regret_sum);
    cuda_free_ptr(gpu_.work.d_scratch);

    gpu_.work.scratch_count = 0;

    // -------------------------------------------------------------------------
    // Reset scalar metadata last.
    // -------------------------------------------------------------------------

    gpu_ = GpuState{};
}

// ------------------------------
// CFR pipeline stages
// ------------------------------
// -------------------------------------------------------------------------
// Stage A: compute edge probabilities under the current strategy.
//
// d_incoming_prob[child] becomes either:
//   - chance probability, if parent is Chance
//   - sigma[q], if parent is P0/P1
// -------------------------------------------------------------------------
void GpuCfrSolver::compute_incoming_action_probabilities() {
    KernelLaunchConfig launch{config_.threads_per_block};

    // Clear the previous iteration's incoming-edge probabilities.
    //
    // d_incoming_prob[v] means:
    //   probability of taking the action/chance outcome from parent(v) to v
    //
    // Root has no incoming edge, so its value is irrelevant.
    launch_fill_float(
        launch,
        gpu_.work.d_incoming_prob,
        0.0f,
        flat_.num_nodes
    );

    // Each level_edges[depth] contains every edge whose parent is at that depth.
    //
    // For each edge parent -> child:
    //   if parent is Chance:
    //       incoming_prob[child] = chance_prob[child]
    //
    //   if parent is P0 or P1:
    //       incoming_prob[child] = sigma[q_index]
    //
    //   q_index is -1 only for chance edges.
    for (std::size_t depth = 0;
         depth < gpu_.game.level_edges.size();
         ++depth) {
        const DeviceLevelEdges& edges = gpu_.game.level_edges[depth];

        if (edges.count == 0) {
            continue;
        }

        launch_compute_incoming_probabilities_level(
            launch,
            edges,
            gpu_.game.d_player,
            gpu_.game.d_chance_prob,
            gpu_.cfr.d_sigma,
            gpu_.work.d_incoming_prob
        );
         }

    // Catch invalid launch configuration or immediate kernel-launch errors.
    //
    // Do not call cudaDeviceSynchronize here unless you are debugging, because
    // synchronizing every stage will make timing much worse.
    check_cuda(
        cudaGetLastError(),
        "Failed to launch compute_incoming_action_probabilities kernels"
    );
}
// -------------------------------------------------------------------------
// Stage B: compute expected utility for P0 at every node.
//
// This is the backward tree pass:
//   u_p0[parent] = sum_child incoming_prob[child] * u_p0[child]
// -------------------------------------------------------------------------
void GpuCfrSolver::compute_expected_utilities() {
    KernelLaunchConfig launch{config_.threads_per_block};

    // Initialize d_u_p0:
    //
    //   terminal node:     d_u_p0[node] = terminal_u_p0[node]
    //   non-terminal node:  d_u_p0[node] = 0
    //
    // After this, we perform a backward dynamic-programming pass from the
    // deepest level toward the root.
    launch_initialize_terminal_utilities(
        launch,
        gpu_.game.d_terminal_u_p0,
        gpu_.work.d_u_p0,
        flat_.num_nodes
    );

    // Backward utility recurrence:
    //
    //   u_p0[parent] += incoming_prob[child] * u_p0[child]
    //
    // Since level_edges[d] stores edges from depth d to depth d + 1, we process
    // depths in reverse order. This ensures child utilities are complete before
    // the parent is accumulated.
    for (int depth = flat_.max_depth - 1; depth >= 0; --depth) {
        const DeviceLevelEdges& edges =
            gpu_.game.level_edges[static_cast<std::size_t>(depth)];

        if (edges.count == 0) {
            continue;
        }

        launch_backward_utility_level(
            launch,
            edges,
            gpu_.work.d_incoming_prob,
            gpu_.work.d_u_p0,
            gpu_.work.d_u_p0
        );
    }

    // Optional diagnostic: copy root value back to host.
    if (flat_.root >= 0 && flat_.root < flat_.num_nodes && config_.synchronize_each_iteration) {
        check_cuda(
            cudaMemcpy(
                &stats_.last_root_value_p0,
                gpu_.work.d_u_p0 + flat_.root,
                sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "Failed to copy GPU root utility to host"
        );
    }

    check_cuda(
        cudaGetLastError(),
        "Failed to launch compute_expected_utilities kernels"
    );
}
// -------------------------------------------------------------------------
// Stage C: compute reach probabilities.
//
// Your current design uses two reach buffers. Keep their semantics explicit:
//   d_reach_p0 = reach excluding P0's own actions
//   d_reach_p1 = reach excluding P1's own actions
// -------------------------------------------------------------------------
void GpuCfrSolver::compute_counterfactual_reach() {
    KernelLaunchConfig launch{config_.threads_per_block};

    // Initialize reach buffers:
    //
    //   d_reach_p0[root] = 1
    //   d_reach_p1[root] = 1
    //
    // and all non-root entries are 0.
    //
    // Recommended semantics:
    //
    //   d_reach_p0[node] = reach probability excluding P0's own actions
    //   d_reach_p1[node] = reach probability excluding P1's own actions
    //
    // With that convention:
    //
    //   P0 regret should use d_reach_p0
    //   P1 regret should use d_reach_p1
    launch_initialize_reach(
        launch,
        gpu_.work.d_reach_p0,
        gpu_.work.d_reach_p1,
        flat_.num_nodes,
        flat_.root
    );

    // Forward reach recurrence. Process from root depth toward leaves.
    //
    // For each edge parent -> child:
    //
    //   if parent is Chance:
    //       both reach buffers multiply by chance_prob[child]
    //
    //   if parent is P0:
    //       reach excluding P0 does not multiply by sigma[q]
    //       reach excluding P1 does multiply by sigma[q]
    //
    //   if parent is P1:
    //       reach excluding P0 does multiply by sigma[q]
    //       reach excluding P1 does not multiply by sigma[q]
    for (int depth = 0; depth < flat_.max_depth; ++depth) {
        const DeviceLevelEdges& edges =
            gpu_.game.level_edges[static_cast<std::size_t>(depth)];

        if (edges.count == 0) {
            continue;
        }

        launch_forward_reach_level(
            launch,
            edges,
            gpu_.game.d_player,
            gpu_.work.d_incoming_prob,
            gpu_.work.d_reach_p0,
            gpu_.work.d_reach_p1,
            gpu_.work.d_reach_p0,
            gpu_.work.d_reach_p1
        );
    }

    check_cuda(
        cudaGetLastError(),
        "Failed to launch compute_counterfactual_reach kernels"
    );
}
void GpuCfrSolver::compute_own_reach() {
    KernelLaunchConfig launch{config_.threads_per_block};

    // Reuse the initializer because it sets:
    //   root = 1
    //   all other nodes = 0
    //
    // For own reach, this is also the correct initialization.
    launch_initialize_reach(
        launch,
        gpu_.work.d_own_reach_p0,
        gpu_.work.d_own_reach_p1,
        flat_.num_nodes,
        flat_.root
    );

    for (int depth = 0; depth < flat_.max_depth; ++depth) {
        const DeviceLevelEdges& edges =
            gpu_.game.level_edges[static_cast<std::size_t>(depth)];

        if (edges.count == 0) {
            continue;
        }

        launch_forward_own_reach_level(
            launch,
            edges,
            gpu_.game.d_player,
            gpu_.cfr.d_sigma,
            gpu_.work.d_own_reach_p0,
            gpu_.work.d_own_reach_p1,
            gpu_.work.d_own_reach_p0,
            gpu_.work.d_own_reach_p1
        );
    }

    check_cuda(
        cudaGetLastError(),
        "Failed to launch compute_own_reach kernels"
    );
}

void GpuCfrSolver::compute_own_infoset_reach() {
    KernelLaunchConfig launch{config_.threads_per_block};

    launch_compute_own_infoset_reach(
        launch,
        gpu_.game.d_infoset_player,
        gpu_.game.d_infoset_node_begin,
        gpu_.game.d_infoset_node_count,
        gpu_.game.d_infoset_nodes,
        gpu_.work.d_own_reach_p0,
        gpu_.work.d_own_reach_p1,
        gpu_.work.d_own_infoset_reach,
        flat_.num_infosets
    );

    check_cuda(
        cudaGetLastError(),
        "Failed to launch compute_own_infoset_reach kernel"
    );
}

// -------------------------------------------------------------------------
// Stage D: aggregate reach by infoset.
//
// This is needed because multiple concrete tree nodes can belong to one
// information set.
// -------------------------------------------------------------------------
void GpuCfrSolver::compute_infoset_reach() {
    KernelLaunchConfig launch{config_.threads_per_block};

    // For each infoset h, sum the appropriate counterfactual reach over all
    // tree nodes belonging to h.
    //
    // The infoset -> nodes mapping is necessary because several concrete game
    // tree nodes can belong to the same information set.
    launch_compute_infoset_reach(
        launch,
        gpu_.game.d_infoset_player,
        gpu_.game.d_infoset_node_begin,
        gpu_.game.d_infoset_node_count,
        gpu_.game.d_infoset_nodes,
        gpu_.work.d_reach_p0,
        gpu_.work.d_reach_p1,
        gpu_.work.d_counterfactual_infoset_reach,
        flat_.num_infosets
    );

    check_cuda(
        cudaGetLastError(),
        "Failed to launch compute_infoset_reach kernel"
    );
}
// -------------------------------------------------------------------------
// Stage E: compute instantaneous regret.
//
// This must use counterfactual reach:
//   P0 regret uses reach excluding P0
//   P1 regret uses reach excluding P1
// -------------------------------------------------------------------------
void GpuCfrSolver::compute_instantaneous_regrets() {
    KernelLaunchConfig launch{config_.threads_per_block};

    // Clear instantaneous regrets from the previous iteration.
    launch_fill_float(
        launch,
        gpu_.work.d_inst_regret,
        0.0f,
        flat_.num_q
    );

    // For each real-player decision edge parent -> child with q index:
    //
    //   P0 regret:
    //       reach_excluding_p0[parent] * (u_p0[child] - u_p0[parent])
    //
    //   P1 regret:
    //       reach_excluding_p1[parent] * (u_p1[child] - u_p1[parent])
    //
    // Since this project stores only P0 utility and the game is zero-sum:
    //
    //   u_p1[x] = -u_p0[x]
    //
    // Therefore:
    //
    //   P1 regret =
    //       reach_excluding_p1[parent] * (u_p0[parent] - u_p0[child])
    //
    // The kernel should atomicAdd into d_inst_regret[q], because multiple
    // concrete tree nodes can share the same infoset-action q.
    launch_compute_instantaneous_regrets(
        launch,
        gpu_.game.decision_edges,
        gpu_.game.d_player,
        gpu_.work.d_u_p0,
        gpu_.work.d_reach_p0,
        gpu_.work.d_reach_p1,
        gpu_.work.d_inst_regret
    );

    check_cuda(
        cudaGetLastError(),
        "Failed to launch compute_instantaneous_regrets kernels"
    );
}

    void GpuCfrSolver::update_regrets() {
    KernelLaunchConfig launch{config_.threads_per_block};

    // Vanilla CFR:
    //
    //   regret_sum[q] += inst_regret[q]
    //
    // CFR+:
    //
    //   regret_sum[q] = max(0, regret_sum[q] + inst_regret[q])
    //
    // Your config_.use_cfr_plus flag controls that behavior inside the kernel.
    launch_update_regrets(
        launch,
        gpu_.work.d_inst_regret,
        gpu_.cfr.d_regret_sum,
        flat_.num_q,
        config_.use_cfr_plus
    );

    check_cuda(
        cudaGetLastError(),
        "Failed to launch update_regrets kernel"
    );
}

void GpuCfrSolver::update_current_strategy() {
    KernelLaunchConfig launch{config_.threads_per_block};

    // Step 1: clear positive-regret denominators.
    //
    // One denominator per infoset:
    //
    //   positive_regret_sum[h] = sum_q_in_h max(regret_sum[q], 0)
    launch_clear_positive_regret_sums(
        launch,
        gpu_.work.d_positive_regret_sum,
        flat_.num_infosets
    );

    // Step 2: accumulate max(regret_sum[q], 0) into the owning infoset.
    //
    // The kernel should use atomicAdd because many q entries can map to
    // different infosets in parallel.
    launch_accumulate_positive_regret_sums(
        launch,
        gpu_.cfr.d_regret_sum,
        gpu_.game.d_q_infoset,
        gpu_.work.d_positive_regret_sum,
        flat_.num_q
    );

    // Step 3: regret matching.
    //
    // For q belonging to infoset h:
    //
    //   if positive_regret_sum[h] > 0:
    //       sigma[q] = max(regret_sum[q], 0) / positive_regret_sum[h]
    //   else:
    //       sigma[q] = sigma_init[q]
    //
    // This implements the standard regret-matching rule.
    launch_regret_matching(
        launch,
        gpu_.cfr.d_regret_sum,
        gpu_.work.d_positive_regret_sum,
        gpu_.cfr.d_sigma_init,
        gpu_.game.d_q_infoset,
        gpu_.cfr.d_sigma,
        flat_.num_q
    );

    check_cuda(
        cudaGetLastError(),
        "Failed to launch update_current_strategy kernels"
    );
}
// -------------------------------------------------------------------------
// Stage F: accumulate average strategy BEFORE updating sigma.
//
// This is important. The average strategy should receive the strategy that
// was actually used during this iteration, not the next strategy computed
// after regret matching.
// -------------------------------------------------------------------------
void GpuCfrSolver::accumulate_average_strategy() {
    KernelLaunchConfig launch{config_.threads_per_block};

    // Vanilla CFR gives each iteration equal weight.
    // Linear averaging gives iteration t weight t.
    //
    // stats_.iterations_run is zero-based before this iteration is counted, so
    // the current iteration number is stats_.iterations_run + 1.
    const float iteration_weight =
        config_.linear_averaging
            ? static_cast<float>(stats_.iterations_run + 1)
            : 1.0f;

    // Accumulate this iteration's current strategy into the running average.
    //
    // Important:
    //   This must be called before update_regrets() and update_current_strategy()
    //   so that the average receives the strategy actually used on this
    //   iteration.
    //
    // CPU reference:
    //   strategy_sum[q] += iteration_weight * player_reach_weight * sigma[q]
    //
    // The launch function receives both reach buffers and infoset ownership so
    // the kernel can choose the correct reach weight. Your CPU solver uses the
    // acting player's own reach for average-strategy accumulation, while regrets
    // use counterfactual reach. Keep that distinction explicit in kernels.cu.
    launch_accumulate_average_strategy(
        launch,
        gpu_.game.d_q_infoset,
        gpu_.game.d_infoset_q_begin,
        gpu_.cfr.d_sigma,
        gpu_.work.d_own_infoset_reach,
        gpu_.cfr.d_strategy_sum,
        gpu_.cfr.d_strategy_weight_sum,
        flat_.num_q,
        iteration_weight
    );

    check_cuda(
        cudaGetLastError(),
        "Failed to launch accumulate_average_strategy kernel"
    );
}

void GpuCfrSolver::normalize_average_strategy() {
    KernelLaunchConfig launch{config_.threads_per_block};

    // Convert running average-strategy numerator into a normalized strategy:
    //
    //   avg_strategy[q] = strategy_sum[q] / strategy_weight_sum[h]
    //
    // where h = q_infoset[q].
    //
    // If the denominator is zero, fall back to sigma_init[q], which is uniform.
    launch_normalize_average_strategy(
        launch,
        gpu_.game.d_q_infoset,
        gpu_.cfr.d_strategy_sum,
        gpu_.cfr.d_strategy_weight_sum,
        gpu_.cfr.d_sigma_init,
        gpu_.cfr.d_avg_strategy,
        flat_.num_q
    );

    check_cuda(
        cudaGetLastError(),
        "Failed to launch normalize_average_strategy kernel"
    );
}

int GpuCfrSolver::blocks_for(int n) const {
    return poker::blocks_for(n, config_.threads_per_block);
}

// ------------------------------
// Optional gpu_state.hpp helpers
// ------------------------------

// TODO:
//   allocate_gpu_state
//   free_gpu_state
//   reset_work_buffers

} // namespace poker