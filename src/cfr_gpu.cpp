// cfr_gpu.cpp
//
// GPU CFR host implementation for the public-tree / hand-aware Game.
//
// Strategy layout:
//
//   tensor_index =
//       action_state.tensor_offset
//     + bucket * action_state.action_count
//     + local_action
//
// Terminal layout in HostPrecomputed mode:
//
//   terminal_value_p0[terminal_index * hand_pair_count + hand_pair_id]

#include "cfr_gpu.hpp"
#include "kernels.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace poker {
namespace {

// -----------------------------------------------------------------------------
// CUDA helpers
// -----------------------------------------------------------------------------

void check_cuda(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(message) + ": " + cudaGetErrorString(result)
        );
    }
}

template <typename T>
void cuda_alloc_copy(T** dst, const std::vector<T>& src) {
    if (dst == nullptr) {
        throw std::invalid_argument("cuda_alloc_copy received null dst.");
    }

    *dst = nullptr;

    if (src.empty()) {
        return;
    }

    check_cuda(
        cudaMalloc(
            reinterpret_cast<void**>(dst),
            sizeof(T) * src.size()
        ),
        "cudaMalloc failed"
    );

    try {
        check_cuda(
            cudaMemcpy(
                *dst,
                src.data(),
                sizeof(T) * src.size(),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy host-to-device failed"
        );
    } catch (...) {
        cudaFree(*dst);
        *dst = nullptr;
        throw;
    }
}

template <typename T>
void cuda_alloc_zero(T** dst, std::size_t count) {
    if (dst == nullptr) {
        throw std::invalid_argument("cuda_alloc_zero received null dst.");
    }

    *dst = nullptr;

    if (count == 0) {
        return;
    }

    check_cuda(
        cudaMalloc(
            reinterpret_cast<void**>(dst),
            sizeof(T) * count
        ),
        "cudaMalloc failed"
    );

    try {
        check_cuda(
            cudaMemset(
                *dst,
                0,
                sizeof(T) * count
            ),
            "cudaMemset failed"
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

std::size_t checked_mul(
    std::size_t a,
    std::size_t b,
    const char* name
) {
    if (a != 0 &&
        b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::overflow_error(
            std::string(name) + " overflow."
        );
    }

    return a * b;
}

std::uint64_t to_u64(std::size_t value) {
    return static_cast<std::uint64_t>(value);
}

int player_to_int(Player player) {
    return static_cast<int>(player);
}

int node_type_to_int(PublicNodeType type) {
    return static_cast<int>(type);
}

bool is_real_player(Player player) {
    return player == Player::P0 || player == Player::P1;
}

} // namespace

// -----------------------------------------------------------------------------
// Flatten public game
// -----------------------------------------------------------------------------

FlatPublicGame flatten_public_game_for_gpu(
    const Game& game
) {
    game.validate();

    FlatPublicGame flat;

    flat.num_nodes = game.num_nodes();
    flat.num_edges = game.num_edges();
    flat.num_actions = game.num_actions();
    flat.num_action_states = game.num_action_states();
    flat.num_players = game.num_players;
    flat.max_depth = game.max_depth;
    flat.root = game.root;

    flat.tensor_entries = game.cfr_tensor_entries();
    flat.state_bucket_entries = game.state_bucket_entries();

    if (flat.num_nodes <= 0) {
        throw std::runtime_error("Cannot flatten empty Game.");
    }

    if (flat.root < 0 || flat.root >= flat.num_nodes) {
        throw std::runtime_error("Game root is out of range.");
    }

    if (flat.tensor_entries == 0) {
        throw std::runtime_error("Game has zero CFR tensor entries.");
    }

    flat.parent.assign(flat.num_nodes, -1);
    flat.depth.assign(flat.num_nodes, 0);
    flat.player.assign(flat.num_nodes, player_to_int(Player::Terminal));
    flat.node_type.assign(flat.num_nodes, node_type_to_int(PublicNodeType::Terminal));
    flat.action_state_index.assign(flat.num_nodes, -1);

    flat.action_state_node.assign(flat.num_action_states, -1);
    flat.action_state_player.assign(flat.num_action_states, player_to_int(Player::Terminal));
    flat.action_state_bucket_count.assign(flat.num_action_states, 0);
    flat.action_state_action_count.assign(flat.num_action_states, 0);
    flat.action_state_first_action.assign(flat.num_action_states, -1);
    flat.action_state_tensor_offset.assign(flat.num_action_states, 0);
    flat.action_state_bucket_offset.assign(flat.num_action_states, 0);

    flat.level_edges.clear();
    flat.level_edges.resize(static_cast<std::size_t>(flat.max_depth + 1));

    for (int node_id = 0; node_id < flat.num_nodes; ++node_id) {
        const PublicNode& node = game.node(node_id);

        if (node.depth < 0 || node.depth > flat.max_depth) {
            throw std::runtime_error("PublicNode depth out of range.");
        }

        flat.parent[node_id] = node.parent;
        flat.depth[node_id] = node.depth;
        flat.player[node_id] = player_to_int(node.player);
        flat.node_type[node_id] = node_type_to_int(node.type);
        flat.action_state_index[node_id] = node.action_state_index;

        if (node.edge_count < 0) {
            throw std::runtime_error("PublicNode edge_count is negative.");
        }

        if (node.edge_count > 0 &&
            node.first_edge + node.edge_count > flat.num_edges) {
            throw std::runtime_error("PublicNode edge range out of bounds.");
        }

        for (int local = 0; local < node.edge_count; ++local) {
            const int edge_id = node.first_edge + local;
            const NodeEdge& edge = game.edge(edge_id);

            if (edge.child < 0 || edge.child >= flat.num_nodes) {
                throw std::runtime_error("NodeEdge child out of range.");
            }

            const PublicNode& child = game.node(edge.child);

            if (child.parent != node_id) {
                throw std::runtime_error("Parent/child mismatch.");
            }

            if (child.depth != node.depth + 1) {
                throw std::runtime_error("Child depth mismatch.");
            }

            if (node.depth < 0 || node.depth >= flat.max_depth + 1) {
                throw std::runtime_error("Edge parent depth out of range.");
            }

            int local_action = -1;
            float chance_prob = edge.chance_prob;

            if (node.type == PublicNodeType::Action) {
                if (!is_real_player(node.player)) {
                    throw std::runtime_error("Action node has non-real player.");
                }

                if (node.action_state_index < 0 ||
                    node.action_state_index >= flat.num_action_states) {
                    throw std::runtime_error("Action node has invalid action_state_index.");
                }

                local_action = edge.local_action;

                if (local_action != local) {
                    throw std::runtime_error("Action edge local_action mismatch.");
                }

                chance_prob = 1.0f;

                flat.action_edge_parent.push_back(node_id);
                flat.action_edge_child.push_back(edge.child);
                flat.action_edge_state.push_back(node.action_state_index);
                flat.action_edge_local_action.push_back(local_action);
            } else if (node.type == PublicNodeType::Chance) {
                if (node.player != Player::Chance) {
                    throw std::runtime_error("Chance node has invalid player.");
                }

                local_action = -1;

                if (!(chance_prob > 0.0f) || chance_prob > 1.0f) {
                    throw std::runtime_error("Chance edge probability invalid.");
                }
            } else {
                throw std::runtime_error("Terminal node cannot have outgoing edges.");
            }

            FlatPublicLevelEdges& level =
                flat.level_edges[static_cast<std::size_t>(node.depth)];

            level.parent.push_back(node_id);
            level.child.push_back(edge.child);
            level.local_action.push_back(local_action);
            level.chance_prob.push_back(chance_prob);
        }
    }

    for (int state_id = 0; state_id < flat.num_action_states; ++state_id) {
        const ActionState& state = game.action_state(state_id);

        if (state.node < 0 || state.node >= flat.num_nodes) {
            throw std::runtime_error("ActionState node out of range.");
        }

        if (!is_real_player(static_cast<Player>(state.player))) {
            throw std::runtime_error("ActionState player is invalid.");
        }

        if (state.bucket_count <= 0 || state.action_count <= 0) {
            throw std::runtime_error("ActionState has invalid dimensions.");
        }

        const std::size_t tensor_size =
            static_cast<std::size_t>(state.bucket_count) *
            static_cast<std::size_t>(state.action_count);

        if (static_cast<std::size_t>(state.tensor_offset) + tensor_size >
            flat.tensor_entries) {
            throw std::runtime_error("ActionState tensor range out of bounds.");
        }

        if (static_cast<std::size_t>(state.state_bucket_offset) +
                static_cast<std::size_t>(state.bucket_count) >
            flat.state_bucket_entries) {
            throw std::runtime_error("ActionState bucket range out of bounds.");
        }

        flat.action_state_node[state_id] = state.node;
        flat.action_state_player[state_id] = state.player;
        flat.action_state_bucket_count[state_id] = state.bucket_count;
        flat.action_state_action_count[state_id] = state.action_count;
        flat.action_state_first_action[state_id] = state.first_action;
        flat.action_state_tensor_offset[state_id] = state.tensor_offset;
        flat.action_state_bucket_offset[state_id] = state.state_bucket_offset;
    }

    return flat;
}

// -----------------------------------------------------------------------------
// Flatten hand data
// -----------------------------------------------------------------------------

FlatHandData flatten_hand_data_for_gpu(
    const Game& game
) {
    FlatHandData flat;

    flat.p0_hands = game.p0_hands.hands;
    flat.p1_hands = game.p1_hands.hands;

    flat.p0_hand_count = static_cast<int>(flat.p0_hands.size());
    flat.p1_hand_count = static_cast<int>(flat.p1_hands.size());

    flat.p0_pair_index = game.hand_pairs.p0_index;
    flat.p1_pair_index = game.hand_pairs.p1_index;
    flat.hand_pair_count = game.hand_pairs.pair_count();

    if (flat.p0_hand_count <= 0 ||
        flat.p1_hand_count <= 0 ||
        flat.hand_pair_count <= 0) {
        throw std::runtime_error("Invalid hand data while flattening.");
    }

    flat.p0_hand_mask.reserve(flat.p0_hands.size());
    flat.p1_hand_mask.reserve(flat.p1_hands.size());

    for (HandId hand_id : flat.p0_hands) {
        validate_hand_id(hand_id);
        flat.p0_hand_mask.push_back(
            static_cast<std::uint64_t>(hand_from_id(hand_id).mask())
        );
    }

    for (HandId hand_id : flat.p1_hands) {
        validate_hand_id(hand_id);
        flat.p1_hand_mask.push_back(
            static_cast<std::uint64_t>(hand_from_id(hand_id).mask())
        );
    }

    // Current exact-domain mode:
    // bucket == domain-local hand index.
    flat.p0_bucket_by_hand_index.resize(flat.p0_hands.size());
    flat.p1_bucket_by_hand_index.resize(flat.p1_hands.size());

    for (int i = 0; i < flat.p0_hand_count; ++i) {
        flat.p0_bucket_by_hand_index[static_cast<std::size_t>(i)] = i;
    }

    for (int j = 0; j < flat.p1_hand_count; ++j) {
        flat.p1_bucket_by_hand_index[static_cast<std::size_t>(j)] = j;
    }

    flat.p0_bucket_count = flat.p0_hand_count;
    flat.p1_bucket_count = flat.p1_hand_count;

    return flat;
}

// -----------------------------------------------------------------------------
// Flatten terminal metadata / values
// -----------------------------------------------------------------------------

FlatTerminalData flatten_terminal_metadata_for_gpu(
    const Game& game
) {
    FlatTerminalData flat;

    flat.terminal_index_by_node.assign(game.num_nodes(), -1);

    for (int node_id = 0; node_id < game.num_nodes(); ++node_id) {
        const PublicNode& node = game.node(node_id);

        if (node.type != PublicNodeType::Terminal) {
            continue;
        }

        const int terminal_index =
            static_cast<int>(flat.terminal_nodes.size());

        flat.terminal_nodes.push_back(node_id);
        flat.terminal_index_by_node[static_cast<std::size_t>(node_id)] =
            terminal_index;
    }

    if (flat.terminal_nodes.empty()) {
        throw std::runtime_error("Game contains no terminal nodes.");
    }

    return flat;
}

FlatTerminalData flatten_terminal_data_for_gpu(const Game& game) {
    FlatTerminalData flat = flatten_terminal_metadata_for_gpu(game);

    const std::size_t expected =
        checked_mul(
            static_cast<std::size_t>(flat.terminal_count()),
            static_cast<std::size_t>(game.hand_pairs.pair_count()),
            "terminal_value_p0 size"
        );

    if (game.terminal_value_p0.size() != expected) {
        throw std::invalid_argument(
            "terminal_value_p0 size must equal terminal_count * hand_pair_count."
        );
    }

    flat.terminal_value_p0 = game.terminal_value_p0;
    return flat;
}

// -----------------------------------------------------------------------------
// Construction / destruction
// -----------------------------------------------------------------------------

GpuCfrSolver::GpuCfrSolver(const Game& game)
    : GpuCfrSolver(game, GpuCfrConfig{}) {
    initialize();
}

GpuCfrSolver::GpuCfrSolver(
    const Game& game,
    GpuCfrConfig config
)
    : game_(game),
      config_(config),
      stats_{},
      gpu_{},
      initialized_(false),
      terminal_values_uploaded_(false) {
    initialize();
}

GpuCfrSolver::~GpuCfrSolver() {
    release();
}

// -----------------------------------------------------------------------------
// Debug Helpers
// -----------------------------------------------------------------------------
std::vector<float> GpuCfrSolver::debug_action_value_p0() const {
    return copy_tensor_from_device(
        gpu_.work.d_action_value_p0,
        gpu_.flat.tensor_entries
    );
}

std::vector<float> GpuCfrSolver::debug_state_bucket_cf_reach() const {
    return copy_tensor_from_device(
        gpu_.work.d_state_bucket_cf_reach,
        gpu_.flat.state_bucket_entries
    );
}

std::vector<float> GpuCfrSolver::debug_state_bucket_own_reach() const {
    return copy_tensor_from_device(
        gpu_.work.d_state_bucket_own_reach,
        gpu_.flat.state_bucket_entries
    );
}

std::vector<float> GpuCfrSolver::debug_state_bucket_value_p0() const {
    return copy_tensor_from_device(
        gpu_.work.d_state_bucket_value_p0,
        gpu_.flat.state_bucket_entries
    );
}

std::vector<float> GpuCfrSolver::debug_node_pair_value_p0() const {
    return copy_tensor_from_device(
        gpu_.work.d_node_pair_value_p0,
        gpu_.work.node_pair_value_entries
    );
}
void GpuCfrSolver::debug_dump_action_value_launch_inputs() const {
    const auto& flat = gpu_.flat;
    const auto& dev_game = gpu_.game;
    const auto& hand = gpu_.hand_data;
    const auto& work = gpu_.work;

    std::cerr << "\n=== action-value launch inputs ===\n";

    std::cerr
        << "threads_per_block=" << config_.threads_per_block << "\n"
        << "flat.num_nodes=" << flat.num_nodes << "\n"
        << "flat.num_edges=" << flat.num_edges << "\n"
        << "flat.num_action_states=" << flat.num_action_states << "\n"
        << "flat.tensor_entries=" << flat.tensor_entries << "\n"
        << "flat.state_bucket_entries=" << flat.state_bucket_entries << "\n"
        << "hand_pair_count=" << hand.hand_pair_count << "\n"
        << "action_edges.count=" << dev_game.action_edges.count << "\n";

    std::cerr
        << "ptr d_action_edge_parent=" << dev_game.action_edges.d_parent << "\n"
        << "ptr d_action_edge_child=" << dev_game.action_edges.d_child << "\n"
        << "ptr d_action_edge_state=" << dev_game.action_edges.d_action_state << "\n"
        << "ptr d_action_edge_local_action=" << dev_game.action_edges.d_local_action << "\n"
        << "ptr d_action_state_player=" << dev_game.d_action_state_player << "\n"
        << "ptr d_action_state_bucket_count=" << dev_game.d_action_state_bucket_count << "\n"
        << "ptr d_action_state_action_count=" << dev_game.d_action_state_action_count << "\n"
        << "ptr d_action_state_tensor_offset=" << dev_game.d_action_state_tensor_offset << "\n"
        << "ptr d_action_state_bucket_offset=" << dev_game.d_action_state_bucket_offset << "\n"
        << "ptr d_p0_pair_index=" << hand.d_p0_pair_index << "\n"
        << "ptr d_p1_pair_index=" << hand.d_p1_pair_index << "\n"
        << "ptr d_p0_bucket_by_hand_index=" << hand.d_p0_bucket_by_hand_index << "\n"
        << "ptr d_p1_bucket_by_hand_index=" << hand.d_p1_bucket_by_hand_index << "\n"
        << "ptr d_node_pair_value_p0=" << work.d_node_pair_value_p0 << "\n"
        << "ptr d_node_pair_reach_p0=" << work.d_node_pair_reach_p0 << "\n"
        << "ptr d_node_pair_reach_p1=" << work.d_node_pair_reach_p1 << "\n"
        << "ptr d_node_pair_reach_chance=" << work.d_node_pair_reach_chance << "\n"
        << "ptr d_state_bucket_cf_reach=" << work.d_state_bucket_cf_reach << "\n"
        << "ptr d_action_value_p0=" << work.d_action_value_p0 << "\n";

    const int root = flat.root;
    const PublicNode& root_node = game_.node(root);

    std::cerr
        << "root=" << root << "\n"
        << "root.type=" << static_cast<int>(root_node.type) << "\n"
        << "root.player=" << static_cast<int>(root_node.player) << "\n"
        << "root.edge_count=" << root_node.edge_count << "\n"
        << "root.action_state_index=" << root_node.action_state_index << "\n";

    if (root_node.action_state_index >= 0) {
        const ActionState& state =
            game_.action_state(root_node.action_state_index);

        std::cerr
            << "root_state.node=" << state.node << "\n"
            << "root_state.player=" << state.player << "\n"
            << "root_state.bucket_count=" << state.bucket_count << "\n"
            << "root_state.action_count=" << state.action_count << "\n"
            << "root_state.tensor_offset=" << state.tensor_offset << "\n"
            << "root_state.state_bucket_offset=" << state.state_bucket_offset << "\n";
    }

    int root_action_edges = 0;

    for (std::size_t i = 0; i < flat.action_edge_parent.size(); ++i) {
        if (flat.action_edge_parent[i] != root) {
            continue;
        }

        ++root_action_edges;

        std::cerr
            << "flat root action edge i=" << i
            << " parent=" << flat.action_edge_parent[i]
            << " child=" << flat.action_edge_child[i]
            << " state=" << flat.action_edge_state[i]
            << " local_action=" << flat.action_edge_local_action[i]
            << "\n";
    }

    std::cerr
        << "flat root action edge count=" << root_action_edges << "\n"
        << "=== end action-value launch inputs ===\n";
}
// -----------------------------------------------------------------------------
// Initialization / cleanup
// -----------------------------------------------------------------------------

void GpuCfrSolver::initialize() {
    if (initialized_) {
        throw std::logic_error("GpuCfrSolver is already initialized.");
    }

    if (config_.threads_per_block <= 0) {
        throw std::invalid_argument("threads_per_block must be positive.");
    }
    gpu_.flat = flatten_public_game_for_gpu(game_);
    gpu_.hands = flatten_hand_data_for_gpu(game_);

    gpu_.terminals = flatten_terminal_data_for_gpu(game_);
    try {
        upload_static_game();
        upload_hand_data();
        upload_terminal_data();

        allocate_cfr_state();
        allocate_work_buffers();

        initialize_strategy();

        initialized_ = true;
    } catch (...) {
        release();
        throw;
    }

    stats_.tensor_entries = gpu_.flat.tensor_entries;
    stats_.state_bucket_entries = gpu_.flat.state_bucket_entries;
    stats_.hand_pair_count = static_cast<std::size_t>(gpu_.hands.hand_pair_count);
}

void GpuCfrSolver::release() {
    DevicePublicGameData& game = gpu_.game;
    DeviceHandData& hands = gpu_.hand_data;
    DeviceTerminalData& terminals = gpu_.terminal_data;
    DevicePublicCfrState& cfr = gpu_.cfr;
    DevicePublicWorkBuffers& work = gpu_.work;

    cuda_free_ptr(game.d_parent);
    cuda_free_ptr(game.d_depth);
    cuda_free_ptr(game.d_player);
    cuda_free_ptr(game.d_node_type);
    cuda_free_ptr(game.d_terminal_type);
    cuda_free_ptr(game.d_action_state_index);

    cuda_free_ptr(game.d_board_index);
    cuda_free_ptr(game.d_pot);
    cuda_free_ptr(game.d_p0_stack);
    cuda_free_ptr(game.d_p1_stack);

    cuda_free_ptr(game.d_action_state_node);
    cuda_free_ptr(game.d_action_state_player);
    cuda_free_ptr(game.d_action_state_bucket_count);
    cuda_free_ptr(game.d_action_state_action_count);
    cuda_free_ptr(game.d_action_state_first_action);
    cuda_free_ptr(game.d_action_state_tensor_offset);
    cuda_free_ptr(game.d_action_state_bucket_offset);

    for (DevicePublicLevelEdges& level : game.level_edges) {
        cuda_free_ptr(level.d_parent);
        cuda_free_ptr(level.d_child);
        cuda_free_ptr(level.d_local_action);
        cuda_free_ptr(level.d_chance_prob);
        level.count = 0;
    }

    game.level_edges.clear();

    cuda_free_ptr(game.action_edges.d_parent);
    cuda_free_ptr(game.action_edges.d_child);
    cuda_free_ptr(game.action_edges.d_action_state);
    cuda_free_ptr(game.action_edges.d_local_action);
    game.action_edges.count = 0;

    cuda_free_ptr(hands.d_p0_hands);
    cuda_free_ptr(hands.d_p1_hands);
    cuda_free_ptr(hands.d_p0_hand_mask);
    cuda_free_ptr(hands.d_p1_hand_mask);
    cuda_free_ptr(hands.d_p0_pair_index);
    cuda_free_ptr(hands.d_p1_pair_index);
    cuda_free_ptr(hands.d_p0_bucket_by_hand_index);
    cuda_free_ptr(hands.d_p1_bucket_by_hand_index);

    cuda_free_ptr(terminals.d_terminal_nodes);
    cuda_free_ptr(terminals.d_terminal_index_by_node);
    cuda_free_ptr(terminals.d_terminal_value_p0);

    cuda_free_ptr(cfr.d_sigma);
    cuda_free_ptr(cfr.d_sigma_init);
    cuda_free_ptr(cfr.d_regret_sum);
    cuda_free_ptr(cfr.d_strategy_sum);
    cuda_free_ptr(cfr.d_avg_strategy);
    cuda_free_ptr(cfr.d_strategy_weight_sum);

    cuda_free_ptr(work.d_node_pair_value_p0);
    cuda_free_ptr(work.d_node_pair_value_p0_next);
    cuda_free_ptr(work.d_action_value_p0);
    cuda_free_ptr(work.d_state_bucket_value_p0);
    cuda_free_ptr(work.d_state_bucket_cf_reach);
    cuda_free_ptr(work.d_state_bucket_own_reach);
    cuda_free_ptr(work.d_node_pair_reach_p0);
    cuda_free_ptr(work.d_node_pair_reach_p1);
    cuda_free_ptr(work.d_node_pair_reach_chance);

    gpu_.game = DevicePublicGameData{};
    gpu_.hand_data = DeviceHandData{};
    gpu_.terminal_data = DeviceTerminalData{};
    gpu_.cfr = DevicePublicCfrState{};
    gpu_.work = DevicePublicWorkBuffers{};

    initialized_ = false;
    terminal_values_uploaded_ = false;
}

// -----------------------------------------------------------------------------
// Upload
// -----------------------------------------------------------------------------

void GpuCfrSolver::upload_static_game() {
    const FlatPublicGame& flat = gpu_.flat;
    DevicePublicGameData& game = gpu_.game;

    game.num_nodes = flat.num_nodes;
    game.num_edges = flat.num_edges;
    game.num_actions = flat.num_actions;
    game.num_action_states = flat.num_action_states;
    game.num_players = flat.num_players;
    game.max_depth = flat.max_depth;
    game.root = flat.root;
    game.tensor_entries = flat.tensor_entries;
    game.state_bucket_entries = flat.state_bucket_entries;

    cuda_alloc_copy(&game.d_parent, flat.parent);
    cuda_alloc_copy(&game.d_depth, flat.depth);
    cuda_alloc_copy(&game.d_player, flat.player);
    cuda_alloc_copy(&game.d_node_type, flat.node_type);
    cuda_alloc_copy(&game.d_action_state_index, flat.action_state_index);

    cuda_alloc_copy(&game.d_action_state_node, flat.action_state_node);
    cuda_alloc_copy(&game.d_action_state_player, flat.action_state_player);
    cuda_alloc_copy(&game.d_action_state_bucket_count, flat.action_state_bucket_count);
    cuda_alloc_copy(&game.d_action_state_action_count, flat.action_state_action_count);
    cuda_alloc_copy(&game.d_action_state_first_action, flat.action_state_first_action);
    cuda_alloc_copy(&game.d_action_state_tensor_offset, flat.action_state_tensor_offset);
    cuda_alloc_copy(&game.d_action_state_bucket_offset, flat.action_state_bucket_offset);

    game.level_edges.resize(flat.level_edges.size());

    for (std::size_t depth = 0; depth < flat.level_edges.size(); ++depth) {
        const FlatPublicLevelEdges& src = flat.level_edges[depth];
        DevicePublicLevelEdges& dst = game.level_edges[depth];

        dst.count = src.size();

        cuda_alloc_copy(&dst.d_parent, src.parent);
        cuda_alloc_copy(&dst.d_child, src.child);
        cuda_alloc_copy(&dst.d_local_action, src.local_action);
        cuda_alloc_copy(&dst.d_chance_prob, src.chance_prob);
    }

    game.action_edges.count =
        static_cast<int>(flat.action_edge_parent.size());

    cuda_alloc_copy(&game.action_edges.d_parent, flat.action_edge_parent);
    cuda_alloc_copy(&game.action_edges.d_child, flat.action_edge_child);
    cuda_alloc_copy(&game.action_edges.d_action_state, flat.action_edge_state);
    cuda_alloc_copy(&game.action_edges.d_local_action, flat.action_edge_local_action);
}

void GpuCfrSolver::upload_hand_data() {
    const FlatHandData& flat = gpu_.hands;
    DeviceHandData& hands = gpu_.hand_data;

    hands.p0_hand_count = flat.p0_hand_count;
    hands.p1_hand_count = flat.p1_hand_count;
    hands.hand_pair_count = flat.hand_pair_count;
    hands.p0_bucket_count = flat.p0_bucket_count;
    hands.p1_bucket_count = flat.p1_bucket_count;

    cuda_alloc_copy(&hands.d_p0_hands, flat.p0_hands);
    cuda_alloc_copy(&hands.d_p1_hands, flat.p1_hands);

    cuda_alloc_copy(&hands.d_p0_hand_mask, flat.p0_hand_mask);
    cuda_alloc_copy(&hands.d_p1_hand_mask, flat.p1_hand_mask);

    cuda_alloc_copy(&hands.d_p0_pair_index, flat.p0_pair_index);
    cuda_alloc_copy(&hands.d_p1_pair_index, flat.p1_pair_index);

    cuda_alloc_copy(&hands.d_p0_bucket_by_hand_index, flat.p0_bucket_by_hand_index);
    cuda_alloc_copy(&hands.d_p1_bucket_by_hand_index, flat.p1_bucket_by_hand_index);
}

void GpuCfrSolver::upload_terminal_data() {
    const FlatTerminalData& flat = gpu_.terminals;
    DeviceTerminalData& terminals = gpu_.terminal_data;

    terminals.terminal_count = flat.terminal_count();
    terminals.hand_pair_count = gpu_.hands.hand_pair_count;

    cuda_alloc_copy(&terminals.d_terminal_nodes, flat.terminal_nodes);
    cuda_alloc_copy(
        &terminals.d_terminal_index_by_node,
        flat.terminal_index_by_node
    );

    if (!flat.terminal_value_p0.empty()) {
        cuda_alloc_copy(
            &terminals.d_terminal_value_p0,
            flat.terminal_value_p0
        );

        terminal_values_uploaded_ = true;
    } else {
        terminal_values_uploaded_ = false;
    }
}

void GpuCfrSolver::set_terminal_values(
    std::vector<float> terminal_value_p0
) {
    const std::size_t expected =
        checked_mul(
            static_cast<std::size_t>(gpu_.terminals.terminal_count()),
            static_cast<std::size_t>(gpu_.hands.hand_pair_count),
            "terminal values"
        );

    if (terminal_value_p0.size() != expected) {
        throw std::invalid_argument(
            "terminal_value_p0 size must equal terminal_count * hand_pair_count."
        );
    }

    gpu_.terminals.terminal_value_p0 = std::move(terminal_value_p0);

    cuda_free_ptr(gpu_.terminal_data.d_terminal_value_p0);

    cuda_alloc_copy(
        &gpu_.terminal_data.d_terminal_value_p0,
        gpu_.terminals.terminal_value_p0
    );

    terminal_values_uploaded_ = true;
}

// -----------------------------------------------------------------------------
// Allocation
// -----------------------------------------------------------------------------

void GpuCfrSolver::allocate_cfr_state() {
    DevicePublicCfrState& cfr = gpu_.cfr;

    cuda_alloc_zero(&cfr.d_sigma, gpu_.flat.tensor_entries);
    cuda_alloc_zero(&cfr.d_sigma_init, gpu_.flat.tensor_entries);
    cuda_alloc_zero(&cfr.d_regret_sum, gpu_.flat.tensor_entries);
    cuda_alloc_zero(&cfr.d_strategy_sum, gpu_.flat.tensor_entries);
    cuda_alloc_zero(&cfr.d_avg_strategy, gpu_.flat.tensor_entries);

    cuda_alloc_zero(
        &cfr.d_strategy_weight_sum,
        gpu_.flat.state_bucket_entries
    );
}

    void GpuCfrSolver::allocate_work_buffers() {
    DevicePublicWorkBuffers& work = gpu_.work;

    work.node_pair_value_entries =
        checked_mul(
            static_cast<std::size_t>(gpu_.flat.num_nodes),
            static_cast<std::size_t>(gpu_.hands.hand_pair_count),
            "node_pair_value_entries"
        );

    cuda_alloc_zero(
        &work.d_node_pair_value_p0,
        work.node_pair_value_entries
    );

    cuda_alloc_zero(
        &work.d_node_pair_value_p0_next,
        work.node_pair_value_entries
    );

    cuda_alloc_zero(
        &work.d_node_pair_reach_p0,
        work.node_pair_value_entries
    );

    cuda_alloc_zero(
        &work.d_node_pair_reach_p1,
        work.node_pair_value_entries
    );

    cuda_alloc_zero(
        &work.d_node_pair_reach_chance,
        work.node_pair_value_entries
    );

    cuda_alloc_zero(
        &work.d_action_value_p0,
        gpu_.flat.tensor_entries
    );

    cuda_alloc_zero(
        &work.d_state_bucket_value_p0,
        gpu_.flat.state_bucket_entries
    );

    cuda_alloc_zero(
        &work.d_state_bucket_cf_reach,
        gpu_.flat.state_bucket_entries
    );

    cuda_alloc_zero(
        &work.d_state_bucket_own_reach,
        gpu_.flat.state_bucket_entries
    );
}

void GpuCfrSolver::initialize_strategy() const {
    KernelLaunchConfig launch;
    launch.threads_per_block = config_.threads_per_block;

    launch_initialize_public_uniform_strategy(
        launch,
        gpu_.game.d_action_state_bucket_count,
        gpu_.game.d_action_state_action_count,
        gpu_.game.d_action_state_tensor_offset,
        gpu_.cfr.d_sigma,
        gpu_.cfr.d_sigma_init,
        gpu_.game.num_action_states
    );

    check_cuda(
        cudaGetLastError(),
        "launch_initialize_public_uniform_strategy failed"
    );
}

// -----------------------------------------------------------------------------
// Run API
// -----------------------------------------------------------------------------

void GpuCfrSolver::run_iterations(int iterations) {
    if (iterations < 0) {
        throw std::invalid_argument("iterations must be nonnegative.");
    }

    for (int i = 0; i < iterations; ++i) {
        run_one_iteration();
    }
}

void GpuCfrSolver::run_one_iteration() {
    run_terminal_evaluation_pass();
    run_backward_value_pass();
    run_reach_pass();
    run_regret_update_pass();
    run_average_strategy_accumulation_pass();
    run_strategy_update_pass();

    ++stats_.iterations_run;

    if (config_.synchronize_each_iteration) {
        check_cuda(
            cudaDeviceSynchronize(),
            "cudaDeviceSynchronize failed after CFR iteration"
        );
    } else {
        check_cuda(
            cudaGetLastError(),
            "CUDA kernel launch failed during CFR iteration"
        );
    }
}

// -----------------------------------------------------------------------------
// Iteration stages
// -----------------------------------------------------------------------------

void GpuCfrSolver::run_terminal_evaluation_pass() const {
    KernelLaunchConfig launch;
    launch.threads_per_block = config_.threads_per_block;

    if (config_.terminal_mode == GpuTerminalMode::HostPrecomputed) {
        if (!terminal_values_uploaded_) {
            throw std::logic_error(
                "HostPrecomputed terminal mode requires uploaded terminal values."
            );
        }

        launch_load_precomputed_terminal_pair_values(
            launch,
            gpu_.game.num_nodes,
            gpu_.hand_data.hand_pair_count,
            gpu_.terminal_data.terminal_count,
            gpu_.terminal_data.d_terminal_index_by_node,
            gpu_.terminal_data.d_terminal_value_p0,
            gpu_.work.d_node_pair_value_p0
        );

        check_cuda(
            cudaGetLastError(),
            "launch_load_precomputed_terminal_pair_values failed"
        );

        return;
    }

    throw std::logic_error(
        "GpuTerminalMode::DeviceComputed is not implemented in this cfr_gpu.cpp."
    );
}

    void GpuCfrSolver::run_backward_value_pass() {
    KernelLaunchConfig launch;
    launch.threads_per_block = config_.threads_per_block;

    for (int depth = gpu_.game.max_depth - 1; depth >= 0; --depth) {
        const DevicePublicLevelEdges& edges =
            gpu_.game.level_edges[static_cast<std::size_t>(depth)];

        if (edges.count == 0) {
            continue;
        }

        launch_public_backward_pair_value_level(
            launch,
            edges,
            gpu_.game.d_node_type,
            gpu_.game.d_player,
            gpu_.game.d_action_state_index,
            gpu_.game.d_action_state_action_count,
            gpu_.game.d_action_state_tensor_offset,
            gpu_.hand_data.d_p0_pair_index,
            gpu_.hand_data.d_p1_pair_index,
            gpu_.hand_data.d_p0_bucket_by_hand_index,
            gpu_.hand_data.d_p1_bucket_by_hand_index,
            gpu_.cfr.d_sigma,

            // Read child values from the same full node-value table.
            gpu_.work.d_node_pair_value_p0,

            // Write parent values back into the same table.
            gpu_.work.d_node_pair_value_p0,

            gpu_.hand_data.hand_pair_count
        );

        check_cuda(
            cudaGetLastError(),
            "launch_public_backward_pair_value_level failed"
        );
    }

    check_cuda(
        cudaMemcpy(
            &stats_.last_root_value_p0,
            gpu_.work.d_node_pair_value_p0 +
                gpu_.game.root * gpu_.hand_data.hand_pair_count,
            sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "Failed to copy diagnostic root pair value"
    );
}

void GpuCfrSolver::run_reach_pass() {
    KernelLaunchConfig launch;
    launch.threads_per_block = config_.threads_per_block;

    // Clear pair-level reaches.
    launch_fill_float(
        launch,
        gpu_.work.d_node_pair_reach_p0,
        0.0f,
        gpu_.work.node_pair_value_entries
    );

    launch_fill_float(
        launch,
        gpu_.work.d_node_pair_reach_p1,
        0.0f,
        gpu_.work.node_pair_value_entries
    );

    launch_fill_float(
        launch,
        gpu_.work.d_node_pair_reach_chance,
        0.0f,
        gpu_.work.node_pair_value_entries
    );

    // Clear state-bucket reaches.
    launch_fill_float(
        launch,
        gpu_.work.d_state_bucket_cf_reach,
        0.0f,
        gpu_.flat.state_bucket_entries
    );

    launch_fill_float(
        launch,
        gpu_.work.d_state_bucket_own_reach,
        0.0f,
        gpu_.flat.state_bucket_entries
    );

    check_cuda(
        cudaGetLastError(),
        "Failed to clear reach buffers"
    );

    // Root reach:
    //
    //   reach_p0[root,pair] = 1
    //   reach_p1[root,pair] = 1
    //   reach_chance[root,pair] = 1
    launch_initialize_public_pair_reaches(
        launch,
        gpu_.game.root,
        gpu_.hand_data.hand_pair_count,
        gpu_.work.d_node_pair_reach_p0,
        gpu_.work.d_node_pair_reach_p1,
        gpu_.work.d_node_pair_reach_chance
    );

    check_cuda(
        cudaGetLastError(),
        "launch_initialize_public_pair_reaches failed"
    );

    // Forward pass through public tree.
    for (int depth = 0; depth < gpu_.game.max_depth; ++depth) {
        const DevicePublicLevelEdges& edges =
            gpu_.game.level_edges[static_cast<std::size_t>(depth)];

        if (edges.count == 0) {
            continue;
        }

        launch_public_forward_pair_reach_level(
            launch,
            edges,
            gpu_.game.d_node_type,
            gpu_.game.d_player,
            gpu_.game.d_action_state_index,
            gpu_.game.d_action_state_action_count,
            gpu_.game.d_action_state_tensor_offset,
            gpu_.hand_data.d_p0_pair_index,
            gpu_.hand_data.d_p1_pair_index,
            gpu_.hand_data.d_p0_bucket_by_hand_index,
            gpu_.hand_data.d_p1_bucket_by_hand_index,
            gpu_.cfr.d_sigma,
            gpu_.work.d_node_pair_reach_p0,
            gpu_.work.d_node_pair_reach_p1,
            gpu_.work.d_node_pair_reach_chance,
            gpu_.hand_data.hand_pair_count
        );

        check_cuda(
            cudaGetLastError(),
            "launch_public_forward_pair_reach_level failed"
        );
    }

    // Aggregate pair-level reaches into state-bucket reaches.
    launch_public_aggregate_state_bucket_reaches(
        launch,
        gpu_.game.d_action_state_node,
        gpu_.game.d_action_state_player,
        gpu_.game.d_action_state_bucket_count,
        gpu_.game.d_action_state_bucket_offset,
        gpu_.hand_data.hand_pair_count,
        gpu_.hand_data.d_p0_pair_index,
        gpu_.hand_data.d_p1_pair_index,
        gpu_.hand_data.d_p0_bucket_by_hand_index,
        gpu_.hand_data.d_p1_bucket_by_hand_index,
        gpu_.work.d_node_pair_reach_p0,
        gpu_.work.d_node_pair_reach_p1,
        gpu_.work.d_node_pair_reach_chance,
        gpu_.work.d_state_bucket_cf_reach,
        gpu_.work.d_state_bucket_own_reach,
        gpu_.game.num_action_states
    );

    check_cuda(
        cudaGetLastError(),
        "launch_public_aggregate_state_bucket_reaches failed"
    );
}

void GpuCfrSolver::run_regret_update_pass() {
    KernelLaunchConfig launch;
    launch.threads_per_block = config_.threads_per_block;

    // debug_dump_action_value_launch_inputs();

    launch_public_compute_action_values_from_pair_values(
        launch,
        gpu_.game.action_edges,
        gpu_.game.d_action_state_player,
        gpu_.game.d_action_state_bucket_count,
        gpu_.game.d_action_state_action_count,
        gpu_.game.d_action_state_tensor_offset,
        gpu_.game.d_action_state_bucket_offset,
        gpu_.hand_data.hand_pair_count,
        gpu_.hand_data.d_p0_pair_index,
        gpu_.hand_data.d_p1_pair_index,
        gpu_.hand_data.d_p0_bucket_by_hand_index,
        gpu_.hand_data.d_p1_bucket_by_hand_index,
        gpu_.work.d_node_pair_value_p0,
        gpu_.work.d_node_pair_reach_p0,
        gpu_.work.d_node_pair_reach_p1,
        gpu_.work.d_node_pair_reach_chance,
        gpu_.work.d_state_bucket_cf_reach,
        gpu_.work.d_action_value_p0
    );

    check_cuda(
        cudaGetLastError(),
        "launch_public_compute_action_values_from_pair_values failed"
    );

    launch_public_update_regrets(
        launch,
        gpu_.game.d_action_state_player,
        gpu_.game.d_action_state_bucket_count,
        gpu_.game.d_action_state_action_count,
        gpu_.game.d_action_state_tensor_offset,
        gpu_.game.d_action_state_bucket_offset,
        gpu_.work.d_action_value_p0,
        gpu_.work.d_state_bucket_value_p0,
        gpu_.work.d_state_bucket_cf_reach,
        gpu_.cfr.d_sigma,
        gpu_.cfr.d_regret_sum,
        gpu_.game.num_action_states,
        config_.use_cfr_plus
    );

    check_cuda(
        cudaGetLastError(),
        "launch_public_update_regrets failed"
    );
}

void GpuCfrSolver::run_average_strategy_accumulation_pass() {
    KernelLaunchConfig launch;
    launch.threads_per_block = config_.threads_per_block;

    const float iteration_weight =
        config_.linear_averaging
            ? static_cast<float>(stats_.iterations_run + 1)
            : 1.0f;

    launch_public_accumulate_average_strategy(
        launch,
        gpu_.game.d_action_state_bucket_count,
        gpu_.game.d_action_state_action_count,
        gpu_.game.d_action_state_tensor_offset,
        gpu_.game.d_action_state_bucket_offset,
        gpu_.work.d_state_bucket_own_reach,
        gpu_.cfr.d_sigma,
        gpu_.cfr.d_strategy_sum,
        gpu_.cfr.d_strategy_weight_sum,
        gpu_.game.num_action_states,
        iteration_weight
    );

    check_cuda(
        cudaGetLastError(),
        "launch_public_accumulate_average_strategy failed"
    );

    launch_public_normalize_average_strategy(
        launch,
        gpu_.game.d_action_state_bucket_count,
        gpu_.game.d_action_state_action_count,
        gpu_.game.d_action_state_tensor_offset,
        gpu_.game.d_action_state_bucket_offset,
        gpu_.cfr.d_strategy_sum,
        gpu_.cfr.d_strategy_weight_sum,
        gpu_.cfr.d_avg_strategy,
        gpu_.game.num_action_states
    );

    check_cuda(
        cudaGetLastError(),
        "launch_public_normalize_average_strategy failed"
    );
}

void GpuCfrSolver::run_strategy_update_pass() {
    KernelLaunchConfig launch;
    launch.threads_per_block = config_.threads_per_block;

    launch_public_regret_matching(
        launch,
        gpu_.game.d_action_state_bucket_count,
        gpu_.game.d_action_state_action_count,
        gpu_.game.d_action_state_tensor_offset,
        gpu_.cfr.d_regret_sum,
        gpu_.cfr.d_sigma,
        gpu_.cfr.d_sigma_init,
        gpu_.game.num_action_states
    );

    check_cuda(
        cudaGetLastError(),
        "launch_public_regret_matching failed"
    );
}

// -----------------------------------------------------------------------------
// Accessors
// -----------------------------------------------------------------------------

std::vector<float> GpuCfrSolver::current_strategy() const {
    return copy_tensor_from_device(
        gpu_.cfr.d_sigma,
        gpu_.flat.tensor_entries
    );
}

std::vector<float> GpuCfrSolver::average_strategy() const {
    return copy_tensor_from_device(
        gpu_.cfr.d_avg_strategy,
        gpu_.flat.tensor_entries
    );
}

std::vector<float> GpuCfrSolver::regret_sum() const {
    return copy_tensor_from_device(
        gpu_.cfr.d_regret_sum,
        gpu_.flat.tensor_entries
    );
}

const GpuCfrStats& GpuCfrSolver::stats() const {
    return stats_;
}

const FlatPublicGame& GpuCfrSolver::flat_game() const {
    return gpu_.flat;
}

const FlatHandData& GpuCfrSolver::flat_hand_data() const {
    return gpu_.hands;
}

const FlatTerminalData& GpuCfrSolver::flat_terminal_data() const {
    return gpu_.terminals;
}

std::vector<float> GpuCfrSolver::copy_tensor_from_device(
    const float* device_ptr,
    std::size_t count
) const {
    if (count == 0) {
        return {};
    }

    if (device_ptr == nullptr) {
        throw std::runtime_error("copy_tensor_from_device received null pointer.");
    }

    std::vector<float> result(count, 0.0f);

    check_cuda(
        cudaMemcpy(
            result.data(),
            device_ptr,
            sizeof(float) * count,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy device-to-host failed"
    );

    return result;
}

} // namespace poker