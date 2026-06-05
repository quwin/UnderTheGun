#pragma once

#include "game.hpp"

#include <cstdint>
#include <vector>

namespace poker {

// -----------------------------------------------------------------------------
// GPU CFR configuration / stats
// -----------------------------------------------------------------------------

enum class TerminalMode : int {
    // Terminal values are supplied by the game tree as: terminal_value[terminal_index][hand_pair]
    // This is simplest for validation, but takes up [terminal_nodes] * sizeOf(float) * [hand_pairs] memory,
    // quickly scaling out of hands
    ValuePrecomputed = 0,
    // Terminal values are computed during CFR from board, terminal type,
    // hand domains, and hand-pair table.
    // Instead, takes [total_nodes] * sizeOf(TerminalRecord) memory
    RecordComputed = 1,
    // Debug value to create both terminal_values and TerminalRecords.
    DebugComputed = 2,
};

struct GpuCfrConfig {
    int num_players = 2;
    int threads_per_block = 512;
    // CFR+ clips cumulative regret at zero after each update.
    bool use_cfr_plus = false;
    // Vanilla CFR uses average weight 1.
    // CFR+ commonly uses linear or later-iteration weighting.
    bool linear_averaging = false;
    // Debug only. Synchronizing every iteration is expensive.
    bool synchronize_each_iteration = false;
    // Terminal evaluation mode.
    TerminalMode terminal_mode = TerminalMode::RecordComputed;
    // 0 means auto-select from available VRAM.
    int pair_chunk_size = 0;
    double vram_usage_fraction = 0.85;
    std::string evaluator_data_dir = "../external/CUDA-Poker-Calculator/src/resources";
};

struct GpuCfrStats {
    int iterations_run = 0;
    float last_root_value_p0 = 0.0f;
    std::size_t tensor_entries = 0;
    std::size_t state_bucket_entries = 0;
    std::size_t hand_pair_count = 0;
    std::size_t static_game_bytes = 0;
    std::size_t hand_data_bytes = 0;
    std::size_t terminal_data_bytes = 0;
    std::size_t cfr_state_bytes = 0;
    std::size_t work_buffer_bytes = 0;
};

// -----------------------------------------------------------------------------
// Host-side flattened public game representation
// -----------------------------------------------------------------------------

struct FlatPublicLevelEdges {
    std::vector<int> parent;
    std::vector<int> child;
    // For action edges:
    //   local action index at the parent action state.
    //
    // For chance edges:
    //   -1
    std::vector<int> local_action;
    // For chance edges:
    //   probability of public-card transition.
    //
    // For action edges:
    //   1.0
    std::vector<float> chance_prob;
    [[nodiscard]] int size() const {
        return static_cast<int>(parent.size());
    }
    [[nodiscard]] bool empty() const {
        return parent.empty();
    }
};

struct FlatPublicGame {
    int num_nodes = 0;
    int num_edges = 0;
    int num_actions = 0;
    int num_action_states = 0;
    int num_players = 2;
    int max_depth = 0;
    int root = 0;
    std::size_t tensor_entries = 0;
    std::size_t state_bucket_entries = 0;
    // ---------------------------------------------------------------------
    // Per-node arrays, length = num_nodes.
    // ---------------------------------------------------------------------
    std::vector<int> parent;
    std::vector<int> depth;
    // static_cast<int>(Player):
    //
    //   Chance   = -1
    //   P0       = 0
    //   P1       = 1
    //   Terminal = 2
    std::vector<int> player;
    // static_cast<int>(PublicNodeType)
    std::vector<int> node_type;
    // -1 for chance and terminal nodes.
    std::vector<int> action_state_index;
    // ---------------------------------------------------------------------
    // Per-action-state arrays, length = num_action_states.
    // ---------------------------------------------------------------------
    std::vector<int> action_state_node;
    // static_cast<int>(Player)
    std::vector<int> action_state_player;
    std::vector<int> action_state_bucket_count;
    std::vector<int> action_state_action_count;
    std::vector<int> action_state_first_action;
    // Tensor layout:
    //
    //   tensor_index =
    //       action_state_tensor_offset[state]
    //     + bucket * action_state_action_count[state]
    //     + local_action
    std::vector<std::uint64_t> action_state_tensor_offset;
    // State-bucket layout:
    //
    //   state_bucket_index =
    //       action_state_bucket_offset[state]
    //     + bucket
    std::vector<std::uint64_t> action_state_bucket_offset;
    // ---------------------------------------------------------------------
    // Edges grouped by parent depth.
    // ---------------------------------------------------------------------

    std::vector<FlatPublicLevelEdges> level_edges;

    // Convenience list containing only action edges.
    //
    // These are useful for regret/action-value kernels.
    std::vector<int> action_edge_parent;
    std::vector<int> action_edge_child;
    std::vector<int> action_edge_state;
    std::vector<int> action_edge_local_action;
};

// -----------------------------------------------------------------------------
// Host-side flattened hand data
// -----------------------------------------------------------------------------

struct FlatHandData {
    int p0_hand_count = 0;
    int p1_hand_count = 0;
    int hand_pair_count = 0;

    std::vector<unsigned char> p0_hand_card0;
    std::vector<unsigned char> p0_hand_card1;
    std::vector<unsigned char> p1_hand_card0;
    std::vector<unsigned char> p1_hand_card1;

    // HandPairTable, flat pair arrays.
    //
    // pair k:
    //   p0_index = p0_pair_index[k]
    //   p1_index = p1_pair_index[k]
    std::vector<int> p0_pair_index;
    std::vector<int> p1_pair_index;

    // Current exact-domain mode:
    //
    //   bucket == domain-local hand index
    //
    // Later, if HandDomain stores abstraction buckets, these arrays should
    // contain the bucket for each exact hand index.
    std::vector<int> p0_bucket_by_hand_index;
    std::vector<int> p1_bucket_by_hand_index;

    int p0_bucket_count = 0;
    int p1_bucket_count = 0;
};

// -----------------------------------------------------------------------------
// Host-side terminal value data
// -----------------------------------------------------------------------------

struct FlatTerminalData {
    std::vector<int> terminal_nodes;
    std::vector<int> terminal_index_by_node;
    // DeviceComputed mode:
    std::vector<int> terminal_type;
    std::vector<unsigned char> terminal_board_cards; // terminal_count * 5
    std::vector<int> pot;
    std::vector<int> p0_committed;
    // HostPrecomputed mode only:
    std::vector<float> terminal_value_p0;

    [[nodiscard]] int terminal_count() const {
        return static_cast<int>(terminal_nodes.size());
    }
};

// -----------------------------------------------------------------------------
// Flattening API
// -----------------------------------------------------------------------------

FlatPublicGame flatten_public_game_for_gpu(
    const Game& game,
    FlatTerminalData& flat_terminal_data
);

FlatHandData flatten_hand_data_for_gpu(
    const Game& game
);

void flatten_terminal_data_for_gpu(
    const Game& game,
    FlatTerminalData& flat,
    TerminalMode terminal_mode
);

// -----------------------------------------------------------------------------
// Device edge storage
// -----------------------------------------------------------------------------

struct DevicePublicLevelEdges {
    int* d_parent = nullptr;
    int* d_child = nullptr;

    // -1 for chance edges.
    int* d_local_action = nullptr;

    float* d_chance_prob = nullptr;

    int count = 0;
};

struct DevicePublicActionEdges {
    int* d_parent = nullptr;
    int* d_child = nullptr;

    int* d_action_state = nullptr;
    int* d_local_action = nullptr;

    int count = 0;
};

// -----------------------------------------------------------------------------
// Static device-side public game data
// -----------------------------------------------------------------------------

struct DevicePublicGameData {
    int num_nodes = 0;
    int num_edges = 0;
    int num_actions = 0;
    int num_action_states = 0;
    int num_players = 2;
    int max_depth = 0;
    int root = 0;

    std::size_t tensor_entries = 0;
    std::size_t state_bucket_entries = 0;

    // ---------------------------------------------------------------------
    // Per-node arrays.
    // ---------------------------------------------------------------------

    int* d_parent = nullptr;
    int* d_depth = nullptr;
    int* d_player = nullptr;
    int* d_node_type = nullptr;
    int* d_terminal_type = nullptr;
    int* d_action_state_index = nullptr;

    int* d_board_index = nullptr;
    int* d_pot = nullptr;
    int* d_p0_stack = nullptr;
    int* d_p1_stack = nullptr;

    // ---------------------------------------------------------------------
    // Per-action-state arrays.
    // ---------------------------------------------------------------------

    int* d_action_state_node = nullptr;
    int* d_action_state_player = nullptr;
    int* d_action_state_bucket_count = nullptr;
    int* d_action_state_action_count = nullptr;
    int* d_action_state_first_action = nullptr;

    std::uint64_t* d_action_state_tensor_offset = nullptr;
    std::uint64_t* d_action_state_bucket_offset = nullptr;

    // ---------------------------------------------------------------------
    // Edges.
    // ---------------------------------------------------------------------

    std::vector<DevicePublicLevelEdges> level_edges;

    DevicePublicActionEdges action_edges;
};

// -----------------------------------------------------------------------------
// Static device-side hand data
// -----------------------------------------------------------------------------

struct DeviceHandData {
    int p0_hand_count = 0;
    int p1_hand_count = 0;
    int hand_pair_count = 0;

    unsigned char* d_p0_hand_card0 = nullptr;
    unsigned char* d_p0_hand_card1 = nullptr;
    unsigned char* d_p1_hand_card0 = nullptr;
    unsigned char* d_p1_hand_card1 = nullptr;

    int* d_p0_pair_index = nullptr;
    int* d_p1_pair_index = nullptr;

    int* d_p0_bucket_by_hand_index = nullptr;
    int* d_p1_bucket_by_hand_index = nullptr;

    int p0_bucket_count = 0;
    int p1_bucket_count = 0;
};

// -----------------------------------------------------------------------------
// Static / semi-static device-side terminal data
// -----------------------------------------------------------------------------

struct DeviceTerminalData {
    int terminal_count = 0;
    int hand_pair_count = 0;

    int* d_terminal_nodes = nullptr;
    int* d_terminal_index_by_node = nullptr;

    // DeviceComputed mode:
    int* d_terminal_type = nullptr;
    unsigned char* d_terminal_board_cards = nullptr;
    int* d_pot = nullptr;
    int* d_p0_committed = nullptr;

    // HostPrecomputed mode:
    float* d_terminal_value_p0 = nullptr;
};

// -----------------------------------------------------------------------------
// Dynamic CFR device-side state
// -----------------------------------------------------------------------------

struct DevicePublicCfrState {
    // Length = tensor_entries.
    //
    // Indexed by:
    //
    //   tensor_offset[state] + bucket * action_count[state] + local_action
    float* d_sigma = nullptr;
    float* d_sigma_init = nullptr;
    float* d_regret_sum = nullptr;
    float* d_strategy_sum = nullptr;
    float* d_avg_strategy = nullptr;

    // Length = state_bucket_entries.
    //
    // Used as denominator for average strategy accumulation.
    float* d_strategy_weight_sum = nullptr;
};

// -----------------------------------------------------------------------------
// Per-iteration work buffers
// -----------------------------------------------------------------------------
struct DevicePublicWorkBuffers {
    // ---------------------------------------------------------------------
    // Pair-level node values.
    // ---------------------------------------------------------------------
    //
    // In exact public-tree CFR, terminal values depend on private hand pair.
    // The most direct layout is: node_pair_value[node_id * hand_pair_count + pair_id]
    // This can be large, but is the clearest validation layout.
    // Per-depth rolling buffers have been experimented with,
    // but increase computation speeds Quadratically, and as such have been ruled out
    //
    float* d_node_pair_value_p0 = nullptr;

    std::size_t node_pair_value_entries = 0;
    // Pair-level forward reaches.
    //
    // Layout:
    //
    //   node_pair_index = node_id * hand_pair_count + pair_id
    //
    // reach_p0:
    //   product of P0 strategy probabilities on the path
    //
    // reach_p1:
    //   product of P1 strategy probabilities on the path
    //
    // reach_chance:
    //   product of public chance probabilities on the path
    float* d_node_pair_reach_p0 = nullptr;
    float* d_node_pair_reach_p1 = nullptr;
    float* d_node_pair_reach_chance = nullptr;
    // State-bucket reaches.
    //
    // Layout:
    //
    //   state_bucket_index =
    //       action_state_bucket_offset[state] + bucket
    //
    // Used by regret update and average-strategy accumulation.
    float* d_state_bucket_own_reach = nullptr;
    // Accumulated across all chunks, applied once per iteration.
    float* d_regret_delta = nullptr;
    int pair_chunk_size = 0;
};

// -----------------------------------------------------------------------------
// Host/device ownership bundle
// -----------------------------------------------------------------------------
struct DeviceHandEvaluatorTables {
    short* d_binaries_by_id = nullptr;
    short* d_suitbit_by_id = nullptr;
    short* d_flush = nullptr;
    short* d_noflush7 = nullptr;
    unsigned char* d_suits = nullptr;
    int* d_dp = nullptr;
};

struct HostHandEvaluatorTables {
    std::vector<short> binaries_by_id;
    std::vector<short> suitbit_by_id;
    std::vector<short> flush;
    std::vector<short> noflush7;
    std::vector<unsigned char> suits;
    std::vector<int> dp;
};
struct GpuPublicState {
    FlatPublicGame flat;
    FlatHandData hands;
    FlatTerminalData terminals;

    DevicePublicGameData game;
    DeviceHandData hand_data;
    DeviceTerminalData terminal_data;

    DevicePublicCfrState cfr;
    DevicePublicWorkBuffers work;

    HostHandEvaluatorTables host_eval_tables;
    DeviceHandEvaluatorTables eval_tables;
};

// -----------------------------------------------------------------------------
// GPU public-tree CFR solver
// -----------------------------------------------------------------------------

class GpuCfrSolver {
public:
    void upload_hand_evaluator_tables();

    // Uses HostPrecomputed terminal mode only if terminal values are later
    // supplied through set_terminal_values().
    explicit GpuCfrSolver(const Game& game);

    GpuCfrSolver(
        const Game& game,
        GpuCfrConfig config
    );

    ~GpuCfrSolver();

    // Debug only.
    [[nodiscard]] std::vector<float> debug_action_value_p0() const;
    [[nodiscard]] std::vector<float> debug_state_bucket_cf_reach() const;
    [[nodiscard]] std::vector<float> debug_state_bucket_own_reach() const;
    [[nodiscard]] std::vector<float> debug_state_bucket_value_p0() const;
    [[nodiscard]] std::vector<float> debug_node_pair_value_p0() const;
    void debug_dump_action_value_launch_inputs() const;

    GpuCfrSolver(const GpuCfrSolver&) = delete;
    GpuCfrSolver& operator=(const GpuCfrSolver&) = delete;

    GpuCfrSolver(GpuCfrSolver&&) = delete;
    GpuCfrSolver& operator=(GpuCfrSolver&&) = delete;

    // Replaces / uploads terminal values after construction.
    //
    // This is useful if terminal values are expensive and generated by a
    // separate evaluator.
    void set_terminal_values(
        std::vector<float> terminal_value_p0
    );

    void run_iterations(int iterations);
    void run_one_iteration();

    // Flat tensor strategy:
    //
    //   action_state.tensor_offset
    // + bucket * action_state.action_count
    // + local_action
    [[nodiscard]] std::vector<float> current_strategy() const;

    [[nodiscard]] std::vector<float> average_strategy() const;

    [[nodiscard]] std::vector<float> regret_sum() const;

    [[nodiscard]] const GpuCfrStats& stats() const;

    [[nodiscard]] const FlatPublicGame& flat_game() const;

    [[nodiscard]] const FlatHandData& flat_hand_data() const;

    [[nodiscard]] const FlatTerminalData& flat_terminal_data() const;

private:
    const Game& game_;
    GpuCfrConfig config_;
    GpuCfrStats stats_;

    GpuPublicState gpu_;

    bool initialized_ = false;
    bool terminal_values_uploaded_ = false;

    // ---------------------------------------------------------------------
    // Initialization / cleanup
    // ---------------------------------------------------------------------

    void initialize();

    void release();

    void upload_static_game();
    void upload_hand_data();
    void upload_terminal_data();

    void allocate_cfr_state();
    void allocate_work_buffers();

    void initialize_strategy() const;

    // ---------------------------------------------------------------------
    // Iteration stages
    // ---------------------------------------------------------------------

    void run_terminal_evaluation_pass_for_chunk(int pair_start, int active_pair_count) const;

    void run_backward_value_pass_for_chunk(
        int pair_start,
        int active_pair_count
    ) const;

    void run_reach_pass_for_chunk(
        int pair_start,
        int active_pair_count
    ) const;

    void clear_iteration_accumulators() const;

    void clear_chunk_node_buffers(int active_pair_count) const;

    void run_action_value_pass_for_chunk(int pair_start, int active_pair_count) const;

    void accumulate_regret_deltas_for_chunk(int pair_start, int active_pair_count) const;

    void apply_regret_deltas() const;

    void run_average_strategy_accumulation_pass();

    void run_strategy_update_pass();

    // ---------------------------------------------------------------------
    // Copy helpers
    // ---------------------------------------------------------------------

    [[nodiscard]] std::vector<float> copy_tensor_from_device(
        const float* device_ptr,
        std::size_t count
    ) const;
};

// -----------------------------------------------------------------------------
// Device-side index formula documentation
// -----------------------------------------------------------------------------
//
// Kernels should use this exact indexing convention:
//
//   tensor_index =
//       d_action_state_tensor_offset[state]
//     + bucket * d_action_state_action_count[state]
//     + local_action
//
//   state_bucket_index =
//       d_action_state_bucket_offset[state]
//     + bucket
//
// In exact-domain mode:
//
//   bucket == domain-local hand index
//
// For a legal hand pair:
//
//   p0_bucket = d_p0_bucket_by_hand_index[d_p0_pair_index[pair]]
//   p1_bucket = d_p1_bucket_by_hand_index[d_p1_pair_index[pair]]
//
// This replaces the old q-index layout entirely.

} // namespace poker