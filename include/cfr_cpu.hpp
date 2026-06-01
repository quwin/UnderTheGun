#pragma once

#include "game.hpp"
#include "exploitability.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace poker {

// -----------------------------------------------------------------------------
// CPU CFR configuration / stats
// -----------------------------------------------------------------------------

struct CfrConfig {
    int num_players = 2;

    // CFR+ clips cumulative regret at zero after each regret update.
    bool use_cfr_plus = false;

    // If false:
    //   average-strategy weight = 1
    //
    // If true:
    //   average-strategy weight = iteration number
    //
    // This is the same high-level option exposed in your GPU config.
    bool linear_averaging = false;

    // If true, update both players' regrets in the same public-tree traversal.
    //
    // For your current two-player zero-sum public-tree solver, keep this true.
    bool simultaneous_updates = true;
};

struct CfrStats {
    int iterations_run = 0;

    // Expected root value from P0 perspective under the current strategy.
    double last_root_value_p0 = 0.0;

    std::size_t tensor_entries = 0;
    std::size_t state_bucket_entries = 0;
    std::size_t hand_pair_count = 0;
};

// -----------------------------------------------------------------------------
// CPU CFR solver for public-tree / hand-aware Game
// -----------------------------------------------------------------------------
//
// This solver is for the new Game representation:
//
//   Game::nodes          compact public tree
//   Game::edges          contiguous child ranges
//   Game::action_states  hand-aware decision blocks
//   Game::p0_hands       legal P0 hand domain
//   Game::p1_hands       legal P1 hand domain
//   Game::hand_pairs     legal non-overlapping P0/P1 hand pairs
//
// Strategy/regret tensors use:
//
//   index =
//       action_state.tensor_offset
//     + hand_bucket * action_state.action_count
//     + local_action
//
// In your current exact-domain mode:
//
//   P0 bucket = game.hand_pairs.p0_index[pair_id]
//   P1 bucket = game.hand_pairs.p1_index[pair_id]
//
// Terminal values are injected through TerminalValueProvider because the compact
// PublicNode does not contain full PrivateState/PublicState terminal payloads.

class CpuCfrSolver {
public:
    CpuCfrSolver(
        const Game& game,
        const TerminalValueProvider& terminal_values
    );

    CpuCfrSolver(
        const Game& game,
        const TerminalValueProvider& terminal_values,
        CfrConfig config
    );

    CpuCfrSolver(
        const Game& game,
        const TerminalValueProvider& terminal_values,
        const HandPairWeightProvider& hand_pair_weights,
        CfrConfig config = CfrConfig{}
    );

    // Runs CFR for a fixed number of iterations.
    void run_iterations(int iterations);

    // Runs exactly one CFR iteration.
    void run_one_iteration();

    // Current regret-matched strategy tensor.
    [[nodiscard]] StrategyTensor current_strategy() const;

    // Normalized average strategy tensor.
    [[nodiscard]] StrategyTensor average_strategy() const;

    // Direct accessors for debugging/tests.
    [[nodiscard]] const std::vector<float>& regret_sum() const;
    [[nodiscard]] const std::vector<float>& strategy_sum() const;
    [[nodiscard]] const std::vector<float>& strategy_weight_sum() const;
    [[nodiscard]] const CfrStats& stats() const;

private:
    const Game& game_;
    const TerminalValueProvider& terminal_values_;
    const HandPairWeightProvider* hand_pair_weights_ = nullptr;

    CfrConfig config_;
    CfrStats stats_;

    // One entry per action-state/bucket/action.
    // Size = game_.cfr_tensor_entries().
    std::vector<float> regret_sum_;
    std::vector<float> strategy_sum_;
    std::vector<float> current_strategy_;

    // One entry per action-state/bucket.
    // Size = game_.state_bucket_entries().
    //
    // Used to normalize average_strategy().
    std::vector<float> strategy_weight_sum_;

    // ---------------------------------------------------------------------
    // Traversal
    // ---------------------------------------------------------------------

    // Traverses the public tree for one exact legal hand pair.
    //
    // reach_p0:
    //   P0's contribution to reaching this node.
    //
    // reach_p1:
    //   P1's contribution to reaching this node.
    //
    // reach_chance:
    //   Public chance contribution to reaching this node.
    //
    // Return:
    //   expected utility from P0 perspective.
    double cfr_traverse_pair(
        int node_id,
        int hand_pair_id,
        double reach_p0,
        double reach_p1,
        double reach_chance
    );

    double traverse_chance_node(
        int node_id,
        int hand_pair_id,
        double reach_p0,
        double reach_p1,
        double reach_chance
    );

    double traverse_player_node(
        int node_id,
        int hand_pair_id,
        double reach_p0,
        double reach_p1,
        double reach_chance
    );

    double terminal_value_p0(
        int node_id,
        int hand_pair_id
    ) const;

    // ---------------------------------------------------------------------
    // Strategy / regret helpers
    // ---------------------------------------------------------------------

    void initialize_uniform_strategy();

    void compute_strategy_for_action_state_bucket(
        const ActionState& action_state,
        int bucket
    );

    void compute_all_current_strategies();

    void accumulate_average_strategy(
        const ActionState& action_state,
        int bucket,
        double weight
    );

    void update_regrets(
        const ActionState& action_state,
        int bucket,
        const std::vector<double>& action_values_p0,
        double node_value_p0,
        double counterfactual_reach,
        Player acting_player
    );

    // ---------------------------------------------------------------------
    // Indexing helpers
    // ---------------------------------------------------------------------

    [[nodiscard]] const ActionState& action_state_for_node(
        const PublicNode& node
    ) const;

    [[nodiscard]] int bucket_for_pair(
        Player player,
        int hand_pair_id
    ) const;

    [[nodiscard]] std::size_t tensor_index(
        const ActionState& action_state,
        int bucket,
        int local_action
    ) const;

    [[nodiscard]] std::size_t state_bucket_index(
        const ActionState& action_state,
        int bucket
    ) const;

    [[nodiscard]] int edge_id(
        const PublicNode& node,
        int local_action
    ) const;

    [[nodiscard]] const NodeEdge& outgoing_edge(
        const PublicNode& node,
        int local_action
    ) const;

    [[nodiscard]] double hand_pair_weight(
        int hand_pair_id
    ) const;

    [[nodiscard]] double normalized_hand_pair_weight_sum() const;

    [[nodiscard]] static bool is_real_player(
        Player player
    );

    [[nodiscard]] static double utility_for_player(
        double utility_p0,
        Player player
    );
};

} // namespace poker