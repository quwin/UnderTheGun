#pragma once

#include "game.hpp"

#include <cstdint>
#include <vector>

namespace poker {

struct CfrConfig {
    // Number of players in standard two-player Kuhn.
    int num_players = 2;

    // Whether to update both players in one traversal.
    // For a first implementation, keep this true.
    bool simultaneous_updates = true;

    // Whether to use CFR+ style regret clipping.
    // Start with false for vanilla CFR.
    bool use_cfr_plus = false;

    // Weight applied to average-strategy accumulation.
    // Vanilla CFR uses 1.0 each iteration.
    bool linear_averaging = false;
};

struct CfrStats {
    int iterations_run = 0;

    // Optional diagnostic values.
    // You can fill these later once exploitability is implemented.
    double last_root_value_p0 = 0.0;
};

class CpuCfrSolver {
public:
    explicit CpuCfrSolver(const Game& game);
    CpuCfrSolver(const Game& game, CfrConfig config);

    // Runs CFR for a fixed number of iterations.
    void run_iterations(int iterations);

    // Runs exactly one CFR iteration.
    void run_one_iteration();

    // Returns the current regret-matched strategy.
    std::vector<float> current_strategy() const;

    // Returns the average strategy accumulated over all iterations.
    std::vector<float> average_strategy() const;

    // Direct accessors useful for tests and debugging.
    const std::vector<float>& regret_sum() const;
    const std::vector<float>& strategy_sum() const;
    const CfrStats& stats() const;

private:
    const Game& game_;
    CfrConfig config_;
    CfrStats stats_;

    // One entry per infoset-action pair q.
    std::vector<float> regret_sum_;
    std::vector<float> strategy_sum_;
    std::vector<float> current_strategy_;

    // Main recursive CFR traversal.
    //
    // reach_p0 and reach_p1 are the probabilities that each player has
    // contributed to reaching this node under the current strategy.
    float cfr_traverse(
        int node_id,
        float reach_p0,
        float reach_p1
    );

    // Handles chance nodes by averaging over chance outcomes.
    float traverse_chance_node(
        int node_id,
        float reach_p0,
        float reach_p1
    );

    // Handles real-player decision nodes.
    float traverse_player_node(
        int node_id,
        float reach_p0,
        float reach_p1
    );

    // Converts regret_sum_ at one infoset into action probabilities.
    void compute_strategy_for_infoset(int infoset_id);

    // Adds the current strategy into strategy_sum_ using the player's reach.
    void accumulate_average_strategy(
        int infoset_id,
        float player_reach_weight
    );

    // Updates regret values for each action at an infoset.
    void update_regrets(
        int infoset_id,
        const std::vector<float>& action_values,
        float node_value,
        float counterfactual_reach
    );

    // Utility helpers.
    float terminal_value_p0(int node_id) const;
    float utility_for_player(float utility_p0, Player player) const;
};

} // namespace poker