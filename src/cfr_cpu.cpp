#include "cfr_cpu.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace poker {

namespace {

constexpr float kEpsilon = 1e-12f;

float safe_uniform_probability(int count) {
    assert(count > 0);
    return 1.0f / static_cast<float>(count);
}

} // namespace

CpuCfrSolver::CpuCfrSolver(const Game& game)
    : CpuCfrSolver(game, CfrConfig{}) {}

CpuCfrSolver::CpuCfrSolver(const Game& game, CfrConfig config)
    : game_(game),
      config_(config),
      stats_{},
      regret_sum_(game.num_q(), 0.0f),
      strategy_sum_(game.num_q(), 0.0f),
      current_strategy_(game.num_q(), 0.0f) {
    for (const InfoSet& infoset : game_.infosets) {
        compute_strategy_for_infoset(infoset.id);
    }
}

void CpuCfrSolver::run_iterations(int iterations) {
    if (iterations < 0) {
        throw std::invalid_argument("iterations must be nonnegative");
    }

    for (int i = 0; i < iterations; ++i) {
        run_one_iteration();
    }
}

void CpuCfrSolver::run_one_iteration() {
    const float root_value = cfr_traverse(
        game_.root,
        1.0f,
        1.0f
    );

    stats_.last_root_value_p0 = root_value;
    ++stats_.iterations_run;
}

std::vector<float> CpuCfrSolver::current_strategy() const {
    return current_strategy_;
}

std::vector<float> CpuCfrSolver::average_strategy() const {
    std::vector<float> avg(game_.num_q(), 0.0f);

    for (const InfoSet& infoset : game_.infosets) {
        float normalizer = 0.0f;

        for (int q : infoset.q_indices) {
            normalizer += strategy_sum_[q];
        }

        if (normalizer > kEpsilon) {
            for (int q : infoset.q_indices) {
                avg[q] = strategy_sum_[q] / normalizer;
            }
        } else {
            const float uniform =
                safe_uniform_probability(static_cast<int>(infoset.q_indices.size()));

            for (int q : infoset.q_indices) {
                avg[q] = uniform;
            }
        }
    }

    return avg;
}

const std::vector<float>& CpuCfrSolver::regret_sum() const {
    return regret_sum_;
}

const std::vector<float>& CpuCfrSolver::strategy_sum() const {
    return strategy_sum_;
}

const CfrStats& CpuCfrSolver::stats() const {
    return stats_;
}

float CpuCfrSolver::cfr_traverse(
    int node_id,
    float reach_p0,
    float reach_p1
) {
    const Node& node = game_.node(node_id);

    if (node.terminal) {
        return terminal_value_p0(node_id);
    }

    if (node.player == Player::Chance) {
        return traverse_chance_node(node_id, reach_p0, reach_p1);
    }

    if (node.player == Player::P0 || node.player == Player::P1) {
        return traverse_player_node(node_id, reach_p0, reach_p1);
    }

    throw std::runtime_error("Invalid non-terminal node player.");
}

float CpuCfrSolver::traverse_chance_node(
    int node_id,
    float reach_p0,
    float reach_p1
) {
    const Node& node = game_.node(node_id);

    float value = 0.0f;

    for (int child_id : node.children) {
        const Node& child = game_.node(child_id);

        value += child.chance_prob *
            cfr_traverse(child_id, reach_p0, reach_p1);
    }

    return value;
}

float CpuCfrSolver::traverse_player_node(
    int node_id,
    float reach_p0,
    float reach_p1
) {
    const Node& node = game_.node(node_id);

    assert(node.infoset >= 0);
    assert(node.player == Player::P0 || node.player == Player::P1);

    const InfoSet& infoset = game_.infoset(node.infoset);

    compute_strategy_for_infoset(infoset.id);

    const int num_actions = static_cast<int>(infoset.actions.size());

    if (static_cast<int>(node.children.size()) != num_actions) {
        throw std::runtime_error("Node children and infoset actions mismatch.");
    }

    std::vector<float> action_values(num_actions, 0.0f);

    float node_value_p0 = 0.0f;

    for (int a = 0; a < num_actions; ++a) {
        const int q = infoset.q_indices[a];
        const int child_id = node.children[a];
        const float action_prob = current_strategy_[q];

        float next_reach_p0 = reach_p0;
        float next_reach_p1 = reach_p1;

        if (node.player == Player::P0) {
            next_reach_p0 *= action_prob;
        } else {
            next_reach_p1 *= action_prob;
        }

        action_values[a] = cfr_traverse(
            child_id,
            next_reach_p0,
            next_reach_p1
        );

        node_value_p0 += action_prob * action_values[a];
    }

    const float player_reach_weight =
        (node.player == Player::P0) ? reach_p0 : reach_p1;

    accumulate_average_strategy(
        infoset.id,
        player_reach_weight
    );

    const float counterfactual_reach =
        (node.player == Player::P0) ? reach_p1 : reach_p0;

    const float node_value_for_player =
        utility_for_player(node_value_p0, node.player);

    std::vector<float> action_values_for_player(num_actions, 0.0f);

    for (int a = 0; a < num_actions; ++a) {
        action_values_for_player[a] =
            utility_for_player(action_values[a], node.player);
    }

    update_regrets(
        infoset.id,
        action_values_for_player,
        node_value_for_player,
        counterfactual_reach
    );

    return node_value_p0;
}

void CpuCfrSolver::compute_strategy_for_infoset(int infoset_id) {
    const InfoSet& infoset = game_.infoset(infoset_id);

    float positive_regret_sum = 0.0f;

    for (int q : infoset.q_indices) {
        positive_regret_sum += std::max(regret_sum_[q], 0.0f);
    }

    if (positive_regret_sum > kEpsilon) {
        for (int q : infoset.q_indices) {
            current_strategy_[q] =
                std::max(regret_sum_[q], 0.0f) / positive_regret_sum;
        }
    } else {
        const float uniform =
            safe_uniform_probability(static_cast<int>(infoset.q_indices.size()));

        for (int q : infoset.q_indices) {
            current_strategy_[q] = uniform;
        }
    }
}

void CpuCfrSolver::accumulate_average_strategy(
    int infoset_id,
    float player_reach_weight
) {
    const InfoSet& infoset = game_.infoset(infoset_id);

    float iteration_weight = 1.0f;

    if (config_.linear_averaging) {
        iteration_weight = static_cast<float>(stats_.iterations_run + 1);
    }

    const float weight = iteration_weight * player_reach_weight;

    for (int q : infoset.q_indices) {
        strategy_sum_[q] += weight * current_strategy_[q];
    }
}

void CpuCfrSolver::update_regrets(
    int infoset_id,
    const std::vector<float>& action_values,
    float node_value,
    float counterfactual_reach
) {
    const InfoSet& infoset = game_.infoset(infoset_id);

    if (action_values.size() != infoset.q_indices.size()) {
        throw std::runtime_error("Action value count mismatch.");
    }

    for (std::size_t a = 0; a < infoset.q_indices.size(); ++a) {
        const int q = infoset.q_indices[a];

        const float instantaneous_regret =
            counterfactual_reach * (action_values[a] - node_value);

        if (config_.use_cfr_plus) {
            regret_sum_[q] = std::max(
                0.0f,
                regret_sum_[q] + instantaneous_regret
            );
        } else {
            regret_sum_[q] += instantaneous_regret;
        }
    }
}

float CpuCfrSolver::terminal_value_p0(int node_id) const {
    const Node& node = game_.node(node_id);

    assert(node.terminal);

    return node.utility_p0;
}

float CpuCfrSolver::utility_for_player(
    float utility_p0,
    Player player
) const {
    if (player == Player::P0) {
        return utility_p0;
    }

    if (player == Player::P1) {
        return -utility_p0;
    }

    throw std::runtime_error("Cannot get utility for non-real player.");
}

} // namespace poker