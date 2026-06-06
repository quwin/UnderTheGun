#include "cfr_cpu.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace poker {
namespace {

constexpr double kEpsilon = 1e-12;

double uniform_probability(int count) {
    if (count <= 0) {
        throw std::invalid_argument("uniform_probability requires count > 0.");
    }

    return 1.0 / static_cast<double>(count);
}

bool is_finite_probability(double value) {
    return std::isfinite(value) && value >= 0.0 && value <= 1.0;
}

// Used by constructors that do not receive an explicit hand-pair weight provider.
const UniformHandPairWeightProvider& default_hand_pair_weights() {
    static const UniformHandPairWeightProvider provider{};
    return provider;
}

} // namespace

// -----------------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------------

CpuCfrSolver::CpuCfrSolver(
    const Game& game,
    const TerminalValueProvider& terminal_values
)
    : CpuCfrSolver(
          game,
          terminal_values,
          default_hand_pair_weights(),
          CfrConfig{}
      ) {}

CpuCfrSolver::CpuCfrSolver(
    const Game& game,
    const TerminalValueProvider& terminal_values,
    CfrConfig config
)
    : CpuCfrSolver(
          game,
          terminal_values,
          default_hand_pair_weights(),
          config
      ) {}

CpuCfrSolver::CpuCfrSolver(
    const Game& game,
    const TerminalValueProvider& terminal_values,
    const HandPairWeightProvider& hand_pair_weights,
    CfrConfig config
)
    : game_(game),
      terminal_values_(terminal_values),
      hand_pair_weights_(&hand_pair_weights),
      config_(config),
      stats_{},
      regret_sum_(game.cfr_tensor_entries(), 0.0f),
      strategy_sum_(game.cfr_tensor_entries(), 0.0f),
      current_strategy_(game.cfr_tensor_entries(), 0.0f),
      strategy_weight_sum_(game.state_bucket_entries(), 0.0f) {
    game_.validate();

    if (config_.num_players != 2 || game_.num_players != 2) {
        throw std::invalid_argument(
            "CpuCfrSolver currently expects a two-player zero-sum Game."
        );
    }

    if (game_.hand_pairs.empty()) {
        throw std::invalid_argument(
            "CpuCfrSolver requires a nonempty HandPairTable."
        );
    }

    if (game_.num_action_states() <= 0) {
        throw std::invalid_argument(
            "CpuCfrSolver requires at least one ActionState."
        );
    }

    if (game_.cfr_tensor_entries() == 0) {
        throw std::invalid_argument(
            "CpuCfrSolver requires nonzero CFR tensor entries."
        );
    }

    stats_.tensor_entries = game_.cfr_tensor_entries();
    stats_.state_bucket_entries = game_.state_bucket_entries();
    stats_.hand_pair_count = static_cast<std::size_t>(game_.hand_pairs.pair_count());

    initialize_uniform_strategy();
}

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

void CpuCfrSolver::run_iterations(int iterations) {
    if (iterations < 0) {
        throw std::invalid_argument("iterations must be nonnegative.");
    }

    for (int i = 0; i < iterations; ++i) {
        run_one_iteration();
    }
}

void CpuCfrSolver::run_one_iteration() {
    // Freeze sigma for this entire iteration. Regret updates performed during
    // traversal should not affect action probabilities until the next iteration.
    compute_all_current_strategies();

    const double weight_sum = normalized_hand_pair_weight_sum();

    double root_value_p0 = 0.0;

    for (int pair_id = 0;
         pair_id < game_.hand_pairs.pair_count();
         ++pair_id) {
        const double weight = hand_pair_weight(pair_id);

        if (weight == 0.0) {
            continue;
        }

        const double pair_probability = weight / weight_sum;

        const double pair_value_p0 =
            cfr_traverse_pair(
                game_.root,
                pair_id,
                1.0,
                1.0,
                1.0
            );

        root_value_p0 += pair_probability * pair_value_p0;
    }

    stats_.last_root_value_p0 = root_value_p0;
    ++stats_.iterations_run;

    // Prepare current_strategy_ for callers immediately after this iteration.
    compute_all_current_strategies();
}

StrategyTensor CpuCfrSolver::current_strategy() const {
    return current_strategy_;
}

StrategyTensor CpuCfrSolver::average_strategy() const {
    StrategyTensor average(game_.cfr_tensor_entries(), 0.0f);

    for (const ActionState& state : game_.action_states) {
        if (state.bucket_count <= 0 || state.action_count <= 0) {
            throw std::runtime_error("ActionState has invalid dimensions.");
        }

        for (int bucket = 0; bucket < state.bucket_count; ++bucket) {
            const std::size_t bucket_idx =
                state_bucket_index(state, bucket);

            const double normalizer =
                static_cast<double>(strategy_weight_sum_[bucket_idx]);

            if (normalizer > kEpsilon) {
                for (int a = 0; a < state.action_count; ++a) {
                    const std::size_t idx =
                        tensor_index(state, bucket, a);

                    average[idx] =
                        static_cast<float>(
                            static_cast<double>(strategy_sum_[idx]) /
                            normalizer
                        );
                }
            } else {
                const float p =
                    static_cast<float>(
                        uniform_probability(state.action_count)
                    );

                for (int a = 0; a < state.action_count; ++a) {
                    average[tensor_index(state, bucket, a)] = p;
                }
            }
        }
    }

    return average;
}

const std::vector<float>& CpuCfrSolver::regret_sum() const {
    return regret_sum_;
}

const std::vector<float>& CpuCfrSolver::strategy_sum() const {
    return strategy_sum_;
}

const std::vector<float>& CpuCfrSolver::strategy_weight_sum() const {
    return strategy_weight_sum_;
}

const CfrStats& CpuCfrSolver::stats() const {
    return stats_;
}

// -----------------------------------------------------------------------------
// Traversal
// -----------------------------------------------------------------------------

double CpuCfrSolver::cfr_traverse_pair(
    int node_id,
    int hand_pair_id,
    double reach_p0,
    double reach_p1,
    double reach_chance
) {
    if (node_id < 0 || node_id >= game_.num_nodes()) {
        throw std::invalid_argument("cfr_traverse_pair node_id out of range.");
    }

    if (hand_pair_id < 0 ||
        hand_pair_id >= game_.hand_pairs.pair_count()) {
        throw std::invalid_argument(
            "cfr_traverse_pair hand_pair_id out of range."
        );
    }

    const PublicNode& node = game_.node(node_id);

    switch (node.type) {
        case PublicNodeType::Terminal:
            return terminal_value_p0(node_id, hand_pair_id);

        case PublicNodeType::Chance:
            return traverse_chance_node(
                node_id,
                hand_pair_id,
                reach_p0,
                reach_p1,
                reach_chance
            );

        case PublicNodeType::Action:
            return traverse_player_node(
                node_id,
                hand_pair_id,
                reach_p0,
                reach_p1,
                reach_chance
            );
    }

    throw std::runtime_error("Unknown PublicNodeType.");
}

double CpuCfrSolver::traverse_chance_node(
    int node_id,
    int hand_pair_id,
    double reach_p0,
    double reach_p1,
    double reach_chance
) {
    const PublicNode& node = game_.node(node_id);

    if (node.type != PublicNodeType::Chance ||
        node.player != Player::Chance) {
        throw std::runtime_error(
            "traverse_chance_node received non-chance node."
        );
    }

    double value = 0.0;

    for (int local = 0; local < node.edge_count; ++local) {
        const NodeEdge& edge = outgoing_edge(node, local);

        if (!is_finite_probability(edge.chance_prob) ||
            edge.chance_prob <= 0.0f) {
            throw std::runtime_error(
                "Chance edge has invalid probability."
            );
        }

        value += static_cast<double>(edge.chance_prob) *
            cfr_traverse_pair(
                edge.child,
                hand_pair_id,
                reach_p0,
                reach_p1,
                reach_chance * static_cast<double>(edge.chance_prob)
            );
    }

    return value;
}

double CpuCfrSolver::traverse_player_node(
    int node_id,
    int hand_pair_id,
    double reach_p0,
    double reach_p1,
    double reach_chance
) {
    const PublicNode& node = game_.node(node_id);

    if (node.type != PublicNodeType::Action ||
        !is_real_player(node.player)) {
        throw std::runtime_error(
            "traverse_player_node received non-action node."
        );
    }

    const Player acting_player = node.player;
    const ActionState& state = action_state_for_node(node);

    if (state.action_count != node.edge_count) {
        throw std::runtime_error(
            "ActionState action_count must equal PublicNode edge_count."
        );
    }

    const int bucket =
        bucket_for_pair(acting_player, hand_pair_id);

    const double pair_probability =
        hand_pair_weight(hand_pair_id) /
        normalized_hand_pair_weight_sum();

    const double average_iteration_weight =
        config_.linear_averaging
            ? static_cast<double>(stats_.iterations_run + 1)
            : 1.0;

    const double own_reach =
        acting_player == Player::P0 ? reach_p0 : reach_p1;

    const double opponent_reach =
        acting_player == Player::P0 ? reach_p1 : reach_p0;

    // Average-strategy accumulation. This stores the current frozen sigma,
    // weighted by the acting player's reach. The pair probability keeps exact
    // hand-bucket accumulation range-aware.
    accumulate_average_strategy(
        state,
        bucket,
        pair_probability * average_iteration_weight * own_reach
    );

    std::vector<double> action_values_p0(
        static_cast<std::size_t>(state.action_count),
        0.0
    );

    double node_value_p0 = 0.0;

    for (int a = 0; a < state.action_count; ++a) {
        const std::size_t idx = tensor_index(state, bucket, a);
        const double action_probability =
            static_cast<double>(current_strategy_[idx]);

        if (!is_finite_probability(action_probability)) {
            throw std::runtime_error(
                "Current strategy contains invalid probability."
            );
        }

        const NodeEdge& edge = outgoing_edge(node, a);

        double next_reach_p0 = reach_p0;
        double next_reach_p1 = reach_p1;

        if (acting_player == Player::P0) {
            next_reach_p0 *= action_probability;
        } else {
            next_reach_p1 *= action_probability;
        }

        const double child_value_p0 =
            cfr_traverse_pair(
                edge.child,
                hand_pair_id,
                next_reach_p0,
                next_reach_p1,
                reach_chance
            );

        action_values_p0[static_cast<std::size_t>(a)] =
            child_value_p0;

        node_value_p0 += action_probability * child_value_p0;
    }

    const double counterfactual_reach =
        pair_probability * reach_chance * opponent_reach;

    if (config_.simultaneous_updates) {
        update_regrets(
            state,
            bucket,
            action_values_p0,
            node_value_p0,
            counterfactual_reach,
            acting_player
        );
    }

    return node_value_p0;
}

double CpuCfrSolver::terminal_value_p0(
    int node_id,
    int hand_pair_id
) const {
    const PublicNode& node = game_.node(node_id);

    if (node.type != PublicNodeType::Terminal) {
        throw std::invalid_argument(
            "terminal_value_p0 requires a terminal node."
        );
    }

    const double value =
        terminal_values_.utility_p0(
            game_,
            node_id,
            hand_pair_id
        );

    if (!std::isfinite(value)) {
        throw std::runtime_error(
            "TerminalValueProvider returned non-finite utility."
        );
    }

    return value;
}

// -----------------------------------------------------------------------------
// Strategy / regret helpers
// -----------------------------------------------------------------------------

void CpuCfrSolver::initialize_uniform_strategy() {
    for (const ActionState& state : game_.action_states) {
        if (state.bucket_count <= 0 || state.action_count <= 0) {
            throw std::runtime_error(
                "Cannot initialize invalid ActionState."
            );
        }

        const float p =
            static_cast<float>(
                uniform_probability(state.action_count)
            );

        for (int bucket = 0; bucket < state.bucket_count; ++bucket) {
            for (int a = 0; a < state.action_count; ++a) {
                current_strategy_[tensor_index(state, bucket, a)] = p;
            }
        }
    }
}

void CpuCfrSolver::compute_strategy_for_action_state_bucket(
    const ActionState& state,
    int bucket
) {
    if (bucket < 0 || bucket >= state.bucket_count) {
        throw std::invalid_argument(
            "compute_strategy_for_action_state_bucket bucket out of range."
        );
    }

    double positive_sum = 0.0;

    for (int a = 0; a < state.action_count; ++a) {
        const std::size_t idx = tensor_index(state, bucket, a);
        const double r = static_cast<double>(regret_sum_[idx]);

        if (r > 0.0) {
            positive_sum += r;
        }
    }

    if (positive_sum > kEpsilon) {
        for (int a = 0; a < state.action_count; ++a) {
            const std::size_t idx = tensor_index(state, bucket, a);
            const double r =
                std::max(0.0, static_cast<double>(regret_sum_[idx]));

            current_strategy_[idx] =
                static_cast<float>(r / positive_sum);
        }
    } else {
        const float p =
            static_cast<float>(
                uniform_probability(state.action_count)
            );

        for (int a = 0; a < state.action_count; ++a) {
            current_strategy_[tensor_index(state, bucket, a)] = p;
        }
    }
}

void CpuCfrSolver::compute_all_current_strategies() {
    for (const ActionState& state : game_.action_states) {
        for (int bucket = 0; bucket < state.bucket_count; ++bucket) {
            compute_strategy_for_action_state_bucket(state, bucket);
        }
    }
}

void CpuCfrSolver::accumulate_average_strategy(
    const ActionState& state,
    int bucket,
    double weight
) {
    if (weight <= 0.0) {
        return;
    }

    if (!std::isfinite(weight)) {
        throw std::runtime_error(
            "Average-strategy accumulation weight is non-finite."
        );
    }

    const std::size_t bucket_idx =
        state_bucket_index(state, bucket);

    strategy_weight_sum_[bucket_idx] +=
        static_cast<float>(weight);

    for (int a = 0; a < state.action_count; ++a) {
        const std::size_t idx = tensor_index(state, bucket, a);

        strategy_sum_[idx] +=
            static_cast<float>(
                weight * static_cast<double>(current_strategy_[idx])
            );
    }
}

void CpuCfrSolver::update_regrets(
    const ActionState& state,
    int bucket,
    const std::vector<double>& action_values_p0,
    double node_value_p0,
    double counterfactual_reach,
    Player acting_player
) {
    if (action_values_p0.size() !=
        static_cast<std::size_t>(state.action_count)) {
        throw std::invalid_argument(
            "update_regrets action_values_p0 size mismatch."
        );
    }

    if (counterfactual_reach <= 0.0) {
        return;
    }

    if (!std::isfinite(counterfactual_reach)) {
        throw std::runtime_error(
            "Counterfactual reach is non-finite."
        );
    }

    const double node_value_for_actor =
        utility_for_player(node_value_p0, acting_player);

    for (int a = 0; a < state.action_count; ++a) {
        const double action_value_for_actor =
            utility_for_player(
                action_values_p0[static_cast<std::size_t>(a)],
                acting_player
            );

        const double instantaneous_regret =
            counterfactual_reach *
            (action_value_for_actor - node_value_for_actor);

        const std::size_t idx = tensor_index(state, bucket, a);

        double updated =
            static_cast<double>(regret_sum_[idx]) +
            instantaneous_regret;

        if (config_.use_cfr_plus) {
            updated = std::max(0.0, updated);
        }

        if (!std::isfinite(updated)) {
            throw std::runtime_error(
                "Regret update produced non-finite value."
            );
        }

        regret_sum_[idx] = static_cast<float>(updated);
    }
}

// -----------------------------------------------------------------------------
// Indexing helpers
// -----------------------------------------------------------------------------

const ActionState& CpuCfrSolver::action_state_for_node(
    const PublicNode& node
) const {
    if (node.action_state_index < 0 ||
        node.action_state_index >= game_.num_action_states()) {
        throw std::runtime_error(
            "PublicNode has invalid action_state_index."
        );
    }

    const ActionState& state =
        game_.action_state(node.action_state_index);

    if (state.node < 0 || state.node >= game_.num_nodes()) {
        throw std::runtime_error(
            "ActionState node is out of range."
        );
    }

    if (state.action_count != node.edge_count) {
        throw std::runtime_error(
            "ActionState action_count does not match node edge_count."
        );
    }

    return state;
}

int CpuCfrSolver::bucket_for_pair(
    Player player,
    int hand_pair_id
) const {
    if (hand_pair_id < 0 ||
        hand_pair_id >= game_.hand_pairs.pair_count()) {
        throw std::invalid_argument(
            "bucket_for_pair hand_pair_id out of range."
        );
    }

    switch (player) {
        case Player::P0: {
            const int bucket =
                game_.hand_pairs.p0_index[
                    static_cast<std::size_t>(hand_pair_id)
                ];

            if (bucket < 0 ||
                bucket >= game_.p0_hands.hand_count()) {
                throw std::runtime_error(
                    "P0 hand-pair bucket is out of range."
                );
            }

            return bucket;
        }

        case Player::P1: {
            const int bucket =
                game_.hand_pairs.p1_index[
                    static_cast<std::size_t>(hand_pair_id)
                ];

            if (bucket < 0 ||
                bucket >= game_.p1_hands.hand_count()) {
                throw std::runtime_error(
                    "P1 hand-pair bucket is out of range."
                );
            }

            return bucket;
        }

        default:
            throw std::invalid_argument(
                "bucket_for_pair requires P0 or P1."
            );
    }
}

std::size_t CpuCfrSolver::tensor_index(
    const ActionState& state,
    int bucket,
    int local_action
) const {
    if (bucket < 0 || bucket >= state.bucket_count) {
        throw std::invalid_argument("tensor_index bucket out of range.");
    }

    const std::size_t idx =
        state.tensor_index(bucket, local_action);

    if (idx >= current_strategy_.size()) {
        throw std::runtime_error("tensor_index exceeds CFR tensor size.");
    }

    return idx;
}

std::size_t CpuCfrSolver::state_bucket_index(
    const ActionState& state,
    int bucket
) const {
    if (bucket < 0 || bucket >= state.bucket_count) {
        throw std::invalid_argument(
            "state_bucket_index bucket out of range."
        );
    }

    const std::size_t idx =
        static_cast<std::size_t>(state.state_bucket_offset) +
        static_cast<std::size_t>(bucket);

    if (idx >= strategy_weight_sum_.size()) {
        throw std::runtime_error(
            "state_bucket_index exceeds state-bucket array size."
        );
    }

    return idx;
}

int CpuCfrSolver::edge_id(
    const PublicNode& node,
    int local_action
) const {
    if (local_action < 0 || local_action >= node.edge_count) {
        throw std::invalid_argument("edge_id local_action out of range.");
    }

    const int id = node.first_edge + local_action;

    if (id < 0 || id >= game_.num_edges()) {
        throw std::runtime_error("edge_id out of Game::edges range.");
    }

    return id;
}

const NodeEdge& CpuCfrSolver::outgoing_edge(
    const PublicNode& node,
    int local_action
) const {
    return game_.edge(edge_id(node, local_action));
}

double CpuCfrSolver::hand_pair_weight(
    int hand_pair_id
) const {
    if (hand_pair_weights_ == nullptr) {
        throw std::logic_error("CpuCfrSolver has no hand-pair weight provider.");
    }

    const double weight =
        hand_pair_weights_->weight(game_, hand_pair_id);

    if (!std::isfinite(weight) || weight < 0.0) {
        throw std::runtime_error(
            "HandPairWeightProvider returned invalid weight."
        );
    }

    return weight;
}

double CpuCfrSolver::normalized_hand_pair_weight_sum() const {
    double total = 0.0;

    for (int pair_id = 0;
         pair_id < game_.hand_pairs.pair_count();
         ++pair_id) {
        total += hand_pair_weight(pair_id);
    }

    if (!(total > 0.0) || !std::isfinite(total)) {
        throw std::runtime_error(
            "Total hand-pair weight must be positive and finite."
        );
    }

    return total;
}

bool CpuCfrSolver::is_real_player(Player player) {
    return player == Player::P0 || player == Player::P1;
}

double CpuCfrSolver::utility_for_player(
    double utility_p0,
    Player player
) {
    switch (player) {
        case Player::P0:
            return utility_p0;

        case Player::P1:
            return -utility_p0;

        default:
            throw std::invalid_argument(
                "utility_for_player requires P0 or P1."
            );
    }
}

} // namespace poker