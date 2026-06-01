#include "game.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace poker {
namespace {
    template <typename T>
std::size_t vector_bytes_size(const std::vector<T>& v) {
    return sizeof(T) * v.size();
}

template <typename T>
std::size_t vector_bytes_capacity(const std::vector<T>& v) {
    return sizeof(T) * v.capacity();
}

double mib(std::size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

std::size_t action_state_tensor_size(const ActionState& state) {
    return static_cast<std::size_t>(state.bucket_count) *
           static_cast<std::size_t>(state.action_count);
}

std::size_t action_state_bucket_size(const ActionState& state) {
    return static_cast<std::size_t>(state.bucket_count);
}

std::uint64_t checked_size_to_u64(std::size_t value, const char* name) {
    constexpr std::size_t max_u64 =
        static_cast<std::size_t>(~std::uint64_t{0});

    if (value > max_u64) {
        throw std::overflow_error(
            std::string(name) + " does not fit in std::uint64_t."
        );
    }

    return static_cast<std::uint64_t>(value);
}

} // namespace

// -----------------------------------------------------------------------------
// Public tree construction
// -----------------------------------------------------------------------------

int Game::add_node(PublicNode node) {
    if (node.parent < -1) {
        throw std::invalid_argument("PublicNode parent cannot be less than -1.");
    }
    if (node.first_edge != 0 || node.edge_count != 0) {
        throw std::invalid_argument(
            "add_node expects a node with an empty edge range."
        );
    }
    if (node.action_state_index != -1) {
        throw std::invalid_argument(
            "add_node expects action_state_index to be -1."
        );
    }
    if (node.parent >= static_cast<int>(nodes.size())) {
        throw std::invalid_argument("PublicNode parent is out of range.");
    }
    if (node.parent >= 0) {
        node.depth = nodes[static_cast<std::size_t>(node.parent)].depth + 1;
    } else {
        node.depth = 0;
    }
    const int node_id = static_cast<int>(nodes.size());
    nodes.push_back(node);
    max_depth = std::max(max_depth, node.depth);

    if (root < 0) {
        root = node_id;
    }
    return node_id;
}

int Game::add_action(const GameAction& action) {
    const int id = static_cast<int>(actions.size());
    actions.push_back(action);
    return id;
}

void Game::attach_edge(
    int parent,
    NodeEdge edge
) {
    if (parent < 0 || parent >= num_nodes()) {
        throw std::invalid_argument("attach_edge parent is out of range.");
    }

    if (edge.child < 0 || edge.child >= num_nodes()) {
        throw std::invalid_argument("attach_edge child is out of range.");
    }

    if (!(edge.chance_prob > 0.0f) || edge.chance_prob > 1.0f) {
        throw std::invalid_argument(
            "attach_edge chance_prob must be in (0, 1]."
        );
    }

    PublicNode& parent_node = node(parent);
    PublicNode& child_node = node(edge.child);

    if (child_node.parent == -1) {
        child_node.parent = parent;
        child_node.depth = parent_node.depth + 1;
        max_depth = std::max(max_depth, child_node.depth);
    } else if (child_node.parent != parent) {
        throw std::invalid_argument(
            "attach_edge child already has a different parent."
        );
    }

    if (child_node.depth != parent_node.depth + 1) {
        throw std::invalid_argument(
            "attach_edge child depth must equal parent depth + 1."
        );
    }

    // This representation requires each node's outgoing edges to be contiguous.
    //
    // Therefore, the builder must add all outgoing edges for a parent before it
    // starts appending edges for another parent. If your recursive builder adds
    // one child edge, fully expands that child, and then returns to add another
    // sibling edge, this check will correctly fail.
    if (parent_node.edge_count == 0) {
        parent_node.first_edge = static_cast<int>(edges.size());
    } else {
        const int expected_next_edge =
            parent_node.first_edge + parent_node.edge_count;

        if (expected_next_edge != static_cast<int>(edges.size())) {
            throw std::logic_error(
                "Outgoing edges for a PublicNode must be appended contiguously."
            );
        }
    }

    edges.push_back(edge);
    ++parent_node.edge_count;
}

void Game::add_action_child(
    int parent,
    int child,
    int action_index
) {
    if (parent < 0 || parent >= num_nodes()) {
        throw std::invalid_argument("add_action_child parent is out of range.");
    }

    if (child < 0 || child >= num_nodes()) {
        throw std::invalid_argument("add_action_child child is out of range.");
    }

    if (action_index < 0 || action_index >= num_actions()) {
        throw std::invalid_argument("add_action_child action_index is out of range.");
    }

    PublicNode& parent_node = node(parent);

    if (parent_node.type != PublicNodeType::Action) {
        throw std::invalid_argument(
            "add_action_child requires an Action parent node."
        );
    }
    NodeEdge edge;
    edge.child = child;
    edge.local_action = parent_node.edge_count;
    edge.action_index = action_index;
    edge.public_card = -1;
    edge.chance_prob = 1.0f;

    attach_edge(parent, edge);
}

void Game::add_chance_child(
    int parent,
    int child,
    int public_card,
    float probability
) {
    if (parent < 0 || parent >= num_nodes()) {
        throw std::invalid_argument("add_chance_child parent is out of range.");
    }

    if (child < 0 || child >= num_nodes()) {
        throw std::invalid_argument("add_chance_child child is out of range.");
    }

    if (public_card < 0 || public_card >= kNumCards) {
        throw std::invalid_argument("add_chance_child public_card is out of range.");
    }

    if (!(probability > 0.0f) || probability > 1.0f) {
        throw std::invalid_argument(
            "add_chance_child probability must be in (0, 1]."
        );
    }

    PublicNode& parent_node = node(parent);

    if (parent_node.type != PublicNodeType::Chance) {
        throw std::invalid_argument(
            "add_chance_child requires a Chance parent node."
        );
    }

    if (parent_node.player != Player::Chance) {
        throw std::invalid_argument(
            "Chance parent must have player == Player::Chance."
        );
    }

    NodeEdge edge;
    edge.child = child;
    edge.local_action = -1;
    edge.action_index = -1;
    edge.public_card = public_card;
    edge.chance_prob = probability;

    attach_edge(parent, edge);
}

// -----------------------------------------------------------------------------
// Hand-domain setup
// -----------------------------------------------------------------------------

void Game::set_hand_domains(
    HandDomain p0,
    HandDomain p1,
    HandPairTable pairs
) {
    if (p0.empty()) {
        throw std::invalid_argument("P0 HandDomain cannot be empty.");
    }

    if (p1.empty()) {
        throw std::invalid_argument("P1 HandDomain cannot be empty.");
    }

    if (pairs.p0_index.size() != pairs.p1_index.size()) {
        throw std::invalid_argument(
            "HandPairTable p0_index and p1_index size mismatch."
        );
    }

    if (pairs.empty()) {
        throw std::invalid_argument("HandPairTable cannot be empty.");
    }

    for (HandId hand : p0.hands) {
        validate_hand_id(hand);
    }

    for (HandId hand : p1.hands) {
        validate_hand_id(hand);
    }

    for (int pair_id = 0; pair_id < pairs.pair_count(); ++pair_id) {
        const int p0_index = pairs.p0_index[static_cast<std::size_t>(pair_id)];
        const int p1_index = pairs.p1_index[static_cast<std::size_t>(pair_id)];

        if (p0_index < 0 || p0_index >= p0.hand_count()) {
            throw std::invalid_argument(
                "HandPairTable contains out-of-range P0 hand index."
            );
        }

        if (p1_index < 0 || p1_index >= p1.hand_count()) {
            throw std::invalid_argument(
                "HandPairTable contains out-of-range P1 hand index."
            );
        }

        const HandId p0_hand = p0.hands[static_cast<std::size_t>(p0_index)];
        const HandId p1_hand = p1.hands[static_cast<std::size_t>(p1_index)];

        if (hands_overlap(p0_hand, p1_hand)) {
            throw std::invalid_argument(
                "HandPairTable contains overlapping private hands."
            );
        }
    }

    p0_hands = std::move(p0);
    p1_hands = std::move(p1);
    hand_pairs = std::move(pairs);
}

int Game::bucket_count(Player player) const {
    return hand_domain(player).hand_count();
}

// -----------------------------------------------------------------------------
// Action-state registration
// -----------------------------------------------------------------------------

int Game::register_action_state(
    int node_id,
    Player player
) {
    if (node_id < 0 || node_id >= num_nodes()) {
        throw std::invalid_argument("register_action_state node_id is out of range.");
    }
    PublicNode& n = node(node_id);

    if (n.type != PublicNodeType::Action) {
        throw std::invalid_argument(
            "register_action_state requires an Action node."
        );
    }

    if (n.player != player) {
        throw std::invalid_argument(
            "register_action_state player does not match node.player."
        );
    }

    if (n.edge_count <= 0) {
        throw std::invalid_argument(
            "Cannot register ActionState before adding action children."
        );
    }

    if (n.action_state_index != -1) {
        return n.action_state_index;
    }

    const int buckets = bucket_count(player);

    if (buckets <= 0) {
        throw std::invalid_argument(
            "Cannot register ActionState with zero hand buckets."
        );
    }

    ActionState state{};
    state.node = node_id;
    state.player = static_cast<int>(player);
    state.bucket_count = buckets;
    state.action_count = n.edge_count;

    // There is no first_action field on PublicNode, so use the action_index of
    // the first outgoing edge. This assumes the builder appends actions in the
    // same order as edges.
    const NodeEdge& first = edges[static_cast<std::size_t>(n.first_edge)];

    if (first.action_index < 0) {
        throw std::logic_error(
            "Action node's first edge does not have a valid action_index."
        );
    }

    state.first_action = first.action_index;

    state.tensor_offset = checked_size_to_u64(
        cfr_tensor_entries(),
        "ActionState tensor_offset"
    );

    state.state_bucket_offset = checked_size_to_u64(
        state_bucket_entries(),
        "ActionState state_bucket_offset"
    );

    const int state_id = static_cast<int>(action_states.size());
    action_states.push_back(state);

    n.action_state_index = state_id;

    return state_id;
}

// -----------------------------------------------------------------------------
// Memory estimate
// -----------------------------------------------------------------------------

GameMemoryEstimate Game::estimate_memory() const {
    GameMemoryEstimate estimate;
    // Used elements.
    estimate.node_bytes =
        sizeof(PublicNode) * nodes.size();
    estimate.edge_bytes =
        sizeof(NodeEdge) * edges.size();
    estimate.action_bytes =
        sizeof(GameAction) * actions.size();
    estimate.action_state_bytes =
        sizeof(ActionState) * action_states.size();
    estimate.p0_hand_bytes =
        sizeof(HandId) * p0_hands.hands.size();
    estimate.p1_hand_bytes =
        sizeof(HandId) * p1_hands.hands.size();
    estimate.hand_pair_bytes =
        sizeof(int) * (
            hand_pairs.p0_index.size() +
            hand_pairs.p1_index.size()
        );
    estimate.terminal_node_bytes =
        sizeof(int) * terminal_nodes.size();
    estimate.terminal_index_by_node_bytes =
        sizeof(int) * terminal_index_by_node.size();
    estimate.terminal_value_p0_bytes =
        sizeof(float) * terminal_value_p0.size();
    estimate.cfr_tensor_entries = cfr_tensor_entries();
    estimate.state_bucket_entries = state_bucket_entries();
    estimate.bytes_per_float_tensor =
        sizeof(float) * estimate.cfr_tensor_entries;
    estimate.bytes_per_state_bucket_float_array =
        sizeof(float) * estimate.state_bucket_entries;
    estimate.public_tree_bytes =
        estimate.node_bytes +
        estimate.edge_bytes +
        estimate.action_bytes +
        estimate.action_state_bytes;
    estimate.hand_side_table_bytes =
        estimate.p0_hand_bytes +
        estimate.p1_hand_bytes +
        estimate.hand_pair_bytes;
    estimate.terminal_bytes =
        estimate.terminal_node_bytes +
        estimate.terminal_index_by_node_bytes +
        estimate.terminal_value_p0_bytes;
    estimate.game_owned_bytes =
        estimate.public_tree_bytes +
        estimate.hand_side_table_bytes +
        estimate.terminal_bytes;
    // Reserved heap capacity. This is usually the better RAM estimate.
    estimate.game_owned_capacity_bytes =
        vector_bytes_capacity(nodes) +
        vector_bytes_capacity(edges) +
        vector_bytes_capacity(actions) +
        vector_bytes_capacity(action_states) +
        vector_bytes_capacity(p0_hands.hands) +
        vector_bytes_capacity(p1_hands.hands) +
        vector_bytes_capacity(hand_pairs.p0_index) +
        vector_bytes_capacity(hand_pairs.p1_index) +
        vector_bytes_capacity(terminal_nodes) +
        vector_bytes_capacity(terminal_index_by_node) +
        vector_bytes_capacity(terminal_value_p0);
    return estimate;
}
// -----------------------------------------------------------------------------
// Validation
// -----------------------------------------------------------------------------

void Game::validate() const {
    if (num_players != 2) {
        throw std::invalid_argument("Game currently expects exactly two players.");
    }

    if (nodes.empty()) {
        throw std::invalid_argument("Game must contain at least one node.");
    }

    if (root < 0 || root >= num_nodes()) {
        throw std::invalid_argument("Game root is out of range.");
    }

    if (max_depth < 0) {
        throw std::invalid_argument("Game max_depth cannot be negative.");
    }

    for (int node_id = 0; node_id < num_nodes(); ++node_id) {
        const PublicNode& n = nodes[static_cast<std::size_t>(node_id)];
        if (n.parent < -1 || n.parent >= num_nodes()) {
            throw std::invalid_argument("PublicNode parent is out of range.");
        }

        if (node_id == root) {
            if (n.parent != -1 && n.parent != root) {
                throw std::invalid_argument(
                    "Root parent must be -1 or self-parented."
                );
            }

            if (n.depth != 0) {
                throw std::invalid_argument("Root depth must be zero.");
            }
        } else {
            if (n.parent < 0) {
                throw std::invalid_argument("Non-root node must have a parent.");
            }

            const PublicNode& parent_node =
                nodes[static_cast<std::size_t>(n.parent)];

            if (n.depth != parent_node.depth + 1) {
                throw std::invalid_argument(
                    "Node depth must equal parent depth + 1."
                );
            }
        }

        if (n.depth < 0 || n.depth > max_depth) {
            throw std::invalid_argument("PublicNode depth is invalid.");
        }

        if (n.edge_count < 0) {
            throw std::invalid_argument("PublicNode edge_count cannot be negative.");
        }

        if (n.first_edge < 0) {
            throw std::invalid_argument("PublicNode first_edge cannot be negative.");
        }

        if (n.edge_count > 0) {
            if (n.first_edge + n.edge_count > num_edges()) {
                throw std::invalid_argument(
                    "PublicNode edge range is out of bounds."
                );
            }
        }

        switch (n.type) {
            case PublicNodeType::Action: {

                if (n.edge_count <= 0) {
                    throw std::invalid_argument(
                        "Action node must have at least one outgoing edge."
                    );
                }

                if (n.action_state_index < 0 ||
                    n.action_state_index >= num_action_states()) {
                    throw std::invalid_argument(
                        "Action node has invalid action_state_index."
                    );
                }

                break;
            }

            case PublicNodeType::Chance: {
                if (n.player != Player::Chance) {
                    throw std::invalid_argument(
                        "Chance node must have player Chance."
                    );
                }

                if (n.edge_count <= 0) {
                    throw std::invalid_argument(
                        "Chance node must have at least one outgoing edge."
                    );
                }

                if (n.action_state_index != -1) {
                    throw std::invalid_argument(
                        "Chance node cannot have action_state_index."
                    );
                }

                double probability_sum = 0.0;

                for (int i = 0; i < n.edge_count; ++i) {
                    const NodeEdge& e =
                        edges[static_cast<std::size_t>(n.first_edge + i)];

                    probability_sum += static_cast<double>(e.chance_prob);
                }

                if (std::abs(probability_sum - 1.0) > 1e-5) {
                    throw std::invalid_argument(
                        "Chance-node probabilities must sum to one."
                    );
                }

                break;
            }

            case PublicNodeType::Terminal: {
                if (n.player != Player::Terminal) {
                    throw std::invalid_argument(
                        "Terminal node must have player Terminal."
                    );
                }
                if (n.edge_count != 0) {
                    throw std::invalid_argument(
                        "Terminal node cannot have outgoing edges."
                    );
                }
                if (n.action_state_index != -1) {
                    throw std::invalid_argument(
                        "Terminal node cannot have action_state_index."
                    );
                }
                break;
            }
        }
    }

    for (int parent_id = 0; parent_id < num_nodes(); ++parent_id) {
        const PublicNode& parent_node =
            nodes[static_cast<std::size_t>(parent_id)];

        for (int i = 0; i < parent_node.edge_count; ++i) {
            const int edge_id = parent_node.first_edge + i;
            const NodeEdge& e = edges[static_cast<std::size_t>(edge_id)];

            if (e.child < 0 || e.child >= num_nodes()) {
                throw std::invalid_argument("NodeEdge child is out of range.");
            }

            const PublicNode& child = nodes[static_cast<std::size_t>(e.child)];

            if (child.parent != parent_id) {
                throw std::invalid_argument(
                    "NodeEdge child parent pointer mismatch."
                );
            }

            if (child.depth != parent_node.depth + 1) {
                throw std::invalid_argument(
                    "NodeEdge child depth mismatch."
                );
            }

            if (!(e.chance_prob > 0.0f) || e.chance_prob > 1.0f) {
                throw std::invalid_argument("NodeEdge chance_prob is invalid.");
            }

            if (parent_node.type == PublicNodeType::Action) {
                if (e.local_action != i) {
                    throw std::invalid_argument(
                        "Action edge local_action must match edge order."
                    );
                }

                if (e.action_index < 0 || e.action_index >= num_actions()) {
                    throw std::invalid_argument(
                        "Action edge action_index is out of range."
                    );
                }

                if (e.public_card != -1) {
                    throw std::invalid_argument(
                        "Action edge public_card must be -1."
                    );
                }

                if (e.chance_prob != 1.0f) {
                    throw std::invalid_argument(
                        "Action edge chance_prob must be 1."
                    );
                }
            } else if (parent_node.type == PublicNodeType::Chance) {
                if (e.local_action != -1) {
                    throw std::invalid_argument(
                        "Chance edge local_action must be -1."
                    );
                }

                if (e.public_card < 0 || e.public_card >= kNumCards) {
                    throw std::invalid_argument(
                        "Chance edge public_card is out of range."
                    );
                }
            } else {
                throw std::invalid_argument(
                    "Terminal node cannot own edges."
                );
            }
        }
    }

    if (!p0_hands.empty()) {
        for (HandId hand : p0_hands.hands) {
            validate_hand_id(hand);
        }
    }

    if (!p1_hands.empty()) {
        for (HandId hand : p1_hands.hands) {
            validate_hand_id(hand);
        }
    }

    if (hand_pairs.p0_index.size() != hand_pairs.p1_index.size()) {
        throw std::invalid_argument(
            "HandPairTable p0_index and p1_index size mismatch."
        );
    }

    if (!hand_pairs.empty()) {
        if (p0_hands.empty() || p1_hands.empty()) {
            throw std::invalid_argument(
                "HandPairTable requires nonempty hand domains."
            );
        }

        for (int pair_id = 0; pair_id < hand_pairs.pair_count(); ++pair_id) {
            const int p0_index =
                hand_pairs.p0_index[static_cast<std::size_t>(pair_id)];

            const int p1_index =
                hand_pairs.p1_index[static_cast<std::size_t>(pair_id)];

            if (p0_index < 0 || p0_index >= p0_hands.hand_count()) {
                throw std::invalid_argument(
                    "HandPairTable P0 index out of range."
                );
            }

            if (p1_index < 0 || p1_index >= p1_hands.hand_count()) {
                throw std::invalid_argument(
                    "HandPairTable P1 index out of range."
                );
            }

            const HandId p0_hand =
                p0_hands.hands[static_cast<std::size_t>(p0_index)];

            const HandId p1_hand =
                p1_hands.hands[static_cast<std::size_t>(p1_index)];

            if (hands_overlap(p0_hand, p1_hand)) {
                throw std::invalid_argument(
                    "HandPairTable contains overlapping hands."
                );
            }
        }
    }

    std::size_t expected_tensor_offset = 0;
    std::size_t expected_state_bucket_offset = 0;

    for (int state_id = 0; state_id < num_action_states(); ++state_id) {
        const ActionState& state =
            action_states[static_cast<std::size_t>(state_id)];

        if (state.node < 0 || state.node >= num_nodes()) {
            throw std::invalid_argument("ActionState node is out of range.");
        }

        const PublicNode& n =
            nodes[static_cast<std::size_t>(state.node)];

        if (n.type != PublicNodeType::Action) {
            throw std::invalid_argument(
                "ActionState must point to an Action node."
            );
        }

        if (n.action_state_index != state_id) {
            throw std::invalid_argument(
                "ActionState/node back-reference mismatch."
            );
        }

        const Player state_player = static_cast<Player>(state.player);
        if (state.bucket_count <= 0) {
            throw std::invalid_argument(
                "ActionState bucket_count must be positive."
            );
        }

        if (state.action_count <= 0) {
            throw std::invalid_argument(
                "ActionState action_count must be positive."
            );
        }

        if (state.action_count != n.edge_count) {
            throw std::invalid_argument(
                "ActionState action_count must equal node edge_count."
            );
        }

        if (state.bucket_count != bucket_count(state_player)) {
            throw std::invalid_argument(
                "ActionState bucket_count does not match acting player's domain."
            );
        }

        if (state.first_action < 0 ||
            state.first_action >= num_actions()) {
            throw std::invalid_argument(
                "ActionState first_action is out of range."
            );
        }

        if (state.tensor_offset != expected_tensor_offset) {
            throw std::invalid_argument(
                "ActionState tensor_offset is not contiguous."
            );
        }

        if (state.state_bucket_offset != expected_state_bucket_offset) {
            throw std::invalid_argument(
                "ActionState state_bucket_offset is not contiguous."
            );
        }

        expected_tensor_offset += action_state_tensor_size(state);
        expected_state_bucket_offset += action_state_bucket_size(state);
    }

    if (expected_tensor_offset != cfr_tensor_entries()) {
        throw std::invalid_argument(
            "Game cfr_tensor_entries does not match ActionState layout."
        );
    }

    if (expected_state_bucket_offset != state_bucket_entries()) {
        throw std::invalid_argument(
            "Game state_bucket_entries does not match ActionState layout."
        );
    }
}
void Game::print_game_memory_usage() const {
    const GameMemoryEstimate m = estimate_memory();

    auto print = [](const char* name, std::size_t bytes) {
        std::cout
            << std::left << std::setw(34) << name
            << std::right << std::setw(14) << bytes << " bytes  "
            << std::fixed << std::setprecision(3)
            << (static_cast<double>(bytes) / (1024.0 * 1024.0))
            << " MiB\n";
    };

    std::cout << "\n=== Game memory usage ===\n";

    print("Public nodes", m.node_bytes);
    print("Edges", m.edge_bytes);
    print("Actions", m.action_bytes);
    print("Action states", m.action_state_bytes);
    print("Public tree subtotal", m.public_tree_bytes);

    std::cout << '\n';

    print("P0 hand domain", m.p0_hand_bytes);
    print("P1 hand domain", m.p1_hand_bytes);
    print("Hand pair table", m.hand_pair_bytes);
    print("Hand side-table subtotal", m.hand_side_table_bytes);

    std::cout << '\n';

    print("Terminal nodes", m.terminal_node_bytes);
    print("Terminal index by node", m.terminal_index_by_node_bytes);
    print("Terminal value P0", m.terminal_value_p0_bytes);
    print("Terminal subtotal", m.terminal_bytes);

    std::cout << '\n';

    print("Game owned used bytes", m.game_owned_bytes);
    print("Game owned capacity bytes", m.game_owned_capacity_bytes);

    std::cout << '\n';

    std::cout << "CFR tensor entries:        "
              << m.cfr_tensor_entries << '\n';

    std::cout << "State-bucket entries:      "
              << m.state_bucket_entries << '\n';

    print("One float CFR tensor", m.bytes_per_float_tensor);
    print("One state-bucket float vec", m.bytes_per_state_bucket_float_array);

    std::cout << "=========================\n";
}

} // namespace poker