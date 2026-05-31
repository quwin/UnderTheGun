#pragma once

#include "poker/board.hpp"
#include "poker/hand.hpp"

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace poker {

enum class Player : std::int8_t  {
    Chance   = -1,
    P0       = 0,
    P1       = 1,
    Terminal = 2
};
inline std::string to_string(Player player) {
    switch (player) {
        case Player::Chance:
            return "Chance";
        case Player::P0:
            return "P0";
        case Player::P1:
            return "P1";
        case Player::Terminal:
            return "Terminal";
    }
    return "Unknown";
}
enum class PublicNodeType : std::uint8_t {
    Action   = 0,
    Chance   = 1,
    Terminal = 2
};
enum class TerminalType : std::uint8_t {
    None     = 0,
    P0_Fold     = 1,
    P1_Fold     = 2,
    Showdown = 3,
    AllIn    = 4
};
// -----------------------------------------------------------------------------
// Action representation
// -----------------------------------------------------------------------------
struct GameAction {
    int action_type = 0;
    int amount = 0;
    GameAction() = default;
    GameAction(const int type, const int action_amount): action_type(type),amount(action_amount) {}
};
inline bool operator==(const GameAction& a, const GameAction& b) {
    return a.action_type == b.action_type && a.amount == b.amount;
}
inline bool operator!=(const GameAction& a, const GameAction& b) {
    return !(a == b);
}
inline std::string to_string(const GameAction& action) {
    return "type=" + std::to_string(action.action_type) + ":amount=" + std::to_string(action.amount);
}
// -----------------------------------------------------------------------------
// Connection between two Nodes:
// -----------------------------------------------------------------------------
struct NodeEdge {
    // Index of child
    int child = -1;
    // Used for action-node parents.
    // This is the local action number at the parent action state.
    // For chance-node parents, this should be -1.
    int local_action = -1;
    // Index into Game::actions.
    // For chance edges, this may be -1 or a synthetic deal action.
    int action_index = -1;
    // Meaningful for public chance edges.
    // Usually raw phevaluator::Card id.
    // For action edges, this should be -1.
    int public_card = -1;
    // Action edges use 1.0.
    // Chance edges use normalized transition probability.
    float chance_prob = 1.0f;
};
// -----------------------------------------------------------------------------
// Public node
// -----------------------------------------------------------------------------
//
// A PublicNode represents only public information:
//
//   - acting player
//   - board/street
//   - pot/stack/commitment state
//   - outgoing public actions/chance outcomes
//
// It deliberately does NOT contain:
//
//   - P0 private hand
//   - P1 private hand
//   - PrivateState
//   - per-hand strategy vectors
//   - per-hand regret vectors
//
// Hand-aware data lives in HandDomain, HandPairTable, and ActionState tensors.

struct PublicNode {
    int parent = -1;
    int depth = 0;
    PublicNodeType type = PublicNodeType::Terminal;
    Player player = Player::Terminal;
    // Contiguous edge range in Game::edges.
    int first_edge = 0;
    int edge_count = 0;
    int action_state_index = -1;
};

// -----------------------------------------------------------------------------
// Hand-aware side tables
// Weights every possible hand in range equally
// -----------------------------------------------------------------------------
struct HandDomain {
    // Exact legal private hands for this player after public board blockers.
    std::vector<HandId> hands;
    [[nodiscard]] int hand_count() const {
        return static_cast<int>(hands.size());
    }
    [[nodiscard]] bool empty() const {
        return hands.empty();
    }
};
// -----------------------------------------------------------------------------
// Legal private hand-pair table
// -----------------------------------------------------------------------------
//
// This table stores legal P0/P1 hand pairs once for the whole subgame.
//
// It replaces:
//
//   root chance -> exact P0/P1 private hand pair -> duplicate subtree
//
// with:
//
//   public tree
//   + one compact legal-pair table
//   + terminal evaluator over legal pairs

struct HandPairTable {
    // Flat legal pair arrays.
    //
    // p0_index[k] and p1_index[k] are indices into:
    //   p0_domain.hands
    //   p1_domain.hands
    std::vector<int> p0_index;
    std::vector<int> p1_index;

    [[nodiscard]] int pair_count() const {
        return static_cast<int>(p0_index.size());
    }
    [[nodiscard]] bool empty() const {
        return p0_index.empty();
    }
};

// -----------------------------------------------------------------------------
// Action-state tensor metadata
// -----------------------------------------------------------------------------
// An ActionState is the hand-aware equivalent of an infoset-action block.
// Strategy/regret arrays are indexed as:
//   tensor_offset
// + bucket * action_count
// + local_action
// State-only hand/bucket arrays are indexed as:
//   state_bucket_offset
// + bucket

struct ActionState {
    int node;
    int player;
    int bucket_count;
    int action_count;
    int first_action;
    std::uint64_t tensor_offset;
    std::uint64_t state_bucket_offset;
    [[nodiscard]] std::size_t tensor_index(
    int bucket,
    int local_action
) const {
        if (bucket < 0 || bucket >= bucket_count) {
            throw std::invalid_argument("ActionState bucket out of range.");
        }

        if (local_action < 0 || local_action >= action_count) {
            throw std::invalid_argument("ActionState local_action out of range.");
        }

        return static_cast<std::size_t>(tensor_offset) +
               static_cast<std::size_t>(bucket) *
                   static_cast<std::size_t>(action_count) +
               static_cast<std::size_t>(local_action);
    }
    [[nodiscard]] std::size_t state_bucket_index(int bucket) const {
        if (bucket < 0 || bucket >= bucket_count) {
            throw std::invalid_argument("ActionState bucket out of range.");
        }

        return static_cast<std::size_t>(state_bucket_offset) +
               static_cast<std::size_t>(bucket);
    }
};

// -----------------------------------------------------------------------------
// Memory estimate
// -----------------------------------------------------------------------------

struct GameMemoryEstimate {
    std::size_t node_bytes = 0;
    std::size_t edge_bytes = 0;
    std::size_t action_bytes = 0;
    std::size_t action_state_bytes = 0;

    std::size_t p0_hand_bytes = 0;
    std::size_t p1_hand_bytes = 0;
    std::size_t hand_pair_bytes = 0;

    std::size_t terminal_node_bytes = 0;
    std::size_t terminal_index_by_node_bytes = 0;
    std::size_t terminal_value_p0_bytes = 0;

    std::size_t cfr_tensor_entries = 0;
    std::size_t state_bucket_entries = 0;

    std::size_t bytes_per_float_tensor = 0;
    std::size_t bytes_per_state_bucket_float_array = 0;

    std::size_t public_tree_bytes = 0;
    std::size_t hand_side_table_bytes = 0;
    std::size_t terminal_bytes = 0;

    std::size_t game_owned_bytes = 0;

    // Optional but useful: actual reserved heap memory.
    std::size_t game_owned_capacity_bytes = 0;
};

// -----------------------------------------------------------------------------
// Game
// -----------------------------------------------------------------------------

class Game {
public:
    int root = -1;
    int num_players = 2;
    int max_depth = 0;
    // Compact public tree.
    std::vector<PublicNode> nodes;
    std::vector<NodeEdge> edges;
    std::vector<GameAction> actions;
    // CFR tensor metadata.
    std::vector<ActionState> action_states;
    // Hand-aware side data.
    HandDomain p0_hands;
    HandDomain p1_hands;
    HandPairTable hand_pairs;

    std::vector<int> terminal_nodes;
    // terminal_index_by_node[node_id] = terminal index, or -1 for nonterminal.
    std::vector<int> terminal_index_by_node;
    std::vector<float> terminal_value_p0;

    Game() = default;
    void print_game_memory_usage() const;
    // ---------------------------------------------------------------------
    // Public tree construction
    // ---------------------------------------------------------------------

    int add_node(PublicNode node);

    int add_action(const GameAction& action);

    void add_action_child(
        int parent,
        int child,
        int action_index
    );

    void add_chance_child(
        int parent,
        int child,
        int public_card,
        float probability
    );

    // ---------------------------------------------------------------------
    // Hand-domain setup
    // ---------------------------------------------------------------------

    void set_hand_domains(
        HandDomain p0,
        HandDomain p1,
        HandPairTable pairs
    );

    const HandDomain& hand_domain(Player player) const;
    HandDomain& hand_domain(Player player);
    int bucket_count(Player player) const;

    // ---------------------------------------------------------------------
    // Action-state registration
    // ---------------------------------------------------------------------

    int register_action_state(
        int node_id,
        Player player
    );

    // ---------------------------------------------------------------------
    // Accessors
    // ---------------------------------------------------------------------

    const PublicNode& node(int id) const;
    PublicNode& node(int id);

    const NodeEdge& edge(int id) const;
    NodeEdge& edge(int id);

    const GameAction& action(int id) const;
    GameAction& action(int id);

    const ActionState& action_state(int id) const;
    ActionState& action_state(int id);

    int num_nodes() const;
    int num_edges() const;
    int num_actions() const;
    int num_action_states() const;

    std::size_t cfr_tensor_entries() const;
    std::size_t state_bucket_entries() const;

    GameMemoryEstimate estimate_memory() const;

    void validate() const;

private:
    void attach_edge(
        int parent,
        NodeEdge edge
    );
};

// -----------------------------------------------------------------------------
// Inline accessors
// -----------------------------------------------------------------------------

inline const PublicNode& Game::node(int id) const {
    assert(id >= 0);
    assert(id < static_cast<int>(nodes.size()));
    return nodes[static_cast<std::size_t>(id)];
}

inline PublicNode& Game::node(int id) {
    assert(id >= 0);
    assert(id < static_cast<int>(nodes.size()));
    return nodes[static_cast<std::size_t>(id)];
}

inline const NodeEdge& Game::edge(int id) const {
    assert(id >= 0);
    assert(id < static_cast<int>(edges.size()));
    return edges[static_cast<std::size_t>(id)];
}

inline NodeEdge& Game::edge(int id) {
    assert(id >= 0);
    assert(id < static_cast<int>(edges.size()));
    return edges[static_cast<std::size_t>(id)];
}

inline const GameAction& Game::action(int id) const {
    assert(id >= 0);
    assert(id < static_cast<int>(actions.size()));
    return actions[static_cast<std::size_t>(id)];
}

inline GameAction& Game::action(int id) {
    assert(id >= 0);
    assert(id < static_cast<int>(actions.size()));
    return actions[static_cast<std::size_t>(id)];
}

inline const ActionState& Game::action_state(int id) const {
    assert(id >= 0);
    assert(id < static_cast<int>(action_states.size()));
    return action_states[static_cast<std::size_t>(id)];
}

inline ActionState& Game::action_state(int id) {
    assert(id >= 0);
    assert(id < static_cast<int>(action_states.size()));
    return action_states[static_cast<std::size_t>(id)];
}

inline int Game::num_nodes() const {
    return static_cast<int>(nodes.size());
}

inline int Game::num_edges() const {
    return static_cast<int>(edges.size());
}

inline int Game::num_actions() const {
    return static_cast<int>(actions.size());
}

inline int Game::num_action_states() const {
    return static_cast<int>(action_states.size());
}

    inline std::size_t Game::cfr_tensor_entries() const {
    if (action_states.empty()) {
        return 0;
    }

    const ActionState& last = action_states.back();
    return static_cast<std::size_t>(last.tensor_offset) +
           static_cast<std::size_t>(last.bucket_count) *
           static_cast<std::size_t>(last.action_count);
}

    inline std::size_t Game::state_bucket_entries() const {
    if (action_states.empty()) {
        return 0;
    }

    const ActionState& last = action_states.back();
    return static_cast<std::size_t>(last.state_bucket_offset) +
           static_cast<std::size_t>(last.bucket_count);
}

inline const HandDomain& Game::hand_domain(Player player) const {
    if (player == Player::P0) {
        return p0_hands;
    }

    if (player == Player::P1) {
        return p1_hands;
    }

    throw std::invalid_argument("hand_domain requires P0 or P1.");
}

inline HandDomain& Game::hand_domain(Player player) {
    if (player == Player::P0) {
        return p0_hands;
    }

    if (player == Player::P1) {
        return p1_hands;
    }

    throw std::invalid_argument("hand_domain requires P0 or P1.");
}

// -----------------------------------------------------------------------------
// Standalone index helpers
// -----------------------------------------------------------------------------

inline std::size_t strategy_tensor_index(
    const Game& game,
    int action_state_id,
    int bucket,
    int local_action
) {
    return game.action_state(action_state_id).tensor_index(
        bucket,
        local_action
    );
}
} // namespace poker