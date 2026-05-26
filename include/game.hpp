#pragma once

#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace poker {

// -----------------------------------------------------------------------------
// Basic generic game enums
// -----------------------------------------------------------------------------

enum class Player : int {
    Chance   = -1,
    P0       = 0,
    P1       = 1,
    Terminal = 2
};

inline int to_index(Player player) {
    if (player == Player::P0) {
        return 0;
    }

    if (player == Player::P1) {
        return 1;
    }

    throw std::invalid_argument("to_index requires P0 or P1.");
}

inline bool is_real_player(Player player) {
    return player == Player::P0 || player == Player::P1;
}

inline bool is_chance(Player player) {
    return player == Player::Chance;
}

inline bool is_terminal_player(Player player) {
    return player == Player::Terminal;
}

inline Player opponent_of(Player player) {
    if (player == Player::P0) {
        return Player::P1;
    }

    if (player == Player::P1) {
        return Player::P0;
    }

    throw std::invalid_argument("opponent_of requires P0 or P1.");
}

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

// -----------------------------------------------------------------------------
// Generic action representation
// -----------------------------------------------------------------------------
//
// This replaces the old Kuhn-only enum:
//
//   enum class Action { Deal, Check, Bet, Call, Fold };
//
// The generic CFR game only needs a stable action id plus optional metadata.
// Hold'em-specific code can encode holdem::Action into this structure.
//
// Examples:
//
//   Kuhn check:
//     action_type = 1
//     amount = 0
//     label = "check"
//
//   Hold'em bet 500:
//     action_type = static_cast<int>(holdem::ActionType::Bet)
//     amount = 500
//     label = "bet:500"
//
//   Chance private deal:
//     action_type = 0
//     amount = 0
//     label = "deal"

struct GameAction {
    int action_type = 0;
    int amount = 0;
    std::string label;

    GameAction() = default;

    GameAction(
        int type,
        int action_amount,
        std::string action_label
    )
        : action_type(type),
          amount(action_amount),
          label(std::move(action_label)) {}
};

inline bool operator==(const GameAction& a, const GameAction& b) {
    return a.action_type == b.action_type &&
           a.amount == b.amount &&
           a.label == b.label;
}

inline bool operator!=(const GameAction& a, const GameAction& b) {
    return !(a == b);
}

inline std::string to_string(const GameAction& action) {
    if (!action.label.empty()) {
        return action.label;
    }

    return "type=" + std::to_string(action.action_type) +
           ":amount=" + std::to_string(action.amount);
}

inline GameAction chance_deal_action() {
    return GameAction{0, 0, "deal"};
}

// -----------------------------------------------------------------------------
// Generic game-tree node
// -----------------------------------------------------------------------------

struct Node {
    int id = -1;
    int parent = -1;
    int depth = 0;

    Player player = Player::Terminal;

    // Infoset id for real-player decision nodes.
    //
    // Use:
    //   -1 for chance nodes
    //   -1 for terminal nodes
    int infoset = -1;

    // Action taken from parent to reach this node.
    //
    // For root, this can be chance_deal_action() or ignored.
    GameAction incoming_action = chance_deal_action();

    // Chance probability from parent to this node.
    //
    // Meaningful only when parent.player == Player::Chance.
    //
    // For non-chance edges, use 0.0f or 1.0f. CFR traversal should look at the
    // parent node's player to decide how to interpret the edge.
    float chance_prob = 0.0f;

    bool terminal = false;

    // Utility from P0's perspective.
    //
    // Two-player zero-sum convention:
    //
    //   utility_p1 = -utility_p0
    //
    // Nonterminal nodes should have utility_p0 = 0.
    float utility_p0 = 0.0f;

    std::vector<int> children;
};

// -----------------------------------------------------------------------------
// Information sets
// -----------------------------------------------------------------------------

struct InfoSet {
    int id = -1;

    Player player = Player::Terminal;

    // Debuggable serialized key.
    //
    // For Kuhn this might be:
    //   "P0|K|cb"
    //
    // For Hold'em this might be:
    //   "p=P0|st=river|board=As7h2cJd4s|hbucket=123|hist=b1000|..."
    //
    // The key must be created by the game-specific builder.
    std::string key;

    // Legal actions available at this infoset.
    //
    // Same size/order as q_indices.
    std::vector<GameAction> actions;

    // Flat strategy-vector indices corresponding to actions.
    //
    // Same size/order as actions.
    std::vector<int> q_indices;
};

// A flat strategy entry q = (infoset, local action).
struct InfoSetAction {
    int q = -1;
    int infoset = -1;
    int local_action = -1;

    GameAction action;
};

// -----------------------------------------------------------------------------
// Complete game container
// -----------------------------------------------------------------------------

struct Game {
    std::vector<Node> nodes;
    std::vector<InfoSet> infosets;
    std::vector<InfoSetAction> q_entries;

    int root = 0;
    int num_players = 2;
    int max_depth = 0;

    const Node& node(int id) const {
        assert(id >= 0);
        assert(id < static_cast<int>(nodes.size()));
        return nodes[static_cast<std::size_t>(id)];
    }

    Node& node(int id) {
        assert(id >= 0);
        assert(id < static_cast<int>(nodes.size()));
        return nodes[static_cast<std::size_t>(id)];
    }

    const InfoSet& infoset(int id) const {
        assert(id >= 0);
        assert(id < static_cast<int>(infosets.size()));
        return infosets[static_cast<std::size_t>(id)];
    }

    InfoSet& infoset(int id) {
        assert(id >= 0);
        assert(id < static_cast<int>(infosets.size()));
        return infosets[static_cast<std::size_t>(id)];
    }

    const InfoSetAction& q_entry(int q) const {
        assert(q >= 0);
        assert(q < static_cast<int>(q_entries.size()));
        return q_entries[static_cast<std::size_t>(q)];
    }

    InfoSetAction& q_entry(int q) {
        assert(q >= 0);
        assert(q < static_cast<int>(q_entries.size()));
        return q_entries[static_cast<std::size_t>(q)];
    }

    int num_nodes() const {
        return static_cast<int>(nodes.size());
    }

    int num_infosets() const {
        return static_cast<int>(infosets.size());
    }

    int num_q() const {
        return static_cast<int>(q_entries.size());
    }

    bool empty() const {
        return nodes.empty();
    }
};

// -----------------------------------------------------------------------------
// Build context
// -----------------------------------------------------------------------------
//
// This replaces the Kuhn-specific BuildContext that used:
//
//   Player + Card + public_history
//
// The new BuildContext uses a generic serialized infoset key.
// KuhnBuilder and HoldemSubgameBuilder are responsible for creating that key.

struct BuildContext {
    Game game;

    std::unordered_map<std::string, int> infoset_key_to_id;

    int add_node(Node node) {
        node.id = static_cast<int>(game.nodes.size());

        if (node.parent >= 0) {
            const Node& parent = game.node(node.parent);
            node.depth = parent.depth + 1;
        }

        game.max_depth = std::max(game.max_depth, node.depth);

        game.nodes.push_back(std::move(node));

        return game.nodes.back().id;
    }

    void add_child(int parent_id, int child_id) {
        if (parent_id < 0 || parent_id >= game.num_nodes()) {
            throw std::invalid_argument("add_child parent_id out of range.");
        }

        if (child_id < 0 || child_id >= game.num_nodes()) {
            throw std::invalid_argument("add_child child_id out of range.");
        }

        Node& parent = game.node(parent_id);
        Node& child = game.node(child_id);

        child.parent = parent_id;

        if (child.depth != parent.depth + 1) {
            child.depth = parent.depth + 1;
            game.max_depth = std::max(game.max_depth, child.depth);
        }

        parent.children.push_back(child_id);
    }

    int get_or_create_infoset(
        Player player,
        const std::string& key,
        const std::vector<GameAction>& actions
    ) {
        if (!is_real_player(player)) {
            throw std::invalid_argument(
                "Infoset owner must be P0 or P1."
            );
        }

        if (key.empty()) {
            throw std::invalid_argument(
                "Infoset key cannot be empty."
            );
        }

        if (actions.empty()) {
            throw std::invalid_argument(
                "Infoset must contain at least one action."
            );
        }

        const auto found = infoset_key_to_id.find(key);

        if (found != infoset_key_to_id.end()) {
            const int infoset_id = found->second;
            InfoSet& infoset = game.infoset(infoset_id);

            if (infoset.player != player) {
                throw std::runtime_error(
                    "Existing infoset player does not match requested player."
                );
            }

            if (infoset.actions.size() != actions.size()) {
                throw std::runtime_error(
                    "Existing infoset action count does not match requested actions."
                );
            }

            for (std::size_t i = 0; i < actions.size(); ++i) {
                if (infoset.actions[i] != actions[i]) {
                    throw std::runtime_error(
                        "Existing infoset actions do not match requested actions."
                    );
                }
            }

            return infoset_id;
        }

        InfoSet infoset;
        infoset.id = static_cast<int>(game.infosets.size());
        infoset.player = player;
        infoset.key = key;
        infoset.actions = actions;

        infoset.q_indices.reserve(actions.size());

        for (int local = 0; local < static_cast<int>(actions.size()); ++local) {
            const int q = static_cast<int>(game.q_entries.size());

            infoset.q_indices.push_back(q);

            InfoSetAction q_entry;
            q_entry.q = q;
            q_entry.infoset = infoset.id;
            q_entry.local_action = local;
            q_entry.action = actions[static_cast<std::size_t>(local)];

            game.q_entries.push_back(std::move(q_entry));
        }

        const int infoset_id = infoset.id;

        game.infosets.push_back(std::move(infoset));
        infoset_key_to_id.emplace(key, infoset_id);

        return infoset_id;
    }
};

// -----------------------------------------------------------------------------
// Validation helpers
// -----------------------------------------------------------------------------

inline void validate_game_basic_shape(const Game& game) {
    if (game.nodes.empty()) {
        throw std::invalid_argument("Game must contain at least one node.");
    }

    if (game.root < 0 || game.root >= game.num_nodes()) {
        throw std::invalid_argument("Game root is out of range.");
    }

    if (game.num_players != 2) {
        throw std::invalid_argument(
            "This CFR implementation currently expects two players."
        );
    }

    for (const Node& node : game.nodes) {
        if (node.id < 0 || node.id >= game.num_nodes()) {
            throw std::invalid_argument("Node id out of range.");
        }

        if (node.id == game.root) {
            if (node.parent != -1) {
                throw std::invalid_argument("Root node must have parent -1.");
            }
        } else {
            if (node.parent < 0 || node.parent >= game.num_nodes()) {
                throw std::invalid_argument(
                    "Non-root node has invalid parent."
                );
            }
        }

        if (node.depth < 0 || node.depth > game.max_depth) {
            throw std::invalid_argument("Node depth out of range.");
        }

        if (node.terminal) {
            if (node.player != Player::Terminal) {
                throw std::invalid_argument(
                    "Terminal node must have player Terminal."
                );
            }

            if (!node.children.empty()) {
                throw std::invalid_argument(
                    "Terminal node cannot have children."
                );
            }

            if (node.infoset != -1) {
                throw std::invalid_argument(
                    "Terminal node cannot have infoset."
                );
            }
        }

        if (node.player == Player::Chance) {
            if (node.infoset != -1) {
                throw std::invalid_argument(
                    "Chance node cannot have infoset."
                );
            }
        }

        if (is_real_player(node.player)) {
            if (node.infoset < 0 || node.infoset >= game.num_infosets()) {
                throw std::invalid_argument(
                    "Real-player node has invalid infoset."
                );
            }
        }

        for (int child_id : node.children) {
            if (child_id < 0 || child_id >= game.num_nodes()) {
                throw std::invalid_argument("Child id out of range.");
            }

            const Node& child = game.node(child_id);

            if (child.parent != node.id) {
                throw std::invalid_argument(
                    "Parent/child pointer mismatch."
                );
            }

            if (child.depth != node.depth + 1) {
                throw std::invalid_argument(
                    "Child depth must equal parent depth + 1 | Child Depth: " + std::to_string(child.depth) + " | Parent Depth: " + std::to_string(node.depth)
                );
            }
        }

        if (node.player == Player::Chance && !node.children.empty()) {
            double probability_sum = 0.0;

            for (int child_id : node.children) {
                const Node& child = game.node(child_id);

                if (child.chance_prob <= 0.0f) {
                    throw std::invalid_argument(
                        "Chance child must have positive probability."
                    );
                }

                probability_sum += static_cast<double>(child.chance_prob);
            }

            if (probability_sum < 0.99999 || probability_sum > 1.00001) {
                throw std::invalid_argument(
                    "Chance child probabilities must sum to 1."
                );
            }
        }
    }

    for (const InfoSet& infoset : game.infosets) {
        if (infoset.id < 0 || infoset.id >= game.num_infosets()) {
            throw std::invalid_argument("Infoset id out of range.");
        }

        if (!is_real_player(infoset.player)) {
            throw std::invalid_argument(
                "Infoset owner must be P0 or P1."
            );
        }

        if (infoset.key.empty()) {
            throw std::invalid_argument(
                "Infoset key cannot be empty."
            );
        }

        if (infoset.actions.empty()) {
            throw std::invalid_argument(
                "Infoset must have actions."
            );
        }

        if (infoset.actions.size() != infoset.q_indices.size()) {
            throw std::invalid_argument(
                "Infoset actions and q_indices size mismatch."
            );
        }

        for (int local = 0; local < static_cast<int>(infoset.q_indices.size()); ++local) {
            const int q = infoset.q_indices[static_cast<std::size_t>(local)];

            if (q < 0 || q >= game.num_q()) {
                throw std::invalid_argument(
                    "Infoset q index out of range."
                );
            }

            const InfoSetAction& q_entry = game.q_entry(q);

            if (q_entry.q != q) {
                throw std::invalid_argument(
                    "q_entry.q does not match index."
                );
            }

            if (q_entry.infoset != infoset.id) {
                throw std::invalid_argument(
                    "q_entry infoset does not match owner."
                );
            }

            if (q_entry.local_action != local) {
                throw std::invalid_argument(
                    "q_entry local_action does not match position."
                );
            }

            if (q_entry.action != infoset.actions[static_cast<std::size_t>(local)]) {
                throw std::invalid_argument(
                    "q_entry action does not match infoset action."
                );
            }
        }
    }
}

} // namespace poker