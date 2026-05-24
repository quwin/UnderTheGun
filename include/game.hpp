#pragma once

#include <cassert>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace poker {

// -----------------------------------------------------------------------------
// Basic game enums
// -----------------------------------------------------------------------------

enum class Player : int {
    Chance   = -1,
    P0       = 0,
    P1       = 1,
    Terminal = 2
};


inline int to_index(Player p) {
    assert(p == Player::P0 || p == Player::P1);
    return static_cast<int>(p);
}

inline bool is_real_player(Player p) {
    return p == Player::P0 || p == Player::P1;
}

inline std::string to_string(Player p) {
    switch (p) {
        case Player::Chance:   return "Chance";
        case Player::P0:       return "P0";
        case Player::P1:       return "P1";
        case Player::Terminal: return "Terminal";
    }
    return "Unknown";
}

// -----------------------------------------------------------------------------
// Game-tree representation
// -----------------------------------------------------------------------------

struct Node {
    int id = -1;
    int parent = -1;
    int depth = 0;

    Player player = Player::Terminal;

    // Infoset id for real-player decision nodes.
    // Use -1 for chance and terminal nodes.
    int infoset = -1;

    // Action taken from parent to reach this node.
    // For root, use Action::Deal or ignore this field.
    Action incoming_action = Action::Deal;

    // Chance probability from parent to this node.
    // For non-chance edges, use 0.0f or 1.0f; the solver should rely on
    // parent player to interpret this.
    float chance_prob = 0.0f;

    Card p0_card = Card::None;
    Card p1_card = Card::None;

    // Public betting history, for example "", "c", "b", "cb".
    std::string history;

    bool terminal = false;

    // Kuhn is two-player zero-sum, so utility_p1 = -utility_p0.
    float utility_p0 = 0.0f;

    std::vector<int> children;
};

struct InfoSet {
    int id = -1;
    Player player = Player::Terminal;
    Card private_card = Card::None;
    std::string public_history;

    // Legal actions at this infoset.
    std::vector<Action> actions;

    // Flat strategy-vector indices corresponding to actions.
    // Same size/order as actions.
    std::vector<int> q_indices;
};

// A flat strategy entry q = (infoset, local action).
struct InfoSetAction {
    int q = -1;
    int infoset = -1;
    int local_action = -1;
    Action action = Action::Deal;
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
        assert(id >= 0 && id < static_cast<int>(nodes.size()));
        return nodes[id];
    }

    Node& node(int id) {
        assert(id >= 0 && id < static_cast<int>(nodes.size()));
        return nodes[id];
    }

    const InfoSet& infoset(int id) const {
        assert(id >= 0 && id < static_cast<int>(infosets.size()));
        return infosets[id];
    }

    InfoSet& infoset(int id) {
        assert(id >= 0 && id < static_cast<int>(infosets.size()));
        return infosets[id];
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
};

// -----------------------------------------------------------------------------
// Helper keys for maps
// -----------------------------------------------------------------------------

inline std::string make_infoset_key(
    Player player,
    Card private_card,
    const std::string& public_history
) {
    return to_string(player) + "|" +
           to_string(private_card) + "|" +
           public_history;
}

struct BuildContext {
    Game game;

    // Maps "player|card|history" to infoset id.
    std::unordered_map<std::string, int> infoset_key_to_id;

    int add_node(Node node) {
        node.id = static_cast<int>(game.nodes.size());
        game.max_depth = std::max(game.max_depth, node.depth);
        game.nodes.push_back(std::move(node));
        return game.nodes.back().id;
    }

    void add_child(int parent, int child) {
        assert(parent >= 0);
        assert(child >= 0);
        game.nodes[parent].children.push_back(child);
        game.nodes[child].parent = parent;
    }

    int get_or_create_infoset(
        Player player,
        Card private_card,
        const std::string& public_history,
        const std::vector<Action>& actions
    ) {
        assert(is_real_player(player));

        const std::string key =
            make_infoset_key(player, private_card, public_history);

        auto it = infoset_key_to_id.find(key);
        if (it != infoset_key_to_id.end()) {
            return it->second;
        }

        InfoSet infoset;
        infoset.id = static_cast<int>(game.infosets.size());
        infoset.player = player;
        infoset.private_card = private_card;
        infoset.public_history = public_history;
        infoset.actions = actions;

        for (int local_action = 0;
             local_action < static_cast<int>(actions.size());
             ++local_action) {
            InfoSetAction q;
            q.q = static_cast<int>(game.q_entries.size());
            q.infoset = infoset.id;
            q.local_action = local_action;
            q.action = actions[local_action];

            infoset.q_indices.push_back(q.q);
            game.q_entries.push_back(q);
        }

        infoset_key_to_id.emplace(key, infoset.id);
        game.infosets.push_back(std::move(infoset));

        return game.infosets.back().id;
    }
};

} // namespace poker