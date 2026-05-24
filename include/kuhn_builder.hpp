#pragma once

#include "game.hpp"

#include <string>
#include <vector>

namespace poker {

struct KuhnConfig {
    // Standard Kuhn uses J, Q, K.
    std::vector<Card> deck = {
        Card::Jack,
        Card::Queen,
        Card::King
    };

    // Standard Kuhn has ante = 1 and one bet size = 1.
    // Using floats keeps the payoff code simple and extensible.
    float ante = 1.0f;
    float bet_size = 1.0f;

    // Keep this false for standard two-player Kuhn.
    bool allow_redeal_same_card = false;
};

class KuhnGameBuilder {
public:
    KuhnGameBuilder();
    explicit KuhnGameBuilder(KuhnConfig config);

    // Build and return the full explicit Kuhn game tree.
    Game build() const;

private:
    KuhnConfig config_;

    // Creates the root chance node and all private-card deal branches.
    void add_chance_deals(BuildContext& ctx, int root_id) const;

    // Recursively adds betting/checking/calling/folding branches.
    void add_betting_subtree(
        BuildContext& ctx,
        int parent_id,
        Player player_to_act,
        Card p0_card,
        Card p1_card,
        const std::string& history
    ) const;

    // Adds a terminal child node with the correct payoff.
    int add_terminal_node(
        BuildContext& ctx,
        int parent_id,
        Action incoming_action,
        Card p0_card,
        Card p1_card,
        const std::string& terminal_history
    ) const;

    // Adds a nonterminal decision node.
    int add_decision_node(
        BuildContext& ctx,
        int parent_id,
        Action incoming_action,
        Player player_to_act,
        Card p0_card,
        Card p1_card,
        const std::string& history
    ) const;

    // Kuhn rule helpers.
    std::vector<Action> legal_actions(const std::string& history) const;

    bool is_terminal_history(const std::string& history) const;

    float terminal_utility_p0(
        Card p0_card,
        Card p1_card,
        const std::string& terminal_history
    ) const;

    Player next_player_after(
        Player current_player,
        const std::string& new_history
    ) const;

    Card private_card_for(Player player, Card p0_card, Card p1_card) const;

    std::string append_action(
        const std::string& history,
        Action action
    ) const;
};

// Optional free-function convenience wrapper.
Game build_kuhn_game();

Game build_kuhn_game(const KuhnConfig& config);

} // namespace poker