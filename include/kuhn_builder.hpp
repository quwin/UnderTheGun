#pragma once

#include "game.hpp"
#include "holdem/action.hpp"
#include "../external/PokerHandEvaluator/cpp/include/phevaluator/card.h"
#include <string>
#include <vector>

namespace poker {

enum class KuhnAction : int {
    Check = 0,
    Call = 1,
    Raise = 2,
    Fold = 3,
};

struct KuhnConfig {
    // Standard Kuhn uses J, Q, K.
    std::vector<phevaluator::Card> deck = {
        phevaluator::Card("Js"),
        phevaluator::Card("Qs"),
        phevaluator::Card("Ks")
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
        phevaluator::Card p0_card,
        phevaluator::Card p1_card,
        const std::string& history
    ) const;

    // Adds a terminal child node with the correct payoff.
    int add_terminal_node(
        BuildContext& ctx,
        int parent_id,
        holdem::ActionType incoming_action,
        phevaluator::Card p0_card,
        phevaluator::Card p1_card,
        const std::string& terminal_history
    ) const;

    // Adds a nonterminal decision node.
    int add_decision_node(
        BuildContext& ctx,
        int parent_id,
        holdem::ActionType incoming_action,
        Player player_to_act,
        phevaluator::Card p0_card,
        phevaluator::Card p1_card,
        const std::string& history
    ) const;

    // Kuhn rule helpers.
    std::vector<holdem::ActionType> legal_actions(const std::string& history) const;

    bool is_terminal_history(const std::string& history) const;

    float terminal_utility_p0(
        phevaluator::Card p0_card,
        phevaluator::Card p1_card,
        const std::string& terminal_history
    ) const;

    Player next_player_after(
        Player current_player,
        const std::string& new_history
    ) const;

    phevaluator::Card private_card_for(Player player, phevaluator::Card p0_card, phevaluator::Card p1_card) const;

    std::string append_action(
        const std::string& history,
        holdem::ActionType action
    ) const;
};

// Optional free-function convenience wrapper.
Game build_kuhn_game();

Game build_kuhn_game(const KuhnConfig& config);

} // namespace poker