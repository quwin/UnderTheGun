#pragma once

#include "game.hpp"
#include <cmath>

namespace poker {

// Public edges describe the board-only distribution. Conditioning on an exact
// private pair removes its four cards and preserves the relative legal weights.
inline std::vector<double> chance_probabilities_for_pair(
    const Game& game, const PublicNode& node, int pair
) {
    const auto private_cards =
        hand_from_id(game.p0_hands.hands.at(game.hand_pairs.p0_index.at(pair))).mask() |
        hand_from_id(game.p1_hands.hands.at(game.hand_pairs.p1_index.at(pair))).mask();
    std::vector<double> probabilities(node.edge_count, 0.0);
    DeckMask seen_cards = 0;
    double public_mass = 0.0, legal_mass = 0.0;
    for (int local = 0; local < node.edge_count; ++local) {
        const auto& edge = game.edge(node.first_edge + local);
        if (!std::isfinite(edge.chance_prob) || edge.chance_prob <= 0.0f ||
            edge.chance_prob > 1.0f || edge.public_card < 0 || edge.public_card >= 52)
            throw std::invalid_argument("Invalid public chance edge.");
        public_mass += edge.chance_prob;
        const auto bit = DeckMask{1} << edge.public_card;
        if (seen_cards & bit)
            throw std::invalid_argument("Duplicate card in public chance outcomes.");
        seen_cards |= bit;
        if (!(private_cards & bit)) {
            probabilities[local] = edge.chance_prob;
            legal_mass += edge.chance_prob;
        }
    }
    if (std::abs(public_mass - 1.0) > 1e-5)
        throw std::invalid_argument("Public chance probabilities must sum to one.");
    // An entirely blocked synthetic subtree is unreachable for this pair.
    if (legal_mass > 0.0)
        for (auto& probability : probabilities) probability /= legal_mass;
    return probabilities;
}

} // namespace poker
