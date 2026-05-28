#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "../../external/PokerHandEvaluator/cpp/include/phevaluator/card.h"

namespace poker {
using DeckMask = std::uint64_t;

inline DeckMask empty_deck_mask() {
    return 0ULL;
}

inline DeckMask full_deck_mask() {
    return (1ULL << 52) - 1ULL;
}

inline int card_index(const phevaluator::Card& card) {
    const int id = static_cast<int>(card);

    if (id < 0 || id >= 52) {
        throw std::invalid_argument("phevaluator::Card id must be in range 0..51.");
    }

    return id;
}

inline void validate_card(const phevaluator::Card& card) {
    (void)card_index(card);
}

inline DeckMask card_mask(const phevaluator::Card& card) {
    return 1ULL << static_cast<std::uint64_t>(card_index(card));
}

inline bool contains_card(DeckMask mask, const phevaluator::Card& card) {
    return (mask & card_mask(card)) != 0ULL;
}

inline DeckMask add_card(DeckMask mask, const phevaluator::Card& card) {
    return mask | card_mask(card);
}

inline DeckMask remove_card(DeckMask mask, const phevaluator::Card& card) {
    return mask & ~card_mask(card);
}

inline bool masks_overlap(DeckMask a, DeckMask b) {
    return (a & b) != 0ULL;
}

inline DeckMask mask_union(DeckMask a, DeckMask b) {
    return a | b;
}

inline DeckMask mask_intersection(DeckMask a, DeckMask b) {
    return a & b;
}

inline DeckMask mask_difference(DeckMask a, DeckMask b) {
    return a & ~b;
}

inline int popcount(DeckMask mask) {
#if defined(__GNUG__) || defined(__clang__)
    return __builtin_popcountll(mask);
#else
    int count = 0;
    while (mask != 0ULL) {
        mask &= (mask - 1ULL);
        ++count;
    }
    return count;
#endif
}

inline bool is_subset(DeckMask subset, DeckMask superset) {
    return (subset & ~superset) == 0ULL;
}

inline bool is_valid_deck_mask(DeckMask mask) {
    return (mask & ~full_deck_mask()) == 0ULL;
}

inline void validate_deck_mask(DeckMask mask) {
    if (!is_valid_deck_mask(mask)) {
        throw std::invalid_argument("DeckMask contains bits outside the 52-card deck.");
    }
}

inline std::vector<phevaluator::Card> cards_from_mask(DeckMask mask) {
    validate_deck_mask(mask);

    std::vector<phevaluator::Card> cards;
    cards.reserve(static_cast<std::size_t>(popcount(mask)));

    for (int id = 0; id < 52; ++id) {
        const phevaluator::Card card{id};

        if (contains_card(mask, card)) {
            cards.push_back(card);
        }
    }

    return cards;
}

inline DeckMask mask_from_cards(const std::vector<phevaluator::Card>& cards) {
    DeckMask mask = empty_deck_mask();

    for (const phevaluator::Card& card : cards) {
        if (contains_card(mask, card)) {
            throw std::invalid_argument("Duplicate card in card collection.");
        }

        mask = add_card(mask, card);
    }

    return mask;
}

template <std::size_t N>
inline DeckMask mask_from_array(const std::array<phevaluator::Card, N>& cards) {
    DeckMask mask = empty_deck_mask();

    for (const phevaluator::Card& card : cards) {
        if (contains_card(mask, card)) {
            throw std::invalid_argument("Duplicate card in card array.");
        }

        mask = add_card(mask, card);
    }

    return mask;
}

inline DeckMask remaining_cards(DeckMask dead_cards) {
    validate_deck_mask(dead_cards);
    return full_deck_mask() & ~dead_cards;
}

inline std::vector<phevaluator::Card> remaining_card_list(DeckMask dead_cards) {
    return cards_from_mask(remaining_cards(dead_cards));
}

} // namespace poker