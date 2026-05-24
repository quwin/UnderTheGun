#pragma once

#include "poker/card.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace poker {

using DeckMask = std::uint64_t;

inline DeckMask empty_deck_mask() {
    return 0ULL;
}

inline DeckMask full_deck_mask() {
    // Lower 52 bits set to 1.
    return (1ULL << 52) - 1ULL;
}

inline DeckMask card_mask(CardId card) {
    validate_card(card);
    return 1ULL << static_cast<std::uint64_t>(card);
}

inline bool contains_card(DeckMask mask, CardId card) {
    return (mask & card_mask(card)) != 0ULL;
}

inline DeckMask add_card(DeckMask mask, CardId card) {
    return mask | card_mask(card);
}

inline DeckMask remove_card(DeckMask mask, CardId card) {
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

inline std::vector<CardId> cards_from_mask(DeckMask mask) {
    validate_deck_mask(mask);

    std::vector<CardId> cards;
    cards.reserve(static_cast<std::size_t>(popcount(mask)));

    for (int card = 0; card < 52; ++card) {
        const CardId card_id = static_cast<CardId>(card);

        if (contains_card(mask, card_id)) {
            cards.push_back(card_id);
        }
    }

    return cards;
}

inline DeckMask mask_from_cards(const std::vector<CardId>& cards) {
    DeckMask mask = empty_deck_mask();

    for (CardId card : cards) {
        if (contains_card(mask, card)) {
            throw std::invalid_argument("Duplicate card in card collection.");
        }

        mask = add_card(mask, card);
    }

    return mask;
}

template <std::size_t N>
inline DeckMask mask_from_array(const std::array<CardId, N>& cards) {
    DeckMask mask = empty_deck_mask();

    for (CardId card : cards) {
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

inline std::vector<CardId> remaining_card_list(DeckMask dead_cards) {
    return cards_from_mask(remaining_cards(dead_cards));
}

} // namespace poker
