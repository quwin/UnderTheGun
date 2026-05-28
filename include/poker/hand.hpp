#pragma once

#include "poker/card.hpp"
#include "poker/deck_mask.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace poker {

using HandId = std::uint16_t;

constexpr int kNumCards = 52;
constexpr int kNumHands = 1326;

struct HoleCards {
    CardId a = 0;
    CardId b = 1;

    HoleCards() = default;

    HoleCards(CardId first, CardId second)
        : a(first), b(second) {
        validate();
        canonicalize();
    }

    void validate() const {
        validate_card(a);
        validate_card(b);

        if (a == b) {
            throw std::invalid_argument("HoleCards must contain two distinct cards.");
        }
    }

    void canonicalize() {
        if (b < a) {
            const CardId tmp = a;
            a = b;
            b = tmp;
        }
    }

    DeckMask mask() const {
        DeckMask result = empty_deck_mask();
        result = add_card(result, a);
        result = add_card(result, b);
        return result;
    }

    bool contains(CardId card) const {
        return a == card || b == card;
    }
};

inline bool operator==(const HoleCards& x, const HoleCards& y) {
    return x.a == y.a && x.b == y.b;
}

inline bool operator!=(const HoleCards& x, const HoleCards& y) {
    return !(x == y);
}

inline bool operator<(const HoleCards& x, const HoleCards& y) {
    if (x.a != y.a) {
        return x.a < y.a;
    }

    return x.b < y.b;
}

inline HoleCards make_hole_cards(CardId a, CardId b) {
    return HoleCards{a, b};
}

inline DeckMask hand_mask(const HoleCards& hand) {
    return hand.mask();
}

inline bool hands_overlap(const HoleCards& a, const HoleCards& b) {
    return masks_overlap(a.mask(), b.mask());
}

inline bool hand_overlaps_mask(const HoleCards& hand, DeckMask dead_cards) {
    return masks_overlap(hand.mask(), dead_cards);
}

namespace detail {

struct HandLookupTables {
    std::array<HoleCards, kNumHands> id_to_hand{};
    std::array<std::array<HandId, kNumCards>, kNumCards> hand_to_id{};

    HandLookupTables() {
        for (int i = 0; i < kNumCards; ++i) {
            for (int j = 0; j < kNumCards; ++j) {
                hand_to_id[i][j] = invalid_hand_id();
            }
        }

        HandId id = 0;

        for (int first = 0; first < kNumCards; ++first) {
            for (int second = first + 1; second < kNumCards; ++second) {
                const CardId a = static_cast<CardId>(first);
                const CardId b = static_cast<CardId>(second);

                id_to_hand[id] = HoleCards{a, b};
                hand_to_id[first][second] = id;
                hand_to_id[second][first] = id;

                ++id;
            }
        }

        if (id != kNumHands) {
            throw std::logic_error("Hand lookup table construction failed.");
        }
    }

    static constexpr HandId invalid_hand_id() {
        return static_cast<HandId>(0xFFFF);
    }
};

inline const HandLookupTables& hand_lookup_tables() {
    static const HandLookupTables tables;
    return tables;
}

} // namespace detail

inline HandId invalid_hand_id() {
    return detail::HandLookupTables::invalid_hand_id();
}

inline bool is_valid_hand_id(HandId hand_id) {
    return hand_id < kNumHands;
}

inline void validate_hand_id(HandId hand_id) {
    if (!is_valid_hand_id(hand_id)) {
        throw std::invalid_argument("HandId must be in range 0..1325.");
    }
}

inline HandId make_hand(CardId a, CardId b) {
    validate_card(a);
    validate_card(b);

    if (a == b) {
        throw std::invalid_argument("Cannot make HandId from duplicate cards.");
    }

    const auto& tables = detail::hand_lookup_tables();
    const HandId id = tables.hand_to_id[a][b];

    if (!is_valid_hand_id(id)) {
        throw std::logic_error("HandId lookup failed.");
    }

    return id;
}

inline HandId make_hand(const HoleCards& hand) {
    return make_hand(hand.a, hand.b);
}

inline HoleCards hand_from_id(HandId hand_id) {
    validate_hand_id(hand_id);
    return detail::hand_lookup_tables().id_to_hand[hand_id];
}

inline CardId first_card(HandId hand_id) {
    return hand_from_id(hand_id).a;
}

inline CardId second_card(HandId hand_id) {
    return hand_from_id(hand_id).b;
}

inline DeckMask hand_mask(HandId hand_id) {
    return hand_from_id(hand_id).mask();
}

inline bool hand_contains(HandId hand_id, CardId card) {
    return hand_from_id(hand_id).contains(card);
}

inline bool hands_overlap(HandId a, HandId b) {
    return masks_overlap(hand_mask(a), hand_mask(b));
}

inline bool hand_overlaps_mask(HandId hand_id, DeckMask dead_cards) {
    return masks_overlap(hand_mask(hand_id), dead_cards);
}

inline std::vector<HandId> all_hand_ids() {
    std::vector<HandId> result;
    result.reserve(kNumHands);

    for (int id = 0; id < kNumHands; ++id) {
        result.push_back(static_cast<HandId>(id));
    }

    return result;
}

inline std::array<HoleCards, kNumHands> all_hole_cards() {
    return detail::hand_lookup_tables().id_to_hand;
}

inline std::vector<HandId> legal_hands_excluding(DeckMask dead_cards) {
    validate_deck_mask(dead_cards);

    std::vector<HandId> result;
    result.reserve(kNumHands);

    for (int id = 0; id < kNumHands; ++id) {
        const HandId hand_id = static_cast<HandId>(id);

        if (!hand_overlaps_mask(hand_id, dead_cards)) {
            result.push_back(hand_id);
        }
    }

    return result;
}

inline std::string to_string(const HoleCards& hand) {
    return to_string(hand.a) + to_string(hand.b);
}

inline std::string to_string(HandId hand_id) {
    return to_string(hand_from_id(hand_id));
}

inline HoleCards parse_hole_cards(const std::string& text) {
    if (text.size() != 4) {
        throw std::invalid_argument("Hole-card string must have length 4, e.g. AsKd.");
    }

    const CardId a = parse_card(text.substr(0, 2));
    const CardId b = parse_card(text.substr(2, 2));

    return HoleCards{a, b};
}

inline HandId parse_hand_id(const std::string& text) {
    return make_hand(parse_hole_cards(text));
}

} // namespace poker