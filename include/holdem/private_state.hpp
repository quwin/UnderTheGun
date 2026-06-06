#pragma once

#include "game.hpp"

#include "poker/deck_mask.hpp"
#include "poker/hand.hpp"

#include <stdexcept>
#include <string>

#include "poker/board.hpp"

namespace poker::holdem {

struct PrivateState {
    HoleCards p0_hand;
    HoleCards p1_hand;

    PrivateState() = default;

    PrivateState(
        const HoleCards p0,
        const HoleCards p1
    )
        : p0_hand(p0),
          p1_hand(p1) {
        validate();
    }

    void validate() const {
        p0_hand.validate();
        p1_hand.validate();

        if (hands_overlap(p0_hand, p1_hand)) {
            throw std::invalid_argument(
                "PrivateState hands cannot share cards."
            );
        }
    }

    [[nodiscard]] HoleCards hand_for(const Player player) const {
        switch (player) {
            case Player::P0:
                return p0_hand;

            case Player::P1:
                return p1_hand;

            default:
                throw std::invalid_argument(
                    "PrivateState::hand_for requires P0 or P1."
                );
        }
    }

    [[nodiscard]] HoleCards opponent_hand_for(Player player) const {
        switch (player) {
            case Player::P0:
                return p1_hand;

            case Player::P1:
                return p0_hand;

            default:
                throw std::invalid_argument(
                    "PrivateState::opponent_hand_for requires P0 or P1."
                );
        }
    }

    [[nodiscard]] DeckMask p0_mask() const {
        return hand_mask(p0_hand);
    }

    [[nodiscard]] DeckMask p1_mask() const {
        return hand_mask(p1_hand);
    }

    [[nodiscard]] DeckMask mask() const {
        return p0_mask() | p1_mask();
    }

    [[nodiscard]] bool contains(phevaluator::Card card) const {
        return p0_hand.contains(card) || p1_hand.contains(card);
    }

    [[nodiscard]] bool overlaps_mask(DeckMask dead_cards) const {
        return masks_overlap(mask(), dead_cards);
    }

    [[nodiscard]] bool overlaps_board(const Board& board) const {
        return overlaps_mask(board_mask(board));
    }
};

inline bool operator==(const PrivateState& a, const PrivateState& b) {
    return a.p0_hand == b.p0_hand &&
           a.p1_hand == b.p1_hand;
}

inline bool operator!=(const PrivateState& a, const PrivateState& b) {
    return !(a == b);
}

inline PrivateState make_private_state(
    const HoleCards p0_hand,
    const HoleCards p1_hand
) {
    return PrivateState{p0_hand, p1_hand};
}

inline PrivateState make_private_state(
    const HandId p0_hand,
    const HandId p1_hand
) {
    return PrivateState{
        hand_from_id(p0_hand),
        hand_from_id(p1_hand)
    };
}

inline std::string to_string(const PrivateState& state) {
    return "p0=" + poker::to_string(state.p0_hand) +
           "|p1=" + poker::to_string(state.p1_hand);
}

} // namespace poker::holdem