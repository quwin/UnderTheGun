#pragma once

#include "poker/board.hpp"
#include "poker/card.hpp"
#include "poker/hand.hpp"

#include <array>
#include <cstdint>
#include <string>

namespace poker {

enum class HandCategory : int {
    HighCard        = 0,
    OnePair         = 1,
    TwoPair         = 2,
    ThreeOfAKind    = 3,
    Straight        = 4,
    Flush           = 5,
    FullHouse       = 6,
    FourOfAKind     = 7,
    StraightFlush   = 8
};

struct HandStrength {
    HandCategory category = HandCategory::HighCard;

    // Tie-break ranks in descending importance.
    //
    // Examples:
    //   High card A K J 9 7:
    //     ranks = {14, 13, 11, 9, 7}
    //
    //   One pair AA K J 9:
    //     ranks = {14, 13, 11, 9, 0}
    //
    //   Two pair AA KK Q:
    //     ranks = {14, 13, 12, 0, 0}
    //
    //   Full house AAA KK:
    //     ranks = {14, 13, 0, 0, 0}
    //
    //   Wheel straight A 2 3 4 5:
    //     ranks = {5, 0, 0, 0, 0}
    std::array<int, 5> ranks{0, 0, 0, 0, 0};

    // Optional compact sortable score.
    //
    // Higher score means stronger hand.
    // This is useful for GPU/array-based comparisons later.
    std::uint32_t score = 0;
};

inline bool operator==(const HandStrength& a, const HandStrength& b) {
    return a.category == b.category && a.ranks == b.ranks;
}

inline bool operator!=(const HandStrength& a, const HandStrength& b) {
    return !(a == b);
}

inline bool operator<(const HandStrength& a, const HandStrength& b) {
    if (static_cast<int>(a.category) != static_cast<int>(b.category)) {
        return static_cast<int>(a.category) < static_cast<int>(b.category);
    }

    return a.ranks < b.ranks;
}

inline bool operator>(const HandStrength& a, const HandStrength& b) {
    return b < a;
}

inline bool operator<=(const HandStrength& a, const HandStrength& b) {
    return !(b < a);
}

inline bool operator>=(const HandStrength& a, const HandStrength& b) {
    return !(a < b);
}

inline int compare_hand_strength(
    const HandStrength& a,
    const HandStrength& b
) {
    if (a > b) {
        return 1;
    }

    if (b > a) {
        return -1;
    }

    return 0;
}

class HandEvaluator {
public:
    HandEvaluator() = default;

    // Evaluates exactly seven cards:
    //   two private cards + five public board cards.
    //
    // Requires:
    //   board.size() == 5
    //   no duplicate cards between hand and board
    HandStrength evaluate_7(
        const HoleCards& hand,
        const Board& board
    ) const;

    HandStrength evaluate_7(
        HandId hand_id,
        const Board& board
    ) const {
        return evaluate_7(hand_from_id(hand_id), board);
    }

    // Convenience comparison:
    //
    // Returns:
    //   +1 if p0 wins
    //    0 if tie
    //   -1 if p1 wins
    int compare_7(
        const HoleCards& p0_hand,
        const HoleCards& p1_hand,
        const Board& board
    ) const;

    int compare_7(
        HandId p0_hand,
        HandId p1_hand,
        const Board& board
    ) const {
        return compare_7(
            hand_from_id(p0_hand),
            hand_from_id(p1_hand),
            board
        );
    }

    bool p0_wins_7(
        const HoleCards& p0_hand,
        const HoleCards& p1_hand,
        const Board& board
    ) const {
        return compare_7(p0_hand, p1_hand, board) > 0;
    }

    bool is_tie_7(
        const HoleCards& p0_hand,
        const HoleCards& p1_hand,
        const Board& board
    ) const {
        return compare_7(p0_hand, p1_hand, board) == 0;
    }
};

// Utility helpers.
std::uint32_t make_hand_score(
    HandCategory category,
    const std::array<int, 5>& ranks
);

std::string to_string(HandCategory category);

std::string to_string(const HandStrength& strength);

} // namespace poker