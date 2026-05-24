#pragma once

#include "poker/board.hpp"
#include "poker/deck_mask.hpp"
#include "poker/hand.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace poker {

struct LegalHandPair {
    HandId p0_hand = invalid_hand_id();
    HandId p1_hand = invalid_hand_id();
    float probability = 0.0f;
};

class Range {
public:
    Range() {
        clear();
    }

    void clear() {
        weights_.fill(0.0f);
        nonzero_hands_.clear();
    }

    float weight(HandId hand_id) const {
        validate_hand_id(hand_id);
        return weights_[hand_id];
    }

    void set_weight(HandId hand_id, float weight) {
        validate_hand_id(hand_id);

        if (!std::isfinite(weight)) {
            throw std::invalid_argument("Range weight must be finite.");
        }

        if (weight < 0.0f) {
            throw std::invalid_argument("Range weight must be nonnegative.");
        }

        const bool was_positive = weights_[hand_id] > 0.0f;
        const bool now_positive = weight > 0.0f;

        weights_[hand_id] = weight;

        if (!was_positive && now_positive) {
            nonzero_hands_.push_back(hand_id);
        } else if (was_positive && !now_positive) {
            erase_nonzero_hand(hand_id);
        }
    }

    bool contains(HandId hand_id) const {
        return weight(hand_id) > 0.0f;
    }

    int nonzero_count() const {
        return static_cast<int>(nonzero_hands_.size());
    }

    bool empty() const {
        return nonzero_hands_.empty();
    }

    const std::vector<HandId>& hands_with_positive_weight() const {
        return nonzero_hands_;
    }

    double total_weight() const {
        double total = 0.0;

        for (HandId hand_id : nonzero_hands_) {
            total += static_cast<double>(weights_[hand_id]);
        }

        return total;
    }

    Range normalized() const {
        const double total = total_weight();

        if (total <= 0.0) {
            throw std::invalid_argument("Cannot normalize an empty range.");
        }

        Range result;

        for (HandId hand_id : nonzero_hands_) {
            result.set_weight(
                hand_id,
                static_cast<float>(
                    static_cast<double>(weights_[hand_id]) / total
                )
            );
        }

        return result;
    }

    Range remove_blocked(DeckMask dead_cards) const {
        validate_deck_mask(dead_cards);

        Range result;

        for (HandId hand_id : nonzero_hands_) {
            if (hand_overlaps_mask(hand_id, dead_cards)) {
                continue;
            }

            result.set_weight(hand_id, weights_[hand_id]);
        }

        return result;
    }

    Range remove_blocked_by_board(const Board& board) const {
        return remove_blocked(board_mask(board));
    }

    std::vector<HandId> legal_hands(DeckMask dead_cards) const {
        validate_deck_mask(dead_cards);

        std::vector<HandId> result;
        result.reserve(nonzero_hands_.size());

        for (HandId hand_id : nonzero_hands_) {
            if (!hand_overlaps_mask(hand_id, dead_cards)) {
                result.push_back(hand_id);
            }
        }

        return result;
    }

private:
    std::array<float, kNumHands> weights_{};
    std::vector<HandId> nonzero_hands_;

    void erase_nonzero_hand(HandId hand_id) {
        const auto it = std::find(
            nonzero_hands_.begin(),
            nonzero_hands_.end(),
            hand_id
        );

        if (it != nonzero_hands_.end()) {
            nonzero_hands_.erase(it);
        }
    }
};

inline Range make_empty_range() {
    Range range;
    range.clear();
    return range;
}

inline Range make_full_combo_range() {
    Range range;
    range.clear();

    for (int id = 0; id < kNumHands; ++id) {
        range.set_weight(static_cast<HandId>(id), 1.0f);
    }

    return range;
}

inline Range make_range(
    std::initializer_list<std::pair<HandId, float>> entries
) {
    Range range;
    range.clear();

    for (const auto& [hand_id, weight] : entries) {
        range.set_weight(hand_id, weight);
    }

    return range;
}

inline std::vector<LegalHandPair> legal_hand_pairs(
    const Range& p0_range,
    const Range& p1_range,
    DeckMask dead_cards
) {
    validate_deck_mask(dead_cards);

    std::vector<LegalHandPair> pairs;

    double total_pair_weight = 0.0;

    for (HandId p0_hand : p0_range.hands_with_positive_weight()) {
        if (hand_overlaps_mask(p0_hand, dead_cards)) {
            continue;
        }

        const DeckMask p0_mask = hand_mask(p0_hand);

        for (HandId p1_hand : p1_range.hands_with_positive_weight()) {
            if (hand_overlaps_mask(p1_hand, dead_cards)) {
                continue;
            }

            const DeckMask p1_mask = hand_mask(p1_hand);

            if (masks_overlap(p0_mask, p1_mask)) {
                continue;
            }

            const double pair_weight =
                static_cast<double>(p0_range.weight(p0_hand)) *
                static_cast<double>(p1_range.weight(p1_hand));

            if (pair_weight <= 0.0) {
                continue;
            }

            pairs.push_back(
                LegalHandPair{
                    p0_hand,
                    p1_hand,
                    static_cast<float>(pair_weight)
                }
            );

            total_pair_weight += pair_weight;
        }
    }

    if (pairs.empty() || total_pair_weight <= 0.0) {
        throw std::invalid_argument(
            "No legal private hand pairs remain after card removal."
        );
    }

    for (LegalHandPair& pair : pairs) {
        pair.probability = static_cast<float>(
            static_cast<double>(pair.probability) / total_pair_weight
        );
    }

    return pairs;
}

inline std::vector<LegalHandPair> legal_hand_pairs(
    const Range& p0_range,
    const Range& p1_range,
    const Board& board
) {
    return legal_hand_pairs(
        p0_range,
        p1_range,
        board_mask(board)
    );
}

} // namespace poker