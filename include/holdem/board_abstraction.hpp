#pragma once

#include "poker/board.hpp"
#include "../../external/PokerHandEvaluator/cpp/include/phevaluator/card.h"
#include "poker/deck_mask.hpp"

#include <memory>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <iostream>
#include <ostream>
#include <unordered_map>
#include "poker/range.hpp"

namespace poker::holdem {

using BoardBucketId = int;

constexpr BoardBucketId kInvalidBoardBucket = -1;

// One possible public-card transition.
//
// Exact mode:
//   board      = exact resulting board
//   bucket_id  = kInvalidBoardBucket
//
// Abstract mode:
//   board      = representative board or exact sampled board
//   bucket_id  = abstraction bucket
//
// probability must be normalized among all returned outcomes.
struct BoardTransition {
    Board board;
    BoardBucketId bucket_id = kInvalidBoardBucket;
    float probability = 0.0f;
};

inline std::string canonical_board_key(const Board& board) {
        std::array<int, 4> suit_map{};
        suit_map.fill(-1);

        int next_suit = 0;

        std::vector<std::pair<int, int>> cards;
        cards.reserve(board.cards.size());

        for (phevaluator::Card c : board.cards) {
            const int rank = phevaluator::suitMap.at(c.describeRank());
            const int suit = phevaluator::suitMap.at(c.describeSuit());
            if (suit_map[suit] == -1) {
                suit_map[suit] = next_suit++;
            }
            cards.emplace_back(rank, suit_map[suit]);
        }
        std::ranges::sort(cards);
        std::string key;
        for (auto [rank, canonical_suit] : cards) {
            key += std::to_string(rank);
            key += ":";
            key += std::to_string(canonical_suit);
            key += "|";
        }
        return key;
    }

class BoardAbstraction {
public:
    virtual ~BoardAbstraction() = default;

    // Returns possible public-board transitions after a betting round closes.
    //
    // Examples:
    //
    // Flop state:
    //   current board has 3 cards
    //   next street is Turn
    //   exact mode returns one transition for every legal turn card
    //
    // Turn state:
    //   current board has 4 cards
    //   next street is River
    //   exact mode returns one transition for every legal river card
    //
    // River state:
    //   no next public-card transitions
    [[nodiscard]] virtual std::vector<BoardTransition> next_board_transitions(
        const Board& current_board,
        DeckMask dead_cards
    ) const = 0;

    // Returns a public-board bucket for an already-existing board.
    //
    // Exact mode can return kInvalidBoardBucket.
    // Abstract mode should return a stable bucket id.
    [[nodiscard]] virtual BoardBucketId bucket_for(
        const Board& board
    ) const = 0;

    [[nodiscard]] virtual bool is_exact() const = 0;
};

// -----------------------------------------------------------------------------
// Exact board abstraction
// -----------------------------------------------------------------------------
//
// This is the default implementation. It does not abstract boards.
// It enumerates every legal public card transition exactly.

class ExactBoardAbstraction final : public BoardAbstraction {
public:
    [[nodiscard]] std::vector<BoardTransition> next_board_transitions(
        const Board& current_board,
        const DeckMask dead_cards
    ) const override {
        current_board.validate();
        validate_deck_mask(dead_cards);
        if (current_board.is_river()) {
            return {};
        }
        const DeckMask unavailable = dead_cards | board_mask(current_board);
        validate_deck_mask(unavailable);
        const std::vector<phevaluator::Card> available_cards = cards_from_mask(remaining_cards(unavailable));
        if (available_cards.empty()) {
            throw std::invalid_argument(
                "No legal public cards remain for board transition."
            );
        }
        const float probability = 1.0f / static_cast<float>(available_cards.size());
        std::vector<BoardTransition> transitions;
        transitions.reserve(available_cards.size());
        for (const phevaluator::Card card : available_cards) {
            const Board next_board = current_board.with_added_card(card);
            transitions.push_back(
                BoardTransition{
                    next_board,
                    kInvalidBoardBucket,
                    probability
                }
            );
        }

        return transitions;
    }

    [[nodiscard]] BoardBucketId bucket_for(
        const Board& board
    ) const override {
        return kInvalidBoardBucket;
    }

    [[nodiscard]] bool is_exact() const override {
        return true;
    }

};

struct IsoPreflopCombo {
    int hand1 = 0;
    int hand2 = 0;
    float probability = 0.0f;
};

struct IsoBoardTransitionGroup {
    Board representative_board;
    int count = 0;
    bool initialized = false;
};

class IsomorphicBoardAbstraction final : public BoardAbstraction {
public:
    IsomorphicBoardAbstraction(Range p0_range,Range p1_range):
        p0_range_(std::move(p0_range)),
        p1_range_(std::move(p1_range)) {}

    [[nodiscard]] std::vector<BoardTransition> next_board_transitions(
        const Board& current_board,
        DeckMask dead_cards
    ) const override {
        current_board.validate();
        validate_deck_mask(dead_cards);
        if (current_board.is_river()) {
            return {};
        }
        const DeckMask unavailable = dead_cards | board_mask(current_board);

        const std::vector<phevaluator::Card> available_cards = cards_from_mask(remaining_cards(unavailable));
        if (available_cards.empty()) {
            throw std::invalid_argument(
                "No legal public cards remain for isomorphic transition."
            );
        }

        const std::vector<IsoPreflopCombo> p0_iso_combos = iso_combos_from_range(p0_range_);

        const std::vector<IsoPreflopCombo> p1_iso_combos = iso_combos_from_range(p1_range_);

        const std::vector<int> iso_board = iso_board_cards(current_board);

        const std::uint64_t iso_dead_mask = unavailable;

        const IsoData iso =
            compute_isomorphism(
                p0_iso_combos,
                p1_iso_combos,
                iso_board,
                iso_dead_mask
            );

        std::unordered_map<std::string, IsoBoardTransitionGroup> groups;
        for (phevaluator::Card card : available_cards) {
            if (const int iso_card = card; iso.skip_cards[iso_card]) {
                const int representative_iso_card = iso.representative_card[iso_card];
                const auto representative_card = phevaluator::Card(representative_iso_card);
                const Board representative_board = current_board.with_added_card(representative_card);

                const std::string key = canonical_transition_key(representative_board);

                auto&[rep_board, count, initialized] = groups[key];
                if (!initialized) {
                    rep_board = representative_board;
                    initialized = true;
                }
                ++count;
                continue;
            }
            const Board next_board = current_board.with_added_card(card);
            const std::string key = canonical_transition_key(next_board);
            auto& group = groups[key];
            if (!group.initialized) {
                group.representative_board = next_board;
                group.initialized = true;
            }
            ++group.count;
        }

        std::vector<BoardTransition> transitions;
        transitions.reserve(groups.size());
        const auto denominator = static_cast<float>(available_cards.size());

        for (const auto& [key, group] : groups) {
            transitions.push_back(
                BoardTransition{
                    group.representative_board,
                    bucket_for(group.representative_board),
                    static_cast<float>(group.count) / denominator
                }
            );
        }
        return transitions;
    }

    [[nodiscard]] BoardBucketId bucket_for(const Board& board) const override {
        board.validate();

        const std::string key = canonical_transition_key(board);

        return static_cast<BoardBucketId>(stable_hash_to_positive_int(key));
    }
    [[nodiscard]] bool is_exact() const override {
        return false;
    }

private:
    Range p0_range_;
    Range p1_range_;

    struct IsoData {
        std::array<bool, 52> skip_cards{};
        std::array<int, 52> representative_card{};
        bool has_isomorphism = false;

        IsoData() {
            skip_cards.fill(false);
            representative_card.fill(-1);
        }
    };

    static IsoData compute_isomorphism(
        const std::vector<IsoPreflopCombo>& p0_combos,
        const std::vector<IsoPreflopCombo>& p1_combos,
        const std::vector<int>& board,
        const std::uint64_t board_mask
    ) {
        IsoData result;
        std::array<std::uint16_t, 4> board_rankset{};
        board_rankset.fill(0);
        for (const int card : board) {
            const int rank = card >> 2;
            const int suit = card & 3;
            board_rankset[suit] |= static_cast<std::uint16_t>(1u << rank);
        }
        std::array<int, 4> isomorphic_suit{};
        isomorphic_suit.fill(-1);
        for (int suit1 = 1; suit1 < 4; ++suit1) {
            for (int suit2 = 0; suit2 < suit1; ++suit2) {
                if (board_rankset[suit1] != board_rankset[suit2]) {
                    continue;
                }
                if (!is_suit_isomorphic(p0_combos, suit1, suit2)) {
                    continue;
                }
                if (!is_suit_isomorphic(p1_combos, suit1, suit2)) {
                    continue;
                }
                isomorphic_suit[suit1] = suit2;
                result.has_isomorphism = true;
                break;
            }
        }
        if (!result.has_isomorphism) {
            return result;
        }
        for (int card = 0; card < 52; ++card) {
            if ((board_mask & (1ULL << card)) != 0ULL) {
                continue;
            }
            const int suit = card & 3;
            if (isomorphic_suit[suit] < 0) {
                continue;
            }
            const int replacement_suit = isomorphic_suit[suit];
            const int replacement_card =
                card - suit + replacement_suit;
            if ((board_mask & (1ULL << replacement_card)) != 0ULL) {
                continue;
            }
            result.skip_cards[card] = true;
            result.representative_card[card] = replacement_card;
        }
        return result;
    }

    static bool is_suit_isomorphic(
        const std::vector<IsoPreflopCombo>& combos,
        int suit1,
        int suit2
    ) {
        auto swap_suit = [suit1, suit2](int card) -> int {
            const int suit = card & 3;
            const int rank = card >> 2;
            if (suit == suit1) {
                return (rank << 2) | suit2;
            }
            if (suit == suit2) {
                return (rank << 2) | suit1;
            }
            return card;
        };
        std::array<float, 52 * 52> weights{};
        std::array<float, 52 * 52> swapped_weights{};
        weights.fill(0.0f);
        swapped_weights.fill(0.0f);

        for (const auto&[hand1, hand2, probability] : combos) {
            int c1 = hand1;
            int c2 = hand2;
            if (c1 > c2) {
                std::swap(c1, c2);
            }
            weights[c1 * 52 + c2] = probability;
            int s1 = swap_suit(hand1);
            int s2 = swap_suit(hand2);
            if (s1 > s2) {
                std::swap(s1, s2);
            }
            swapped_weights[s1 * 52 + s2] = probability;
        }
        for (int i = 0; i < 52 * 52; ++i) {
            if (std::abs(weights[i] - swapped_weights[i]) > 1e-6f) {
                return false;
            }
        }

        return true;
    }

    static std::vector<IsoPreflopCombo> iso_combos_from_range(
        const Range& range
    ) {
        std::vector<IsoPreflopCombo> combos;
        combos.reserve(range.hands_with_positive_weight().size());

        for (const HandId hand_id : range.hands_with_positive_weight()) {
            const HoleCards hand = hand_from_id(hand_id);

            combos.push_back(
                IsoPreflopCombo{
                    hand.a,
                    hand.b,
                    range.weight(hand_id)
                }
            );
        }

        return combos;
    }

    static std::vector<int> iso_board_cards(const Board& board) {
        std::vector<int> result;
        result.reserve(board.cards.size());
        for (const phevaluator::Card card : board.cards) {
            result.push_back(card);
        }
        return result;
    }

    static std::string canonical_transition_key(const Board& board) {
        std::vector<int> iso_cards;
        iso_cards.reserve(board.cards.size());
        for (const phevaluator::Card card : board.cards) {
            iso_cards.push_back(card);
        }
        std::ranges::sort(iso_cards);

        std::string key;

        for (const int card : iso_cards) {
            key += std::to_string(card >> 2);
            key += ":";
            key += std::to_string(card & 3);
            key += "|";
        }
        return key;
    }

    static int stable_hash_to_positive_int(const std::string& key) {
        std::uint32_t hash = 2166136261u;

        for (const unsigned char c : key) {
            hash ^= c;
            hash *= 16777619u;
        }

        return static_cast<int>(hash & 0x7fffffff);
    }
};
// -----------------------------------------------------------------------------
// Factory helpers
// -----------------------------------------------------------------------------

inline std::shared_ptr<const BoardAbstraction> make_exact_board_abstraction() {
    return std::make_shared<ExactBoardAbstraction>();
}

inline std::shared_ptr<const BoardAbstraction>
make_isomorphic_board_abstraction(
    Range p0_range,
    Range p1_range
) {
    return std::make_shared<IsomorphicBoardAbstraction>(
        std::move(p0_range),
        std::move(p1_range)
    );
}
} // namespace poker::holdem