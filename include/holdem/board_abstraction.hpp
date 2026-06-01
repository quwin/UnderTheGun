#pragma once

#include "poker/board.hpp"
#include "../../external/PokerHandEvaluator/cpp/include/phevaluator/card.h"
#include "poker/deck_mask.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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
    virtual std::vector<BoardTransition> next_board_transitions(
        const Board& current_board,
        DeckMask dead_cards
    ) const = 0;

    // Returns a public-board bucket for an already-existing board.
    //
    // Exact mode can return kInvalidBoardBucket.
    // Abstract mode should return a stable bucket id.
    virtual BoardBucketId bucket_for(
        const Board& board
    ) const = 0;

    virtual bool is_exact() const = 0;
};

// -----------------------------------------------------------------------------
// Exact board abstraction
// -----------------------------------------------------------------------------
//
// This is the default implementation. It does not abstract boards.
// It enumerates every legal public card transition exactly.

class ExactBoardAbstraction final : public BoardAbstraction {
public:
    std::vector<BoardTransition> next_board_transitions(
        const Board& current_board,
        DeckMask dead_cards
    ) const override {
        current_board.validate();
        validate_deck_mask(dead_cards);

        if (current_board.is_river()) {
            return {};
        }
        int cards_to_deal = 1;

        // For your postflop subgame builder:
        //
        //   Flop -> Turn: deal 1 card
        //   Turn -> River: deal 1 card
        //
        // Preflop -> Flop needs 3-card combinations. I would not implement
        // preflop here until the rest of the solver is stable.
        if (cards_to_deal != 1) {
            throw std::invalid_argument(
                "ExactBoardAbstraction currently supports only one-card transitions."
            );
        }

        DeckMask unavailable = dead_cards | board_mask(current_board);
        validate_deck_mask(unavailable);

        const std::vector<phevaluator::Card> available_cards = cards_from_mask(remaining_cards(unavailable));

        if (available_cards.empty()) {
            throw std::invalid_argument(
                "No legal public cards remain for board transition."
            );
        }

        const float probability =
            1.0f / static_cast<float>(available_cards.size());

        std::vector<BoardTransition> transitions;
        transitions.reserve(available_cards.size());

        for (phevaluator::Card card : available_cards) {
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
        board.validate();
        // TODO:
        return kInvalidBoardBucket;
    }

    bool is_exact() const override {
        return true;
    }
};

// -----------------------------------------------------------------------------
// Bucketed board abstraction interface
// -----------------------------------------------------------------------------
//
// This is a base class for later abstractions. For example:
//   - texture buckets
//   - equity-distribution buckets
//   - hand-crafted flop classes
//   - learned buckets
//
// You can implement this later without changing HoldemSubgameBuilder.

class BucketedBoardAbstraction : public BoardAbstraction {
public:
    bool is_exact() const override {
        return false;
    }
};

// -----------------------------------------------------------------------------
// Simple representative-bucket abstraction placeholder
// -----------------------------------------------------------------------------
//
// This is intentionally minimal. It is useful for wiring the builder and
// infoset keys before implementing a real board-clustering system.

class NullBucketedBoardAbstraction final : public BucketedBoardAbstraction {
public:
    std::vector<BoardTransition> next_board_transitions(
        const Board& current_board,
        DeckMask dead_cards
    ) const override {
        // Until real board bucketing exists, fall back to exact transitions.
        return exact_.next_board_transitions(
            current_board,
            dead_cards
        );
    }

    BoardBucketId bucket_for(
        const Board& board
    ) const override {
        board.validate();
        // One bucket per street as a placeholder.
        // TODO:
        // This is not strategically meaningful. It only lets the rest of the
        // abstraction plumbing compile.
        if (board.is_flop()) {
            return 0;
        }
        if (board.is_turn()) {
            return 1;
        }
        if (board.is_river()) {
            return 2;
        }
        return kInvalidBoardBucket;
    }

private:
    ExactBoardAbstraction exact_;
};

// -----------------------------------------------------------------------------
// Factory helpers
// -----------------------------------------------------------------------------

inline std::shared_ptr<const BoardAbstraction> make_exact_board_abstraction() {
    return std::make_shared<ExactBoardAbstraction>();
}

inline std::shared_ptr<const BoardAbstraction> make_null_bucketed_board_abstraction() {
    return std::make_shared<NullBucketedBoardAbstraction>();
}

inline std::string to_string(const BoardTransition& transition) {
    return "|board=" + poker::to_string(transition.board) +
           "|bucket=" + std::to_string(transition.bucket_id) +
           "|prob=" + std::to_string(transition.probability);
}

} // namespace poker::holdem