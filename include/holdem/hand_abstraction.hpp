#pragma once

#include "game.hpp"

#include "street.hpp"

#include "poker/board.hpp"
#include "poker/hand.hpp"
#include "../../external/PokerHandEvaluator/cpp/include/phevaluator/phevaluator.h"
#include "../../external/PokerHandEvaluator/cpp/include/phevaluator/rank.h"

#include <memory>
#include <stdexcept>
#include <string>

namespace poker::holdem {

using HandBucketId = int;

constexpr HandBucketId kInvalidHandBucket = -1;

inline HandBucketId bucket_id_from_rank(const phevaluator::Rank rank) {
    return rank.category();
};

// -----------------------------------------------------------------------------
// HandAbstraction
// -----------------------------------------------------------------------------
//
// This interface controls what private-hand identity goes into an infoset key.
//
// Exact mode:
//   P0 AhKh on As7h2cJd4s => bucket = exact HandId
//
// Bucketed mode:
//   P0 AhKh on As7h2cJd4s => bucket = "top pair good kicker" bucket,
//                             equity bucket,
//                             EHS bucket,
//                             learned cluster bucket,
//                             etc.
//
// CFR does not care which one you use. It only needs stable infoset keys.

class HandAbstraction {
public:
    virtual ~HandAbstraction() = default;

    // Returns the bucket used in the acting player's infoset key.
    //
    // The bucket must depend only on:
    //   acting player
    //   acting player's private hand
    //   public board
    //   public street
    //
    // It must not depend on the opponent's exact private hand.
    virtual HandBucketId bucket_for(
        Player player,
        const HoleCards& private_hand,
        const Board& board,
        Street street
    ) const = 0;

    virtual HandBucketId bucket_for(
        Player player,
        HandId private_hand,
        const Board& board,
        Street street
    ) const {
        return bucket_for(
            player,
            hand_from_id(private_hand),
            board,
            street
        );
    }

    virtual bool is_exact() const = 0;

    // Useful for sizing/debugging. Exact abstraction returns kNumHands.
    virtual int num_buckets(Street street) const = 0;
};

// -----------------------------------------------------------------------------
// ExactHandAbstraction
// -----------------------------------------------------------------------------
//
// No private-hand abstraction. Each exact combo gets its own bucket id.
// This is the correct default while validating river/turn/flop tree logic.

class ExactHandAbstraction final : public HandAbstraction {
public:
    HandBucketId bucket_for(
        Player player,
        const HoleCards& private_hand,
        const Board& board,
        Street street
    ) const override {
        if (player != Player::P0 && player != Player::P1) {
            throw std::invalid_argument(
                "ExactHandAbstraction requires P0 or P1."
            );
        }

        validate_street(street);
        board.validate();
        private_hand.validate();

        if (!board_size_matches_street(street, board.size())) {
            throw std::invalid_argument(
                "Board size does not match street."
            );
        }

        if (hand_overlaps_mask(private_hand, board_mask(board))) {
            throw std::invalid_argument(
                "Private hand overlaps public board."
            );
        }

        return static_cast<HandBucketId>(make_hand(private_hand));
    }

    bool is_exact() const override {
        return true;
    }

    int num_buckets(Street /*street*/) const override {
        return kNumHands;
    }
};

// -----------------------------------------------------------------------------
// BucketedHandAbstraction
// -----------------------------------------------------------------------------
//
// Base class for real abstractions. You can later implement:
//   - river made-hand-strength buckets
//   - equity buckets
//   - EHS / HS2 buckets
//   - board-texture-aware buckets
//   - learned/clustering buckets

class BucketedHandAbstraction : public HandAbstraction {
public:
    bool is_exact() const override {
        return false;
    }
};

// -----------------------------------------------------------------------------
// SimpleStrengthHandAbstraction
// -----------------------------------------------------------------------------
//
// A small placeholder abstraction. This is not production-quality poker
// abstraction, but it is useful for testing that infoset bucketing works.
//
// It maps exact hand strength category to bucket:
//
//   0 high card
//   1 one pair
//   2 two pair
//   3 trips
//   4 straight
//   5 flush
//   6 full house
//   7 quads
//   8 straight flush
//
// On flop/turn, this evaluates the current visible board plus private hand only
// if your evaluator supports partial-board evaluation. Since the current
// HandEvaluator interface evaluates 7-card river hands, this implementation
// should only be used on river unless you add partial-board support.

class RiverStrengthHandAbstraction final : public BucketedHandAbstraction {
public:
    [[nodiscard]] HandBucketId bucket_for(
        Player player,
        const HoleCards& private_hand,
        const Board& board,
        Street street
    ) const override {
        if (player != Player::P0 && player != Player::P1) {
            throw std::invalid_argument(
                "RiverStrengthHandAbstraction requires P0 or P1."
            );
        }
        if (street != Street::River) {
            throw std::invalid_argument(
                "RiverStrengthHandAbstraction only supports river boards."
            );
        }
        board.validate();
        private_hand.validate();

        if (!board.is_river()) {
            throw std::invalid_argument(
                "RiverStrengthHandAbstraction requires a five-card board."
            );
        }
        if (hand_overlaps_mask(private_hand, board_mask(board))) {
            throw std::invalid_argument(
                "Private hand overlaps public board."
            );
        }
        const phevaluator::Rank hand_rank = phevaluator::EvaluateCards(
            private_hand.a,
            private_hand.b,
            board.cards[0],
            board.cards[1],
            board.cards[2],
            board.cards[3],
            board.cards[4]
            );

        return bucket_id_from_rank(hand_rank);
    }
    int num_buckets(Street street) const override {
        if (street != Street::River) {
            throw std::invalid_argument(
                "RiverStrengthHandAbstraction only supports river."
            );
        }

        return 9;
    }
};

// -----------------------------------------------------------------------------
// NullBucketedHandAbstraction
// -----------------------------------------------------------------------------
//
// Placeholder for wiring/debugging. It maps all hands to one bucket per street.
// This creates a very lossy abstraction and should not be used for real solving.

class NullBucketedHandAbstraction final : public BucketedHandAbstraction {
public:
    [[nodiscard]] HandBucketId bucket_for(
        Player player,
        const HoleCards& private_hand,
        const Board& board,
        Street street
    ) const override {
        if (player != Player::P0 && player != Player::P1) {
            throw std::invalid_argument(
                "NullBucketedHandAbstraction requires P0 or P1."
            );
        }

        validate_street(street);
        board.validate();
        private_hand.validate();

        if (!board_size_matches_street(street, board.size())) {
            throw std::invalid_argument(
                "Board size does not match street."
            );
        }

        if (hand_overlaps_mask(private_hand, board_mask(board))) {
            throw std::invalid_argument(
                "Private hand overlaps public board."
            );
        }

        switch (street) {
            case Street::Preflop:
                return 0;

            case Street::Flop:
                return 1;

            case Street::Turn:
                return 2;

            case Street::River:
                return 3;
        }

        return kInvalidHandBucket;
    }

    int num_buckets(Street /*street*/) const override {
        return 4;
    }
};

// -----------------------------------------------------------------------------
// Factory helpers
// -----------------------------------------------------------------------------

inline std::shared_ptr<const HandAbstraction> make_exact_hand_abstraction() {
    return std::make_shared<ExactHandAbstraction>();
}

inline std::shared_ptr<const HandAbstraction> make_river_strength_hand_abstraction() {
    return std::make_shared<RiverStrengthHandAbstraction>();
}

inline std::shared_ptr<const HandAbstraction> make_null_bucketed_hand_abstraction() {
    return std::make_shared<NullBucketedHandAbstraction>();
}

// -----------------------------------------------------------------------------
// String helpers
// -----------------------------------------------------------------------------

inline std::string to_string_bucket(
    HandBucketId bucket_id
) {
    if (bucket_id == kInvalidHandBucket) {
        return "invalid";
    }

    return std::to_string(bucket_id);
}

} // namespace poker::holdem