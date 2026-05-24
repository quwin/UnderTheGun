#pragma once

#include "board_abstraction.hpp"
#include "private_state.hpp"
#include "public_state.hpp"
#include "subgame_config.hpp"

#include "poker/board.hpp"
#include "poker/deck_mask.hpp"
#include "poker/hand.hpp"
#include "poker/range.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace poker::holdem {

// -----------------------------------------------------------------------------
// Private hand-pair chance
// -----------------------------------------------------------------------------

struct PrivateDealOutcome {
    PrivateState private_state;

    // Normalized probability of this P0/P1 hand pair.
    float probability = 0.0f;
};

// -----------------------------------------------------------------------------
// Public-card chance
// -----------------------------------------------------------------------------

struct PublicBoardOutcome {
    PublicState public_state;

    // Exact mode:
    //   bucket_id = kInvalidBoardBucket
    //
    // Abstract mode:
    //   bucket_id = public board bucket.
    BoardBucketId bucket_id = kInvalidBoardBucket;

    // Normalized chance probability.
    float probability = 0.0f;
};

// -----------------------------------------------------------------------------
// Chance model
// -----------------------------------------------------------------------------

class ChanceModel {
public:
    ChanceModel()
        : board_abstraction_(make_exact_board_abstraction()) {}

    explicit ChanceModel(
        std::shared_ptr<const BoardAbstraction> board_abstraction
    )
        : board_abstraction_(std::move(board_abstraction)) {
        if (!board_abstraction_) {
            throw std::invalid_argument(
                "ChanceModel requires a non-null BoardAbstraction."
            );
        }
    }

    // Root chance:
    //
    //   choose P0 private hand from P0 range
    //   choose P1 private hand from P1 range
    //
    // subject to:
    //
    //   no hand overlaps board
    //   no hand overlaps opponent hand
    //   range weights > 0
    //
    // Returned probabilities are normalized.
    std::vector<PrivateDealOutcome> private_deal_outcomes(
        const Range& p0_range,
        const Range& p1_range,
        const Board& board
    ) const {
        board.validate();

        const std::vector<LegalHandPair> legal_pairs =
            legal_hand_pairs(
                p0_range,
                p1_range,
                board
            );

        std::vector<PrivateDealOutcome> outcomes;
        outcomes.reserve(legal_pairs.size());

        for (const LegalHandPair& pair : legal_pairs) {
            outcomes.push_back(
                PrivateDealOutcome{
                    PrivateState{
                        hand_from_id(pair.p0_hand),
                        hand_from_id(pair.p1_hand)
                    },
                    pair.probability
                }
            );
        }

        validate_probability_sum(outcomes);

        return outcomes;
    }

    // Root chance overload for config.
    std::vector<PrivateDealOutcome> private_deal_outcomes(
        const HoldemSubgameConfig& config
    ) const {
        return private_deal_outcomes(
            config.p0_range,
            config.p1_range,
            config.board
        );
    }

    // Public-card chance after a betting round closes.
    //
    // Examples:
    //
    //   Flop -> Turn:
    //     enumerate legal turn cards
    //
    //   Turn -> River:
    //     enumerate legal river cards
    //
    //   River:
    //     returns empty
    //
    // This does not decide whether a betting round is closed. That is the
    // BettingEngine's job. The subgame builder should call this only when
    // public chance is actually needed.
    std::vector<PublicBoardOutcome> public_board_outcomes(
        const PublicState& state,
        const PrivateState& private_state,
        Player next_player_to_act
    ) const {
        state.validate();
        private_state.validate();

        if (state.terminal) {
            return {};
        }

        if (state.street == Street::River) {
            return {};
        }

        if (next_player_to_act != Player::P0 &&
            next_player_to_act != Player::P1) {
            throw std::invalid_argument(
                "next_player_to_act must be P0 or P1."
            );
        }

        const DeckMask dead_cards =
            board_mask(state.board) |
            private_state.mask();

        const std::vector<BoardTransition> transitions =
            board_abstraction_->next_board_transitions(
                state.street,
                state.board,
                dead_cards
            );

        std::vector<PublicBoardOutcome> outcomes;
        outcomes.reserve(transitions.size());

        for (const BoardTransition& transition : transitions) {
            if (transition.probability <= 0.0f) {
                throw std::invalid_argument(
                    "Board transition probability must be positive."
                );
            }

            if (!std::isfinite(transition.probability)) {
                throw std::invalid_argument(
                    "Board transition probability must be finite."
                );
            }

            PublicState next_state = make_next_street_public_state(
                state,
                transition.street,
                transition.board,
                next_player_to_act
            );

            outcomes.push_back(
                PublicBoardOutcome{
                    next_state,
                    transition.bucket_id,
                    transition.probability
                }
            );
        }

        validate_probability_sum(outcomes);

        return outcomes;
    }

    const BoardAbstraction& board_abstraction() const {
        return *board_abstraction_;
    }

private:
    std::shared_ptr<const BoardAbstraction> board_abstraction_;

    static void validate_probability_value(float probability) {
        if (!std::isfinite(probability)) {
            throw std::invalid_argument(
                "Chance probability must be finite."
            );
        }

        if (probability <= 0.0f) {
            throw std::invalid_argument(
                "Chance probability must be positive."
            );
        }

        if (probability > 1.0f) {
            throw std::invalid_argument(
                "Chance probability cannot exceed 1."
            );
        }
    }

    static void validate_probability_sum(
        const std::vector<PrivateDealOutcome>& outcomes
    ) {
        if (outcomes.empty()) {
            throw std::invalid_argument(
                "Chance outcome list cannot be empty."
            );
        }

        double sum = 0.0;

        for (const PrivateDealOutcome& outcome : outcomes) {
            validate_probability_value(outcome.probability);
            sum += static_cast<double>(outcome.probability);
        }

        if (std::abs(sum - 1.0) > 1e-5) {
            throw std::invalid_argument(
                "Private deal probabilities must sum to 1."
            );
        }
    }

    static void validate_probability_sum(
        const std::vector<PublicBoardOutcome>& outcomes
    ) {
        if (outcomes.empty()) {
            throw std::invalid_argument(
                "Chance outcome list cannot be empty."
            );
        }

        double sum = 0.0;

        for (const PublicBoardOutcome& outcome : outcomes) {
            validate_probability_value(outcome.probability);
            sum += static_cast<double>(outcome.probability);
        }

        if (std::abs(sum - 1.0) > 1e-5) {
            throw std::invalid_argument(
                "Public board probabilities must sum to 1."
            );
        }
    }
};

} // namespace poker::holdem