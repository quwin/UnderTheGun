#pragma once

#include "game.hpp"

#include "holdem/private_state.hpp"
#include "holdem/public_state.hpp"
#include "holdem/street.hpp"

#include "poker/board.hpp"

#include <stdexcept>
#include <string>

#include "../../external/PokerHandEvaluator/cpp/include/phevaluator/phevaluator.h"
#include "../../external/PokerHandEvaluator/cpp/include/phevaluator/rank.h"

namespace poker::holdem {

// -----------------------------------------------------------------------------
// Terminal utility convention
// -----------------------------------------------------------------------------
//
// Utility is expressed from P0's perspective.
//
// Subgame-net convention:
//
//   P0 wins pot:
//     +pot - p0_committed_this_round
//
//   P0 loses at showdown:
//     -p0_committed_this_round
//
//   P0 folds:
//     -p0_committed_this_round
//
//   P1 folds:
//     +pot - p0_committed_this_round
//
//   Tie:
//     p0_share_of_pot - p0_committed_this_round
//
// This treats the starting pot as contestable value rather than as already
// owned by either player.

inline float utility_p0_when_p0_wins(
    const PublicState& state
) {
    return static_cast<float>(
        state.pot - state.betting.p0_committed_this_round
    );
}

inline float utility_p0_when_p0_loses(
    const PublicState& state
) {
    return static_cast<float>(
        -state.betting.p0_committed_this_round
    );
}

inline float utility_p0_when_tie(
    const PublicState& state
) {
    const double half_pot =
        static_cast<double>(state.pot) * 0.5;

    return static_cast<float>(
        half_pot -
        static_cast<double>(state.betting.p0_committed_this_round)
    );
}

// -----------------------------------------------------------------------------
// Fold utility
// -----------------------------------------------------------------------------

inline float fold_terminal_utility_p0(
    const PublicState& state
) {
    if (!state.terminal ||
        state.terminal_reason != TerminalReason::Fold) {
        throw std::invalid_argument(
            "fold_terminal_utility_p0 requires a fold terminal state."
        );
    }

    if (state.folded_player == Player::P0) {
        return utility_p0_when_p0_loses(state);
    }

    if (state.folded_player == Player::P1) {
        return utility_p0_when_p0_wins(state);
    }

    throw std::invalid_argument(
        "Fold terminal must identify folded player."
    );
}

// -----------------------------------------------------------------------------
// Showdown utility
// -----------------------------------------------------------------------------

inline float showdown_terminal_utility_p0(
    const PublicState& state,
    const PrivateState& private_state
) {
    if (!state.terminal ||
        state.terminal_reason != TerminalReason::Showdown) {
        throw std::invalid_argument(
            "showdown_terminal_utility_p0 requires a showdown terminal state."
        );
    }

    if (state.street != Street::River) {
        throw std::invalid_argument(
            "Showdown terminal requires river street."
        );
    }

    if (!state.board.is_river()) {
        throw std::invalid_argument(
            "Showdown terminal requires five-card board."
        );
    }

    private_state.validate();

    if (private_state.overlaps_mask(board_mask(state.board))) {
        throw std::invalid_argument(
            "Private hands overlap public board."
        );
    }

    const phevaluator::Rank p0_rank = phevaluator::EvaluateCards(
        private_state.p0_hand.a,
        private_state.p0_hand.b,
        state.board.cards[0],
        state.board.cards[1],
        state.board.cards[2],
        state.board.cards[3],
        state.board.cards[4]
        );

    const phevaluator::Rank p1_rank = phevaluator::EvaluateCards(
        private_state.p1_hand.a,
        private_state.p1_hand.b,
        state.board.cards[0],
        state.board.cards[1],
        state.board.cards[2],
        state.board.cards[3],
        state.board.cards[4]
        );
    // Smaller value is stronger, from 1 to 7462
    if (p0_rank < p1_rank) {
        return utility_p0_when_p0_wins(state);
    }
    else if (p1_rank < p0_rank) {
        return utility_p0_when_p0_loses(state);
    }
    return utility_p0_when_tie(state);
}

// -----------------------------------------------------------------------------
// All-in called utility
// -----------------------------------------------------------------------------
//
// This function is intentionally strict: all-in called before river should
// usually be handled by either:
//   1. expanding future runouts as chance nodes, or
//   2. using an AllInEquityResolver that averages all legal runouts.
//
// If the all-in call happens on the river, it is just showdown.

inline float all_in_called_terminal_utility_p0(
    const PublicState& state,
    const PrivateState& private_state
) {
    if (!state.terminal ||
        state.terminal_reason != TerminalReason::AllInCalled) {
        throw std::invalid_argument(
            "all_in_called_terminal_utility_p0 requires an all-in-called terminal state."
        );
    }

    if (state.street != Street::River) {
        throw std::invalid_argument(
            "Pre-river all-in EV should be resolved by runout expansion or AllInEquityResolver."
        );
    }

    PublicState showdown_state = state;
    showdown_state.terminal_reason = TerminalReason::Showdown;

    return showdown_terminal_utility_p0(
        showdown_state,
        private_state
    );
}

// -----------------------------------------------------------------------------
// Main terminal utility entry point
// -----------------------------------------------------------------------------

inline float terminal_utility_p0(
    const PublicState& state,
    const PrivateState& private_state
) {
    state.validate();
    private_state.validate();

    if (!state.terminal) {
        throw std::invalid_argument(
            "terminal_utility_p0 requires a terminal PublicState."
        );
    }

    switch (state.terminal_reason) {
        case TerminalReason::Fold:
            return fold_terminal_utility_p0(state);

        case TerminalReason::Showdown:
            return showdown_terminal_utility_p0(
                state,
                private_state
            );

        case TerminalReason::AllInCalled:
            return all_in_called_terminal_utility_p0(
                state,
                private_state
            );

        case TerminalReason::None:
            break;
    }

    throw std::invalid_argument(
        "Invalid terminal reason for terminal utility."
    );
}

inline float terminal_utility_for_player(
    const PublicState& state,
    const PrivateState& private_state,
    Player player
) {
    const float utility_p0 =
        terminal_utility_p0(state, private_state
    );
    if (player == Player::P0) {
        return utility_p0;
    }
    if (player == Player::P1) {
        return -utility_p0;
    }
    throw std::invalid_argument(
        "terminal_utility_for_player requires P0 or P1."
    );
}

} // namespace poker::holdem