#pragma once

#include "game.hpp"

#include "holdem/public_state.hpp"
#include "poker/board.hpp"

#include <stdexcept>

#include "all_in_equity_cache.hpp"
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
    if (state.terminal_type == TerminalType::P0_Fold) {
        return utility_p0_when_p0_loses(state);
    }
    if (state.terminal_type == TerminalType::P1_Fold) {
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
    // Smaller value is weaker, from 1 to 7462
    if (p0_rank > p1_rank) {
        return utility_p0_when_p0_wins(state);
    }
    if (p1_rank > p0_rank) {
        return utility_p0_when_p0_loses(state);
    }
    return utility_p0_when_tie(state);
}

inline float all_in_called_terminal_utility_p0(
    const PublicState& state,
    const PrivateState& private_state,
    AllInEquityCache* all_in_equity_cache
) {
    if (state.board.is_river()) {
        return showdown_terminal_utility_p0(
            state,
            private_state
        );
    }
    if (all_in_equity_cache == nullptr) {
        throw std::invalid_argument(
            "Pre-river collapsed all-in EV requires AllInEquityCache."
        );
    }
    const float p0_equity = all_in_equity_cache->p0_equity(state.board,private_state);
    if (!std::isfinite(p0_equity) || p0_equity < 0.0f || p0_equity > 1.0f) {
        throw std::runtime_error(
            "AllInEquityCache returned invalid equity."
        );
    }
    const float win_utility = utility_p0_when_p0_wins(state);
    const float lose_utility = utility_p0_when_p0_loses(state);

    return p0_equity * win_utility + (1.0f - p0_equity) * lose_utility;
}

// -----------------------------------------------------------------------------
// Main terminal utility entry point
// -----------------------------------------------------------------------------

inline float terminal_utility_p0(
    const PublicState& state,
    const PrivateState& private_state,
    AllInEquityCache* all_in_equity_cache
) {
    if (!state.is_terminal()) {
        throw std::invalid_argument(
            "terminal_utility_p0 requires a terminal PublicState."
        );
    }

    switch (state.terminal_type) {
        case TerminalType::P0_Fold:
        case TerminalType::P1_Fold:
            return fold_terminal_utility_p0(state);
        case TerminalType::Showdown:
            return showdown_terminal_utility_p0(state,private_state);
        case TerminalType::AllIn:
            return all_in_called_terminal_utility_p0(state, private_state, all_in_equity_cache);
        case TerminalType::None:
            break;
    }
    throw std::invalid_argument(
        "Invalid terminal reason for terminal utility."
    );
}

inline float terminal_utility_for_player(
    const PublicState& state,
    const PrivateState& private_state,
    const Player player,
    AllInEquityCache* all_in_equity_cache = nullptr
) {
    const float utility_p0 = terminal_utility_p0(state, private_state, all_in_equity_cache);
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