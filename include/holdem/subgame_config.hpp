#pragma once

#include "game.hpp"

#include "betting_abstraction.hpp"
#include "board_abstraction.hpp"
#include "hand_abstraction.hpp"
#include "public_state.hpp"
#include "street.hpp"

#include "poker/board.hpp"
#include "poker/range.hpp"

#include <memory>
#include <stdexcept>

namespace poker::holdem {

struct HoldemSubgameConfig {
    // -------------------------------------------------------------------------
    // Starting public state
    // -------------------------------------------------------------------------
    Street start_street = Street::River;
    Board board;
    // Pot at the start of the subgame.
    //
    // Convention:
    //   pot_size already includes all chips committed before this subgame starts.
    int pot_size = 0;
    // Remaining effective stack for each player at the start of the subgame.
    //
    // For now this assumes symmetric effective stacks:
    //   p0_stack = effective_stack
    //   p1_stack = effective_stack
    int effective_stack = 0;
    Player player_to_act = Player::P0;
    // -------------------------------------------------------------------------
    // Private hand ranges
    // -------------------------------------------------------------------------
    Range p0_range;
    Range p1_range;
    // -------------------------------------------------------------------------
    // Betting abstraction
    // -------------------------------------------------------------------------
    BettingAbstraction betting_abstraction;
    // -------------------------------------------------------------------------
    // Optional card abstractions
    // -------------------------------------------------------------------------
    // Exact by default.
    std::shared_ptr<const HandAbstraction> hand_abstraction = make_exact_hand_abstraction();
    // Exact by default.
    std::shared_ptr<const BoardAbstraction> board_abstraction = make_exact_board_abstraction();
    // -------------------------------------------------------------------------
    // All-in handling
    // -------------------------------------------------------------------------
    // If true, all-in calls before the river are collapsed into one expected-EV
    // terminal node.
    bool collapse_all_in_runouts_to_ev = true;
    // -------------------------------------------------------------------------
    // Builder/debug options
    // -------------------------------------------------------------------------
    // Useful for early testing. If true, builder may do extra expensive
    // consistency checks.
    bool validate_tree_during_build = false;
    // If true, reject preflop configs until preflop logic is implemented.
    bool reject_preflop = true;
    // -------------------------------------------------------------------------
    // Validation
    // -------------------------------------------------------------------------
    void validate() const {
        validate_street(start_street);

        if (reject_preflop && start_street == Street::Preflop) {
            throw std::invalid_argument(
                "Preflop subgames are not implemented yet."
            );
        }

        board.validate();

        if (!board_size_matches_street(start_street, board.size())) {
            throw std::invalid_argument(
                "HoldemSubgameConfig board size does not match start_street."
            );
        }

        if (pot_size < 0) {
            throw std::invalid_argument(
                "HoldemSubgameConfig pot_size must be nonnegative."
            );
        }

        if (effective_stack < 0) {
            throw std::invalid_argument(
                "HoldemSubgameConfig effective_stack must be nonnegative."
            );
        }

        if (player_to_act != Player::P0 && player_to_act != Player::P1) {
            throw std::invalid_argument(
                "HoldemSubgameConfig player_to_act must be P0 or P1."
            );
        }

        if (p0_range.empty()) {
            throw std::invalid_argument(
                "HoldemSubgameConfig p0_range cannot be empty."
            );
        }

        if (p1_range.empty()) {
            throw std::invalid_argument(
                "HoldemSubgameConfig p1_range cannot be empty."
            );
        }

        betting_abstraction.validate();

        if (!hand_abstraction) {
            throw std::invalid_argument(
                "HoldemSubgameConfig hand_abstraction cannot be null."
            );
        }

        if (!board_abstraction) {
            throw std::invalid_argument(
                "HoldemSubgameConfig board_abstraction cannot be null."
            );
        }

        // Verify at least one legal private hand pair remains after board
        // blockers. This is potentially O(range^2), but config validation is
        // not in the CFR loop.
        const std::vector<LegalHandPair> pairs =
            legal_hand_pairs(
                p0_range,
                p1_range,
                board
            );

        if (pairs.empty()) {
            throw std::invalid_argument(
                "No legal private hand pairs remain after board card removal."
            );
        }
    }

    PublicState initial_public_state() const {
        validate();

        return make_initial_public_state(
            start_street,
            board,
            pot_size,
            effective_stack,
            player_to_act
        );
    }
};

inline HoldemSubgameConfig make_default_river_subgame_config(
    const Board& board,
    const Range& p0_range,
    const Range& p1_range,
    int pot_size,
    int effective_stack,
    Player player_to_act
) {
    HoldemSubgameConfig config;

    config.start_street = Street::River;
    config.board = board;

    config.pot_size = pot_size;
    config.effective_stack = effective_stack;
    config.player_to_act = player_to_act;

    config.p0_range = p0_range;
    config.p1_range = p1_range;

    config.betting_abstraction = make_standard_abstraction();

    config.hand_abstraction = make_exact_hand_abstraction();
    config.board_abstraction = make_exact_board_abstraction();
    config.collapse_all_in_runouts_to_ev = true;

    config.validate();

    return config;
}

inline HoldemSubgameConfig make_tiny_river_subgame_config(
    const Board& board,
    const Range& p0_range,
    const Range& p1_range
) {
    HoldemSubgameConfig config;

    config.start_street = Street::River;
    config.board = board;

    config.pot_size = 1000;
    config.effective_stack = 2000;
    config.player_to_act = Player::P0;

    config.p0_range = p0_range;
    config.p1_range = p1_range;

    config.betting_abstraction = make_tiny_betting_abstraction();

    config.hand_abstraction = make_exact_hand_abstraction();
    config.board_abstraction = make_exact_board_abstraction();
    config.collapse_all_in_runouts_to_ev = false;

    config.validate();

    return config;
}

} // namespace poker::holdem