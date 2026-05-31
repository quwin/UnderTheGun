#pragma once

#include "game.hpp"
#include "action.hpp"
#include "betting_state.hpp"
#include "poker/board.hpp"
#include <stdexcept>

namespace poker::holdem {

struct PublicState {
    Board board;
    // ---------------------------------------------------------------------
    // Public chip state
    // ---------------------------------------------------------------------
    // Total pot in chips, including chips already committed on this street.
    int pot = 0;
    // Remaining stacks.
    int p0_stack = 0;
    int p1_stack = 0;
    // Nonterminal states:
    //   P0 or P1
    //
    // Terminal states:
    //   Terminal
    Player player_to_act = Player::P0;
    // Per-street betting state.
    BettingState betting;
    // ---------------------------------------------------------------------
    // Terminal state
    // ---------------------------------------------------------------------
    TerminalType terminal_type = TerminalType::None;
    [[nodiscard]] bool is_terminal() const {
        return terminal_type != TerminalType::None;
    }
    [[nodiscard]] int stack(Player player) const {
        switch (player) {
            case Player::P0:
                return p0_stack;

            case Player::P1:
                return p1_stack;

            default:
                throw std::invalid_argument(
                    "PublicState::stack requires P0 or P1."
                );
        }
    }

    void set_stack(Player player, int amount) {
        if (amount < 0) {
            throw std::invalid_argument(
                "PublicState stack amount must be nonnegative."
            );
        }

        switch (player) {
            case Player::P0:
                p0_stack = amount;
                return;

            case Player::P1:
                p1_stack = amount;
                return;

            default:
                throw std::invalid_argument(
                    "PublicState::set_stack requires P0 or P1."
                );
        }
    }

    [[nodiscard]] int committed(Player player) const {
        return betting.committed(player);
    }

    [[nodiscard]] int amount_to_call_for(Player player) const {
        return betting.amount_to_call(player);
    }

    [[nodiscard]] int player_total_available_this_street(Player player) const {
        return stack(player) + committed(player);
    }

    [[nodiscard]] bool player_is_all_in(Player player) const {
        return stack(player) == 0;
    }

    [[nodiscard]] bool either_player_all_in() const {
        return p0_stack == 0 || p1_stack == 0;
    }

    [[nodiscard]] bool both_players_have_matched_current_bet() const {
        return betting.p0_committed_this_round == betting.p1_committed_this_round;
    }

    [[nodiscard]] bool betting_locked_by_all_in() const {
        return either_player_all_in();
    }
    // ---------------------------------------------------------------------
    // Mutation helpers used by BettingEngine / builder
    // ---------------------------------------------------------------------

    void switch_player_to_act() {
        if (player_to_act == Player::P0) {
            player_to_act = Player::P1;
            return;
        }

        if (player_to_act == Player::P1) {
            player_to_act = Player::P0;
            return;
        }

        throw std::logic_error(
            "Cannot switch player_to_act from non-real player."
        );
    }

    void reset_betting_for_new_street(
        Board new_board,
        Player first_to_act
    ) {
        new_board.validate();

        if (first_to_act != Player::P0 && first_to_act != Player::P1) {
            throw std::invalid_argument(
                "first_to_act must be P0 or P1."
            );
        }

        board = std::move(new_board);
        player_to_act = first_to_act;

        betting = BettingState{};

        terminal_type = TerminalType::None;
    }

    void make_fold_terminal(Player folder) {
        if (folder != Player::P0 && folder != Player::P1) {
            throw std::invalid_argument(
                "make_fold_terminal requires P0 or P1 folder."
            );
        }
        if (folder == Player::P0) {
            terminal_type = TerminalType::P0_Fold;
        }
        if (folder == Player::P1) {
            terminal_type = TerminalType::P1_Fold;
        }
        player_to_act = Player::Terminal;
    }

    void make_showdown_terminal() {
        if (!board.is_river()) {
            throw std::invalid_argument(
                "make_showdown_terminal requires river street."
            );
        }
        terminal_type = TerminalType::Showdown;
        player_to_act = Player::Terminal;
    }

    void make_all_in_called_terminal() {
        terminal_type = TerminalType::AllIn;
        player_to_act = Player::Terminal;
    }

    // Clears AllInCalled terminal marker so the public-tree builder can expand
    // remaining board runouts as chance nodes.
    void clear_all_in_terminal_for_runout() {
        if (terminal_type != TerminalType::AllIn) {
            throw std::invalid_argument(
                "clear_all_in_terminal_for_runout requires AllInCalled terminal."
            );
        }
        terminal_type = TerminalType::None;
        // This value is not used for betting while all-in. It just keeps
        // PublicState valid until the builder creates a chance node.
        player_to_act = Player::P0;
    }

    // ---------------------------------------------------------------------
    // Validation
    // ---------------------------------------------------------------------

    void validate() const {
        board.validate();
        if (pot < 0) {
            throw std::invalid_argument(
                "PublicState pot must be nonnegative."
            );
        }
        if (p0_stack < 0 || p1_stack < 0) {
            throw std::invalid_argument(
                "PublicState stacks must be nonnegative."
            );
        }
        betting.validate();
    }
};

// -----------------------------------------------------------------------------
// Free helpers
// -----------------------------------------------------------------------------

inline void switch_player_to_act(PublicState& state) {
    state.switch_player_to_act();
}

inline PublicState make_next_street_public_state(
    PublicState state,
    Board next_board,
    Player first_to_act
) {
    state.reset_betting_for_new_street(
        std::move(next_board),
        first_to_act
    );

    return state;
}
} // namespace poker::holdem