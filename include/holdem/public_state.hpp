#pragma once

#include "game.hpp"

#include "action.hpp"
#include "betting_state.hpp"
#include "street.hpp"

#include "poker/board.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace poker::holdem {

enum class TerminalReason : int {
    None = 0,

    // A player folded to a bet or raise.
    Fold = 1,

    // Betting is complete on the river.
    Showdown = 2,

    // An all-in was called before the river.
    //
    // The subgame builder can either:
    //   1. expand remaining board run-outs as chance nodes, or
    //   2. collapse the all-in to expected showdown EV.
    AllInCalled = 3
};

struct PublicState {
    // -------------------------------------------------------------------------
    // Public board / street
    // -------------------------------------------------------------------------

    Street street = Street::River;
    Board board;

    // -------------------------------------------------------------------------
    // Public chip state
    // -------------------------------------------------------------------------

    // Total pot in chips, including chips already committed on this street.
    int pot = 0;

    // Remaining stacks.
    int p0_stack = 0;
    int p1_stack = 0;

    // Player whose turn it is.
    //
    // Nonterminal states:
    //   P0 or P1
    //
    // Terminal states:
    //   Terminal
    Player player_to_act = Player::P0;

    // Per-street betting mechanics:
    //
    //   committed amounts
    //   current bet to call
    //   last raise size
    //   raise count
    //   last aggressor
    //   check/check state
    BettingState betting;

    // -------------------------------------------------------------------------
    // Terminal state
    // -------------------------------------------------------------------------

    bool terminal = false;
    TerminalReason terminal_reason = TerminalReason::None;

    // Meaningful only for Fold terminals.
    Player folded_player = Player::Terminal;

    // For Fold terminals, winner is the non-folding player.
    //
    // For Showdown terminals, winner may be:
    //   P0
    //   P1
    //   Terminal for tie / unresolved until evaluator
    Player winner = Player::Terminal;

    // Public action history for infoset keys/debugging.
    std::vector<Action> action_history;

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    bool is_terminal() const {
        return terminal;
    }

    bool is_fold_terminal() const {
        return terminal && terminal_reason == TerminalReason::Fold;
    }

    bool is_showdown_terminal() const {
        return terminal && terminal_reason == TerminalReason::Showdown;
    }

    bool is_all_in_called_terminal() const {
        return terminal && terminal_reason == TerminalReason::AllInCalled;
    }

    int stack(Player player) const {
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
            throw std::invalid_argument("Stack amount must be nonnegative.");
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

    int committed(Player player) const {
        return betting.committed(player);
    }

    int amount_to_call_for(Player player) const {
        return betting.amount_to_call(player);
    }

    int player_total_available_this_street(Player player) const {
        return stack(player) + committed(player);
    }

    bool player_is_all_in(Player player) const {
        return stack(player) == 0;
    }

    bool either_player_all_in() const {
        return p0_stack == 0 || p1_stack == 0;
    }

    bool both_players_have_matched_current_bet() const {
        return betting.p0_committed_this_round ==
               betting.p1_committed_this_round;
    }

    bool betting_locked_by_all_in() const {
        return either_player_all_in();
    }

    Player opponent_of_player_to_act() const {
        return poker::opponent_of(player_to_act);
    }

    // -------------------------------------------------------------------------
    // Validation
    // -------------------------------------------------------------------------

    void validate() const {
        validate_street(street);

        if (!board_size_matches_street(street, board.size())) {
            throw std::invalid_argument(
                "PublicState board size does not match street."
            );
        }

        board.validate();

        if (pot < 0) {
            throw std::invalid_argument("PublicState pot must be nonnegative.");
        }

        if (p0_stack < 0 || p1_stack < 0) {
            throw std::invalid_argument(
                "PublicState stacks must be nonnegative."
            );
        }

        if (player_to_act != Player::P0 &&
            player_to_act != Player::P1 &&
            player_to_act != Player::Terminal) {
            throw std::invalid_argument(
                "PublicState player_to_act must be P0, P1, or Terminal."
            );
        }

        betting.validate();

        if (folded_player != Player::Terminal &&
            folded_player != Player::P0 &&
            folded_player != Player::P1) {
            throw std::invalid_argument(
                "PublicState folded_player must be Terminal, P0, or P1."
            );
        }

        if (winner != Player::Terminal &&
            winner != Player::P0 &&
            winner != Player::P1) {
            throw std::invalid_argument(
                "PublicState winner must be Terminal, P0, or P1."
            );
        }

        if (terminal) {
            if (terminal_reason == TerminalReason::None) {
                throw std::invalid_argument(
                    "Terminal PublicState must have terminal_reason."
                );
            }

            if (player_to_act != Player::Terminal) {
                throw std::invalid_argument(
                    "Terminal PublicState must have player_to_act = Terminal."
                );
            }

            validate_terminal_consistency();
        } else {
            if (terminal_reason != TerminalReason::None) {
                throw std::invalid_argument(
                    "Nonterminal PublicState must have TerminalReason::None."
                );
            }

            if (player_to_act != Player::P0 &&
                player_to_act != Player::P1) {
                throw std::invalid_argument(
                    "Nonterminal PublicState must have real player_to_act."
                );
            }

            if (folded_player != Player::Terminal) {
                throw std::invalid_argument(
                    "Nonterminal PublicState cannot have folded_player set."
                );
            }

            if (winner != Player::Terminal) {
                throw std::invalid_argument(
                    "Nonterminal PublicState cannot have winner set."
                );
            }
        }
    }

private:
    void validate_terminal_consistency() const {
        switch (terminal_reason) {
            case TerminalReason::None:
                throw std::invalid_argument(
                    "Terminal state cannot have TerminalReason::None."
                );

            case TerminalReason::Fold:
                if (folded_player != Player::P0 &&
                    folded_player != Player::P1) {
                    throw std::invalid_argument(
                        "Fold terminal must have folded_player P0 or P1."
                    );
                }

                if (winner != opponent_of(folded_player)) {
                    throw std::invalid_argument(
                        "Fold terminal winner must be opponent of folded player."
                    );
                }
                break;

            case TerminalReason::Showdown:
                if (folded_player != Player::Terminal) {
                    throw std::invalid_argument(
                        "Showdown terminal cannot have folded_player set."
                    );
                }

                // winner may be Terminal here to represent tie or evaluator-resolved
                // showdown utility.
                break;

            case TerminalReason::AllInCalled:
                if (folded_player != Player::Terminal) {
                    throw std::invalid_argument(
                        "AllInCalled terminal cannot have folded_player set."
                    );
                }

                // Usually only used before the river. On the river, prefer Showdown.
                break;
        }
    }
};

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------


inline std::string to_string(TerminalReason reason) {
    switch (reason) {
        case TerminalReason::None:
            return "none";

        case TerminalReason::Fold:
            return "fold";

        case TerminalReason::Showdown:
            return "showdown";

        case TerminalReason::AllInCalled:
            return "all_in_called";
    }

    return "unknown";
}

inline std::string action_history_string(const PublicState& state) {
    std::string result;

    for (std::size_t i = 0; i < state.action_history.size(); ++i) {
        if (i > 0) {
            result += "-";
        }

        result += action_history_token(state.action_history[i]);
    }

    return result;
}

inline std::string to_string(const PublicState& state) {
    return "street=" + to_string(state.street) +
           "|board=" + poker::to_string(state.board) +
           "|pot=" + std::to_string(state.pot) +
           "|p0_stack=" + std::to_string(state.p0_stack) +
           "|p1_stack=" + std::to_string(state.p1_stack) +
           "|" + to_string(state.betting) +
           "|to_act=" + poker::to_string(state.player_to_act) +
           "|history=" + action_history_string(state) +
           "|terminal=" + std::to_string(state.terminal ? 1 : 0) +
           "|reason=" + to_string(state.terminal_reason);
}

inline PublicState make_initial_public_state(
    Street street,
    const Board& board,
    int pot_size,
    int effective_stack,
    Player player_to_act
) {
    if (pot_size < 0) {
        throw std::invalid_argument("pot_size must be nonnegative.");
    }

    if (effective_stack < 0) {
        throw std::invalid_argument("effective_stack must be nonnegative.");
    }

    if (player_to_act != Player::P0 && player_to_act != Player::P1) {
        throw std::invalid_argument("player_to_act must be P0 or P1.");
    }

    PublicState state;

    state.street = street;
    state.board = board;

    state.pot = pot_size;
    state.p0_stack = effective_stack;
    state.p1_stack = effective_stack;

    state.player_to_act = player_to_act;

    state.betting = make_fresh_betting_state();

    state.terminal = false;
    state.terminal_reason = TerminalReason::None;

    state.folded_player = Player::Terminal;
    state.winner = Player::Terminal;

    state.action_history.clear();

    state.validate();

    return state;
}

inline PublicState make_next_street_public_state(
    const PublicState& previous,
    Street next,
    const Board& next_board,
    Player first_to_act
) {
    if (!has_next_street(previous.street)) {
        throw std::invalid_argument(
            "Cannot advance public state beyond river."
        );
    }

    if (next != next_street(previous.street)) {
        throw std::invalid_argument(
            "next street must be exactly one street after previous.street."
        );
    }

    if (first_to_act != Player::P0 && first_to_act != Player::P1) {
        throw std::invalid_argument("first_to_act must be P0 or P1.");
    }

    PublicState state = previous;

    state.street = next;
    state.board = next_board;
    state.player_to_act = first_to_act;

    state.betting.reset_for_new_street();

    state.terminal = false;
    state.terminal_reason = TerminalReason::None;
    state.folded_player = Player::Terminal;
    state.winner = Player::Terminal;

    // Keep cumulative public action history across streets.
    // The new street/betting reset is represented by street + board.
    state.validate();

    return state;
}

inline PublicState make_fold_terminal_state(
    PublicState state,
    Player folded_player
) {
    if (folded_player != Player::P0 && folded_player != Player::P1) {
        throw std::invalid_argument("folded_player must be P0 or P1.");
    }

    state.terminal = true;
    state.terminal_reason = TerminalReason::Fold;
    state.player_to_act = Player::Terminal;
    state.folded_player = folded_player;
    state.winner = opponent_of(folded_player);

    state.validate();

    return state;
}

inline PublicState make_showdown_terminal_state(
    PublicState state
) {
    state.terminal = true;
    state.terminal_reason = TerminalReason::Showdown;
    state.player_to_act = Player::Terminal;
    state.folded_player = Player::Terminal;
    state.winner = Player::Terminal;

    state.validate();

    return state;
}

inline PublicState make_all_in_called_terminal_state(
    PublicState state
) {
    state.terminal = true;
    state.terminal_reason = TerminalReason::AllInCalled;
    state.player_to_act = Player::Terminal;
    state.folded_player = Player::Terminal;
    state.winner = Player::Terminal;

    state.validate();

    return state;
}

} // namespace poker::holdem