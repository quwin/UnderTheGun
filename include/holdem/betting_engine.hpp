#pragma once

#include "game.hpp"

#include "action.hpp"
#include "betting_abstraction.hpp"
#include "public_state.hpp"
#include "street.hpp"

#include <vector>

namespace poker::holdem {

class BettingEngine {
public:
    BettingEngine() = default;

    // Returns all legal abstract actions from the current public state.
    //
    // Examples:
    //
    // Unopened pot:
    //   check
    //   bet 50% pot
    //   bet 100% pot
    //   all-in
    //
    // Facing bet:
    //   fold
    //   call
    //   raise 2.5x
    //   all-in
    std::vector<Action> legal_actions(
        const PublicState& state,
        const BettingAbstraction& abstraction
    ) const;

    // Applies one legal action and returns the next public state.
    //
    // This updates:
    //   pot
    //   stacks
    //   committed amounts
    //   current_bet_to_call
    //   last_raise_size
    //   num_raises_this_street
    //   player_to_act
    //   last_aggressor
    //   terminal flags
    //   action history
    PublicState apply_action(
        const PublicState& state,
        const Action& action
    ) const;

    // True when the current betting round has closed.
    //
    // Examples:
    //   check-check
    //   bet-call
    //   bet-raise-call
    //   all-in-call
    bool betting_round_closed(
        const PublicState& state
    ) const;

    // True if state is terminal because somebody folded.
    bool is_fold_terminal(
        const PublicState& state
    ) const;

    // True if all remaining betting decisions are over because stacks are all-in.
    bool is_all_in_terminal_for_betting(
        const PublicState& state
    ) const;

private:
    // Legal-action helpers.
    std::vector<Action> legal_unopened_actions(
        const PublicState& state,
        const BettingAbstraction& abstraction
    ) const;

    std::vector<Action> legal_facing_bet_actions(
        const PublicState& state,
        const BettingAbstraction& abstraction
    ) const;

    // Converts abstract bet sizes into concrete chip actions.
    int resolve_first_bet_amount(
        const PublicState& state,
        const BetSize& size,
        const BettingAbstraction& abstraction
    ) const;

    int resolve_raise_to_amount(
        const PublicState& state,
        const BetSize& size,
        const BettingAbstraction& abstraction
    ) const;

    // Stack / pot helpers.
    int actor_stack(
        const PublicState& state
    ) const;

    int opponent_stack(
        const PublicState& state
    ) const;

    int actor_committed(
        const PublicState& state
    ) const;

    int opponent_committed(
        const PublicState& state
    ) const;

    int amount_to_call(
        const PublicState& state
    ) const;

    int actor_total_stack_available_this_street(
        const PublicState& state
    ) const;

    int min_legal_raise_to(
        const PublicState& state
    ) const;

    int max_legal_commitment(
        const PublicState& state
    ) const;

    bool can_raise(
        const PublicState& state,
        const BettingAbstraction& abstraction
    ) const;

    bool is_unopened_pot(
        const PublicState& state
    ) const;

    bool is_facing_bet(
        const PublicState& state
    ) const;

    Player opponent_of(
        Player player
    ) const;

    int round_to_chip_unit(
        int amount,
        int chip_unit
    ) const;

    // Mutation helpers used by apply_action.
    void commit_chips(
        PublicState& state,
        Player player,
        int additional_chips
    ) const;

    void switch_player_to_act(
        PublicState& state
    ) const;

    void append_action_history(
        PublicState& state,
        const Action& action
    ) const;

    void validate_action_basic(
        const PublicState& state,
        const Action& action
    ) const;
};

} // namespace poker::holdem