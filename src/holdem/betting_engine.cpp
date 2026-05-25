#include "holdem/betting_engine.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace poker::holdem {

namespace {

bool same_action(const Action& a, const Action& b) {
    return a.type == b.type && a.amount == b.amount;
}

void push_unique_action(
    std::vector<Action>& actions,
    const Action& action
) {
    const auto found = std::find_if(
        actions.begin(),
        actions.end(),
        [&](const Action& existing) {
            return same_action(existing, action);
        }
    );

    if (found == actions.end()) {
        actions.push_back(action);
    }
}

bool is_call_like_all_in(
    const PublicState& state,
    Player player,
    const Action& action
) {
    if (action.type != ActionType::AllIn) {
        return false;
    }

    const int committed = state.betting.committed(player);
    const int total_after_all_in = action.amount;

    return total_after_all_in <= state.betting.current_bet_to_call &&
           total_after_all_in > committed;
}

} // namespace

std::vector<Action> BettingEngine::legal_actions(
    const PublicState& state,
    const BettingAbstraction& abstraction
) const {
    state.validate();
    abstraction.validate();

    if (state.terminal) {
        return {};
    }

    if (state.player_to_act != Player::P0 &&
        state.player_to_act != Player::P1) {
        throw std::invalid_argument(
            "legal_actions requires P0 or P1 to act."
        );
    }

    if (is_unopened_pot(state)) {
        return legal_unopened_actions(state, abstraction);
    }

    if (is_facing_bet(state)) {
        return legal_facing_bet_actions(state, abstraction);
    }

    // This can happen after one player called and the round is already closed.
    return {};
}

PublicState BettingEngine::apply_action(
    const PublicState& state,
    const Action& action
) const {
    state.validate();
    validate_action_basic(state, action);

    PublicState next = state;

    const Player actor = state.player_to_act;
    const Player opponent = opponent_of(actor);

    switch (action.type) {
        case ActionType::Fold: {
            if (!is_facing_bet(state)) {
                throw std::invalid_argument(
                    "Cannot fold when not facing a bet."
                );
            }

            next.terminal = true;
            next.terminal_reason = TerminalReason::Fold;
            next.player_to_act = Player::Terminal;
            next.folded_player = actor;
            next.winner = opponent;
            append_action_history(next, action);
            next.validate();
            return next;
        }

        case ActionType::Check: {
            if (!is_unopened_pot(state)) {
                throw std::invalid_argument(
                    "Cannot check while facing a bet."
                );
            }

            next.betting.last_action_was_check = true;
            append_action_history(next, action);
            switch_player_to_act(next);
            next.validate();
            return next;
        }

        case ActionType::Bet: {
            if (!is_unopened_pot(state)) {
                throw std::invalid_argument(
                    "Cannot bet after a bet already exists; use raise."
                );
            }

            const int target_commitment = action.amount;
            const int max_commitment = max_legal_commitment(state);

            if (target_commitment <= actor_committed(state) ||
                target_commitment > max_commitment) {
                throw std::invalid_argument(
                    "Illegal bet commitment amount."
                );
            }

            const int additional =
                target_commitment - actor_committed(state);

            commit_chips(next, actor, additional);

            next.betting.round_has_bet = true;
            next.betting.current_bet_to_call = target_commitment;
            next.betting.last_raise_size = target_commitment;
            next.betting.last_aggressor = actor;
            next.betting.last_action_was_check = false;

            append_action_history(next, action);
            switch_player_to_act(next);
            next.validate();
            return next;
        }

        case ActionType::Raise: {
            if (!is_facing_bet(state)) {
                throw std::invalid_argument(
                    "Cannot raise when not facing a bet."
                );
            }

            const int target_commitment = action.amount;
            const int min_raise_to = min_legal_raise_to(state);
            const int max_commitment = max_legal_commitment(state);

            if (target_commitment < min_raise_to ||
                target_commitment > max_commitment) {
                throw std::invalid_argument(
                    "Illegal raise-to commitment amount."
                );
            }

            const int previous_bet_to_call =
                state.betting.current_bet_to_call;

            const int additional =
                target_commitment - actor_committed(state);

            commit_chips(next, actor, additional);

            next.betting.round_has_bet = true;
            next.betting.current_bet_to_call = target_commitment;
            next.betting.last_raise_size =
                target_commitment - previous_bet_to_call;
            next.betting.num_raises_this_street += 1;
            next.betting.last_aggressor = actor;
            next.betting.last_action_was_check = false;

            append_action_history(next, action);
            switch_player_to_act(next);
            next.validate();
            return next;
        }

        case ActionType::Call: {
            if (!is_facing_bet(state)) {
                throw std::invalid_argument(
                    "Cannot call when not facing a bet."
                );
            }

            const int required_call = amount_to_call(state);

            if (action.amount != std::min(required_call, actor_stack(state))) {
                throw std::invalid_argument(
                    "Call amount must equal the required call or remaining stack."
                );
            }

            commit_chips(next, actor, action.amount);

            next.betting.last_action_was_check = false;
            append_action_history(next, action);

            if (next.street != Street::River &&
                is_all_in_terminal_for_betting(next)) {
                next.terminal = true;
                next.terminal_reason = TerminalReason::AllInCalled;
                next.player_to_act = Player::Terminal;
                next.folded_player = Player::Terminal;
                next.winner = Player::Terminal;
            } else {
                switch_player_to_act(next);
            }

            next.validate();
            return next;
        }

        case ActionType::AllIn: {
            const int target_commitment = action.amount;
            const int max_commitment = max_legal_commitment(state);

            if (target_commitment != max_commitment) {
                throw std::invalid_argument(
                    "AllIn amount must equal actor's maximum commitment."
                );
            }

            if (target_commitment <= actor_committed(state)) {
                throw std::invalid_argument(
                    "AllIn must commit additional chips."
                );
            }

            const int additional =
                target_commitment - actor_committed(state);

            if (is_unopened_pot(state)) {
                commit_chips(next, actor, additional);

                next.betting.round_has_bet = true;
                next.betting.current_bet_to_call = target_commitment;
                next.betting.last_raise_size = target_commitment;
                next.betting.last_aggressor = actor;
                next.betting.last_action_was_check = false;

                append_action_history(next, action);
                switch_player_to_act(next);
                next.validate();
                return next;
            }

            if (!is_facing_bet(state)) {
                throw std::invalid_argument(
                    "AllIn is only valid unopened or facing a bet."
                );
            }

            const int previous_bet_to_call =
                state.betting.current_bet_to_call;

            commit_chips(next, actor, additional);
            next.betting.last_action_was_check = false;

            if (target_commitment <= previous_bet_to_call) {
                // All-in call. This may be a full call or short all-in call.
                append_action_history(next, action);

                if (next.street != Street::River &&
                    is_all_in_terminal_for_betting(next)) {
                    next.terminal = true;
                    next.terminal_reason = TerminalReason::AllInCalled;
                    next.player_to_act = Player::Terminal;
                    next.folded_player = Player::Terminal;
                    next.winner = Player::Terminal;
                } else {
                    switch_player_to_act(next);
                }

                next.validate();
                return next;
            }

            // All-in raise.
            next.betting.round_has_bet = true;
            next.betting.current_bet_to_call = target_commitment;
            next.betting.last_raise_size =
                target_commitment - previous_bet_to_call;
            next.betting.num_raises_this_street += 1;
            next.betting.last_aggressor = actor;

            append_action_history(next, action);
            switch_player_to_act(next);
            next.validate();
            return next;
        }
    }

    throw std::invalid_argument("Unknown action type.");
}

bool BettingEngine::betting_round_closed(
    const PublicState& state
) const {
    state.validate();

    if (state.terminal) {
        return true;
    }

    if (state.action_history.empty()) {
        return false;
    }

    if (!state.betting.round_has_bet) {
        if (state.action_history.size() < 2) {
            return false;
        }

        const Action& last =
            state.action_history[state.action_history.size() - 1];

        const Action& previous =
            state.action_history[state.action_history.size() - 2];

        return previous.type == ActionType::Check &&
               last.type == ActionType::Check;
    }

    const Action& last = state.action_history.back();

    if (last.type == ActionType::Call) {
        return state.betting.commitments_matched();
    }

    if (last.type == ActionType::AllIn &&
        state.betting.commitments_matched()) {
        return true;
    }

    return false;
}

bool BettingEngine::is_fold_terminal(
    const PublicState& state
) const {
    return state.terminal &&
           state.terminal_reason == TerminalReason::Fold;
}

bool BettingEngine::is_all_in_terminal_for_betting(
    const PublicState& state
) const {
    if (!state.betting.round_has_bet) {
        return false;
    }

    if (!state.either_player_all_in()) {
        return false;
    }

    return state.betting.commitments_matched();
}

std::vector<Action> BettingEngine::legal_unopened_actions(
    const PublicState& state,
    const BettingAbstraction& abstraction
) const {
    std::vector<Action> actions;

    push_unique_action(actions, check_action());

    for (const BetSize& size : abstraction.first_bet_sizes) {
        const int amount =
            resolve_first_bet_amount(state, size, abstraction);

        if (amount <= 0) {
            continue;
        }

        const int max_commitment = max_legal_commitment(state);

        if (amount >= max_commitment) {
            push_unique_action(
                actions,
                all_in_action(max_commitment)
            );
        } else {
            push_unique_action(
                actions,
                bet_action(amount)
            );
        }
    }

    if (abstraction.always_allow_all_in) {
        push_unique_action(
            actions,
            all_in_action(max_legal_commitment(state))
        );
    }

    return actions;
}

std::vector<Action> BettingEngine::legal_facing_bet_actions(
    const PublicState& state,
    const BettingAbstraction& abstraction
) const {
    std::vector<Action> actions;

    push_unique_action(actions, fold_action());

    const int call_amount =
        std::min(amount_to_call(state), actor_stack(state));

    if (call_amount > 0) {
        push_unique_action(
            actions,
            call_action(call_amount)
        );
    }

    if (can_raise(state, abstraction)) {
        for (const BetSize& size : abstraction.raise_sizes) {
            const int amount =
                resolve_raise_to_amount(state, size, abstraction);

            if (amount <= 0) {
                continue;
            }

            const int max_commitment = max_legal_commitment(state);

            if (amount >= max_commitment) {
                push_unique_action(
                    actions,
                    all_in_action(max_commitment)
                );
            } else {
                push_unique_action(
                    actions,
                    raise_action(amount)
                );
            }
        }

        if (abstraction.always_allow_all_in) {
            push_unique_action(
                actions,
                all_in_action(max_legal_commitment(state))
            );
        }
    }

    return actions;
}

int BettingEngine::resolve_first_bet_amount(
    const PublicState& state,
    const BetSize& size,
    const BettingAbstraction& abstraction
) const {
    const int max_commitment = max_legal_commitment(state);

    int amount = 0;

    switch (size.type) {
        case BetSizeType::PotFraction:
            amount = static_cast<int>(
                std::llround(static_cast<double>(state.pot) * size.value)
            );
            break;

        case BetSizeType::FixedAmount:
            amount = static_cast<int>(std::llround(size.value));
            break;

        case BetSizeType::AllIn:
            return max_commitment;

        case BetSizeType::RaiseMultiplier:
            throw std::invalid_argument(
                "RaiseMultiplier is invalid for first bet sizing."
            );
    }

    amount = round_to_chip_unit(amount, abstraction.chip_unit);

    if (amount <= 0) {
        return 0;
    }

    return std::min(amount, max_commitment);
}

int BettingEngine::resolve_raise_to_amount(
    const PublicState& state,
    const BetSize& size,
    const BettingAbstraction& abstraction
) const {
    const int min_raise_to = min_legal_raise_to(state);
    const int max_commitment = max_legal_commitment(state);

    int amount = 0;

    switch (size.type) {
        case BetSizeType::RaiseMultiplier:
            amount = static_cast<int>(
                std::llround(
                    static_cast<double>(state.betting.current_bet_to_call) *
                    size.value
                )
            );
            break;

        case BetSizeType::FixedAmount:
            amount = static_cast<int>(std::llround(size.value));
            break;

        case BetSizeType::PotFraction:
            // Interpreted as: raise-to current bet + fraction of current pot.
            amount =
                state.betting.current_bet_to_call +
                static_cast<int>(
                    std::llround(static_cast<double>(state.pot) * size.value)
                );
            break;

        case BetSizeType::AllIn:
            return max_commitment;
    }

    amount = round_to_chip_unit(amount, abstraction.chip_unit);

    if (amount < min_raise_to) {
        return 0;
    }

    return std::min(amount, max_commitment);
}

int BettingEngine::actor_stack(
    const PublicState& state
) const {
    return state.stack(state.player_to_act);
}

int BettingEngine::opponent_stack(
    const PublicState& state
) const {
    return state.stack(opponent_of(state.player_to_act));
}

int BettingEngine::actor_committed(
    const PublicState& state
) const {
    return state.betting.committed(state.player_to_act);
}

int BettingEngine::opponent_committed(
    const PublicState& state
) const {
    return state.betting.committed(opponent_of(state.player_to_act));
}

int BettingEngine::amount_to_call(
    const PublicState& state
) const {
    return state.betting.amount_to_call(state.player_to_act);
}

int BettingEngine::actor_total_stack_available_this_street(
    const PublicState& state
) const {
    return actor_stack(state) + actor_committed(state);
}

int BettingEngine::min_legal_raise_to(
    const PublicState& state
) const {
    if (!state.betting.round_has_bet) {
        throw std::invalid_argument(
            "min_legal_raise_to requires an existing bet."
        );
    }

    const int last_raise =
        std::max(1, state.betting.last_raise_size);

    return state.betting.current_bet_to_call + last_raise;
}

int BettingEngine::max_legal_commitment(
    const PublicState& state
) const {
    return actor_committed(state) + actor_stack(state);
}

bool BettingEngine::can_raise(
    const PublicState& state,
    const BettingAbstraction& abstraction
) const {
    if (!is_facing_bet(state)) {
        return false;
    }

    if (state.betting.num_raises_this_street >=
        abstraction.max_raises_per_street) {
        return false;
    }

    const int max_commitment = max_legal_commitment(state);
    const int min_raise_to = min_legal_raise_to(state);

    return max_commitment >= min_raise_to;
}

bool BettingEngine::is_unopened_pot(
    const PublicState& state
) const {
    return state.betting.unopened() &&
           state.betting.amount_to_call(state.player_to_act) == 0;
}

bool BettingEngine::is_facing_bet(
    const PublicState& state
) const {
    return state.betting.player_is_facing_bet(state.player_to_act);
}

Player BettingEngine::opponent_of(
    Player player
) const {
    return poker::opponent_of(player);
}

int BettingEngine::round_to_chip_unit(
    int amount,
    int chip_unit
) const {
    if (chip_unit <= 0) {
        throw std::invalid_argument("chip_unit must be positive.");
    }

    if (amount <= 0) {
        return 0;
    }

    const int remainder = amount % chip_unit;

    if (remainder == 0) {
        return amount;
    }

    return amount + (chip_unit - remainder);
}

void BettingEngine::commit_chips(
    PublicState& state,
    Player player,
    int additional_chips
) const {
    if (player != Player::P0 && player != Player::P1) {
        throw std::invalid_argument(
            "commit_chips requires P0 or P1."
        );
    }

    if (additional_chips < 0) {
        throw std::invalid_argument(
            "Cannot commit negative chips."
        );
    }

    if (additional_chips == 0) {
        return;
    }

    const int stack_before = state.stack(player);

    if (additional_chips > stack_before) {
        throw std::invalid_argument(
            "Cannot commit more chips than player has."
        );
    }

    state.set_stack(player, stack_before - additional_chips);
    state.betting.add_committed(player, additional_chips);
    state.pot += additional_chips;
}

void BettingEngine::switch_player_to_act(
    PublicState& state
) const {
    state.player_to_act = opponent_of(state.player_to_act);
}

void BettingEngine::append_action_history(
    PublicState& state,
    const Action& action
) const {
    state.action_history.push_back(action);
}

void BettingEngine::validate_action_basic(
    const PublicState& state,
    const Action& action
) const {
    if (state.terminal) {
        throw std::invalid_argument(
            "Cannot apply an action to a terminal state."
        );
    }

    if (state.player_to_act != Player::P0 &&
        state.player_to_act != Player::P1) {
        throw std::invalid_argument(
            "Action requires P0 or P1 to act."
        );
    }

    action.validate();

    if (actor_stack(state) <= 0) {
        throw std::invalid_argument(
            "Player to act has no chips."
        );
    }

    if (action.type == ActionType::Fold ||
        action.type == ActionType::Call ||
        action.type == ActionType::Raise) {
        if (!is_facing_bet(state)) {
            throw std::invalid_argument(
                "Fold, Call, and Raise require facing a bet."
            );
        }
    }

    if (action.type == ActionType::Check ||
        action.type == ActionType::Bet) {
        if (!is_unopened_pot(state)) {
            throw std::invalid_argument(
                "Check and Bet require an unopened betting round."
            );
        }
    }
}

} // namespace poker::holdem