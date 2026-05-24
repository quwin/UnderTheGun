#include "game.hpp"

#include "holdem/action.hpp"
#include "holdem/betting_abstraction.hpp"
#include "holdem/betting_engine.hpp"
#include "holdem/public_state.hpp"
#include "holdem/street.hpp"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kStartingPot = 1000;
constexpr int kStartingStack = 2000;

void check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void check_eq(
    int actual,
    int expected,
    const std::string& message
) {
    if (actual != expected) {
        std::ostringstream oss;
        oss << message
            << " actual=" << actual
            << " expected=" << expected;
        throw std::runtime_error(oss.str());
    }
}

bool has_action_type(
    const std::vector<poker::holdem::Action>& actions,
    poker::holdem::ActionType type
) {
    return std::any_of(
        actions.begin(),
        actions.end(),
        [type](const poker::holdem::Action& action) {
            return action.type == type;
        }
    );
}

bool has_action(
    const std::vector<poker::holdem::Action>& actions,
    poker::holdem::ActionType type,
    int amount
) {
    return std::any_of(
        actions.begin(),
        actions.end(),
        [type, amount](const poker::holdem::Action& action) {
            return action.type == type && action.amount == amount;
        }
    );
}

poker::holdem::BettingAbstraction make_test_abstraction() {
    poker::holdem::BettingAbstraction abstraction;

    // Use deterministic integer sizing so tests do not depend on rounding
    // choices inside the betting engine.
    abstraction.first_bet_sizes = {
        poker::holdem::BetSize::pot_fraction(0.5),
        poker::holdem::BetSize::pot_fraction(1.0),
        poker::holdem::BetSize::all_in()
    };

    abstraction.raise_sizes = {
        poker::holdem::BetSize::raise_multiplier(2.5),
        poker::holdem::BetSize::all_in()
    };

    abstraction.max_raises_per_street = 2;

    return abstraction;
}

poker::holdem::PublicState make_initial_river_state() {
    poker::holdem::PublicState state;

    state.street = poker::holdem::Street::River;
    state.pot = kStartingPot;
    state.p0_stack = kStartingStack;
    state.p1_stack = kStartingStack;
    state.p0_committed_this_round = 0;
    state.p1_committed_this_round = 0;
    state.current_bet_to_call = 0;
    state.last_raise_size = 0;
    state.num_raises_this_street = 0;
    state.player_to_act = poker::Player::P0;
    state.last_aggressor = poker::Player::Chance;
    state.round_has_bet = false;
    state.last_action_was_check = false;
    state.terminal = false;
    state.terminal_reason = poker::holdem::TerminalReason::None;
    state.action_history.clear();

    return state;
}

poker::holdem::Action find_action(
    const std::vector<poker::holdem::Action>& actions,
    poker::holdem::ActionType type,
    int amount = -1
) {
    for (const poker::holdem::Action& action : actions) {
        if (action.type != type) {
            continue;
        }

        if (amount >= 0 && action.amount != amount) {
            continue;
        }

        return action;
    }

    std::ostringstream oss;
    oss << "Could not find action type="
        << static_cast<int>(type)
        << " amount="
        << amount;
    throw std::runtime_error(oss.str());
}

void test_unopened_pot_has_check_and_bets() {
    const poker::holdem::BettingEngine engine;
    const poker::holdem::BettingAbstraction abstraction =
        make_test_abstraction();

    const poker::holdem::PublicState state = make_initial_river_state();

    const std::vector<poker::holdem::Action> actions =
        engine.legal_actions(state, abstraction);

    check(
        has_action_type(actions, poker::holdem::ActionType::Check),
        "Unopened pot should allow check."
    );

    check(
        has_action(actions, poker::holdem::ActionType::Bet, 500),
        "Unopened pot should allow 50% pot bet."
    );

    check(
        has_action(actions, poker::holdem::ActionType::Bet, 1000),
        "Unopened pot should allow pot-sized bet."
    );

    check(
        has_action(actions, poker::holdem::ActionType::AllIn, kStartingStack),
        "Unopened pot should allow all-in."
    );

    check(
        !has_action_type(actions, poker::holdem::ActionType::Call),
        "Unopened pot should not allow call."
    );

    check(
        !has_action_type(actions, poker::holdem::ActionType::Fold),
        "Unopened pot should not allow fold."
    );

    std::cout << "[pass] test_unopened_pot_has_check_and_bets\n";
}

void test_check_switches_player_to_act() {
    const poker::holdem::BettingEngine engine;
    const poker::holdem::BettingAbstraction abstraction =
        make_test_abstraction();

    const poker::holdem::PublicState state = make_initial_river_state();

    const poker::holdem::Action check_action{
        poker::holdem::ActionType::Check,
        0
    };

    const poker::holdem::PublicState next =
        engine.apply_action(state, check_action);

    check(
        next.player_to_act == poker::Player::P1,
        "After P0 checks, P1 should act."
    );

    check_eq(next.pot, kStartingPot, "Check should not change pot.");
    check_eq(next.p0_stack, kStartingStack, "Check should not change P0 stack.");
    check_eq(next.p1_stack, kStartingStack, "Check should not change P1 stack.");
    check_eq(next.current_bet_to_call, 0, "Check should not create a bet to call.");

    check(
        !engine.betting_round_closed(next),
        "One check should not close the betting round."
    );

    std::cout << "[pass] test_check_switches_player_to_act\n";
}

void test_check_check_closes_round() {
    const poker::holdem::BettingEngine engine;

    poker::holdem::PublicState state = make_initial_river_state();

    state = engine.apply_action(
        state,
        poker::holdem::Action{poker::holdem::ActionType::Check, 0}
    );

    state = engine.apply_action(
        state,
        poker::holdem::Action{poker::holdem::ActionType::Check, 0}
    );

    check(
        engine.betting_round_closed(state),
        "Check-check should close the betting round."
    );

    check_eq(state.pot, kStartingPot, "Check-check should not change pot.");
    check_eq(state.current_bet_to_call, 0, "Check-check should leave call amount at zero.");

    std::cout << "[pass] test_check_check_closes_round\n";
}

void test_bet_updates_pot_stack_and_player_to_act() {
    const poker::holdem::BettingEngine engine;

    const poker::holdem::PublicState state = make_initial_river_state();

    const poker::holdem::PublicState next =
        engine.apply_action(
            state,
            poker::holdem::Action{poker::holdem::ActionType::Bet, 500}
        );

    check_eq(next.pot, 1500, "Bet should add chips to the pot.");
    check_eq(next.p0_stack, 1500, "Bet should subtract chips from bettor stack.");
    check_eq(next.p1_stack, 2000, "Bet should not affect opponent stack yet.");

    check_eq(
        next.p0_committed_this_round,
        500,
        "Bet should update bettor committed amount."
    );

    check_eq(
        next.p1_committed_this_round,
        0,
        "Bet should not update opponent committed amount."
    );

    check_eq(
        next.current_bet_to_call,
        500,
        "Bet should create a call amount."
    );

    check(
        next.player_to_act == poker::Player::P1,
        "After P0 bets, P1 should act."
    );

    check(
        next.last_aggressor == poker::Player::P0,
        "P0 should be recorded as last aggressor."
    );

    check(
        !engine.betting_round_closed(next),
        "A bet alone should not close the betting round."
    );

    std::cout << "[pass] test_bet_updates_pot_stack_and_player_to_act\n";
}

void test_facing_bet_has_fold_call_and_raises() {
    const poker::holdem::BettingEngine engine;
    const poker::holdem::BettingAbstraction abstraction =
        make_test_abstraction();

    poker::holdem::PublicState state = make_initial_river_state();

    state = engine.apply_action(
        state,
        poker::holdem::Action{poker::holdem::ActionType::Bet, 500}
    );

    const std::vector<poker::holdem::Action> actions =
        engine.legal_actions(state, abstraction);

    check(
        has_action_type(actions, poker::holdem::ActionType::Fold),
        "Facing a bet should allow fold."
    );

    check(
        has_action(actions, poker::holdem::ActionType::Call, 500),
        "Facing a 500 bet should allow call for 500."
    );

    check(
        has_action_type(actions, poker::holdem::ActionType::Raise),
        "Facing a bet should allow at least one abstract raise."
    );

    check(
        has_action(actions, poker::holdem::ActionType::AllIn, 2000),
        "Facing a bet should allow all-in when stack remains."
    );

    check(
        !has_action_type(actions, poker::holdem::ActionType::Check),
        "Facing a bet should not allow check."
    );

    check(
        !has_action_type(actions, poker::holdem::ActionType::Bet),
        "Facing a bet should not allow a fresh bet action."
    );

    std::cout << "[pass] test_facing_bet_has_fold_call_and_raises\n";
}

void test_call_closes_round_and_equalizes_commitments() {
    const poker::holdem::BettingEngine engine;

    poker::holdem::PublicState state = make_initial_river_state();

    state = engine.apply_action(
        state,
        poker::holdem::Action{poker::holdem::ActionType::Bet, 500}
    );

    state = engine.apply_action(
        state,
        poker::holdem::Action{poker::holdem::ActionType::Call, 500}
    );

    check_eq(state.pot, 2000, "Call should add matching chips to pot.");
    check_eq(state.p0_stack, 1500, "P0 stack after bet-call mismatch.");
    check_eq(state.p1_stack, 1500, "P1 stack after call mismatch.");

    check_eq(
        state.p0_committed_this_round,
        500,
        "P0 committed amount should remain 500."
    );

    check_eq(
        state.p1_committed_this_round,
        500,
        "P1 committed amount should become 500 after call."
    );

    check(
        engine.betting_round_closed(state),
        "Bet-call should close the betting round."
    );

    std::cout << "[pass] test_call_closes_round_and_equalizes_commitments\n";
}

void test_fold_creates_fold_terminal_state() {
    const poker::holdem::BettingEngine engine;

    poker::holdem::PublicState state = make_initial_river_state();

    state = engine.apply_action(
        state,
        poker::holdem::Action{poker::holdem::ActionType::Bet, 500}
    );

    state = engine.apply_action(
        state,
        poker::holdem::Action{poker::holdem::ActionType::Fold, 0}
    );

    check(state.terminal, "Fold should create terminal state.");

    check(
        state.terminal_reason == poker::holdem::TerminalReason::Fold,
        "Fold terminal state should have Fold reason."
    );

    check(
        state.folded_player == poker::Player::P1,
        "P1 should be marked as folded player."
    );

    check(
        state.winner == poker::Player::P0,
        "P0 should win after P1 folds to P0's bet."
    );

    check_eq(
        state.pot,
        1500,
        "Fold should not add more chips to pot."
    );

    std::cout << "[pass] test_fold_creates_fold_terminal_state\n";
}

void test_raise_updates_call_amount_and_raise_count() {
    const poker::holdem::BettingEngine engine;

    poker::holdem::PublicState state = make_initial_river_state();

    state = engine.apply_action(
        state,
        poker::holdem::Action{poker::holdem::ActionType::Bet, 500}
    );

    // Interpret Action::amount for Raise as the raise-to amount for the acting
    // player's total committed amount on this street.
    state = engine.apply_action(
        state,
        poker::holdem::Action{poker::holdem::ActionType::Raise, 1250}
    );

    check_eq(state.pot, 2250, "Raise-to 1250 should add 1250 chips from P1.");
    check_eq(state.p0_stack, 1500, "P0 stack should still reflect original 500 bet.");
    check_eq(state.p1_stack, 750, "P1 stack should subtract raise-to amount.");

    check_eq(
        state.p1_committed_this_round,
        1250,
        "P1 committed amount should equal raise-to amount."
    );

    check_eq(
        state.current_bet_to_call,
        1250,
        "Current bet to call should be the raised-to amount."
    );

    check_eq(
        state.last_raise_size,
        750,
        "Last raise size should be raise-to amount minus previous bet."
    );

    check_eq(
        state.num_raises_this_street,
        1,
        "Raise count should increment."
    );

    check(
        state.player_to_act == poker::Player::P0,
        "After P1 raises, P0 should act."
    );

    check(
        state.last_aggressor == poker::Player::P1,
        "P1 should be the last aggressor after raising."
    );

    std::cout << "[pass] test_raise_updates_call_amount_and_raise_count\n";
}

void test_raise_cap_removes_raise_actions() {
    poker::holdem::BettingEngine engine;
    poker::holdem::BettingAbstraction abstraction =
        make_test_abstraction();

    abstraction.max_raises_per_street = 1;

    poker::holdem::PublicState state = make_initial_river_state();

    state = engine.apply_action(
        state,
        poker::holdem::Action{poker::holdem::ActionType::Bet, 500}
    );

    state = engine.apply_action(
        state,
        poker::holdem::Action{poker::holdem::ActionType::Raise, 1250}
    );

    const std::vector<poker::holdem::Action> actions =
        engine.legal_actions(state, abstraction);

    check(
        has_action_type(actions, poker::holdem::ActionType::Fold),
        "Facing capped raise should still allow fold."
    );

    check(
        has_action(actions, poker::holdem::ActionType::Call, 750),
        "Facing capped raise should allow call for difference."
    );

    check(
        !has_action_type(actions, poker::holdem::ActionType::Raise),
        "Raise cap should remove normal raise actions."
    );

    std::cout << "[pass] test_raise_cap_removes_raise_actions\n";
}

void test_all_in_clamps_to_stack() {
    const poker::holdem::BettingEngine engine;

    poker::holdem::PublicState state = make_initial_river_state();

    state = engine.apply_action(
        state,
        poker::holdem::Action{poker::holdem::ActionType::AllIn, kStartingStack}
    );

    check_eq(state.pot, 3000, "All-in should add full stack to pot.");
    check_eq(state.p0_stack, 0, "All-in bettor stack should become zero.");
    check_eq(state.p0_committed_this_round, 2000, "All-in committed mismatch.");
    check_eq(state.current_bet_to_call, 2000, "All-in should set call amount.");

    check(
        state.player_to_act == poker::Player::P1,
        "After P0 all-in, P1 should act."
    );

    std::cout << "[pass] test_all_in_clamps_to_stack\n";
}

void test_invalid_action_throws() {
    const poker::holdem::BettingEngine engine;

    const poker::holdem::PublicState state = make_initial_river_state();

    bool threw = false;

    try {
        // Cannot call in an unopened pot.
        (void)engine.apply_action(
            state,
            poker::holdem::Action{poker::holdem::ActionType::Call, 0}
        );
    } catch (const std::exception&) {
        threw = true;
    }

    check(threw, "Calling in unopened pot should throw.");

    std::cout << "[pass] test_invalid_action_throws\n";
}

void run_all_tests() {
    test_unopened_pot_has_check_and_bets();
    test_check_switches_player_to_act();
    test_check_check_closes_round();
    test_bet_updates_pot_stack_and_player_to_act();
    test_facing_bet_has_fold_call_and_raises();
    test_call_closes_round_and_equalizes_commitments();
    test_fold_creates_fold_terminal_state();
    test_raise_updates_call_amount_and_raise_count();
    test_raise_cap_removes_raise_actions();
    test_all_in_clamps_to_stack();
    test_invalid_action_throws();
}

} // namespace

int main() {
    try {
        run_all_tests();
    } catch (const std::exception& e) {
        std::cerr << "[fail] " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "[pass] all betting engine tests passed\n";
    return EXIT_SUCCESS;
}