#include "game.hpp"

#include "holdem/private_state.hpp"
#include "holdem/public_state.hpp"
#include "holdem/street.hpp"
#include "holdem/terminal_utility.hpp"

#include "poker/board.hpp"
#include "poker/card.hpp"
#include "poker/hand.hpp"
#include "poker/hand_evaluator.hpp"

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

constexpr double kTol = 1e-6;

void check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void check_near(
    double actual,
    double expected,
    double tolerance,
    const std::string& message
) {
    if (std::abs(actual - expected) > tolerance) {
        std::ostringstream oss;
        oss << message
            << " actual=" << actual
            << " expected=" << expected
            << " tolerance=" << tolerance;
        throw std::runtime_error(oss.str());
    }
}

poker::CardId c(poker::Rank rank, poker::Suit suit) {
    return poker::make_card(rank, suit);
}

poker::HoleCards hand(
    poker::CardId a,
    poker::CardId b
) {
    return poker::HoleCards{a, b};
}

poker::Board board(
    poker::CardId a,
    poker::CardId b,
    poker::CardId c0,
    poker::CardId d,
    poker::CardId e
) {
    return poker::Board{{a, b, c0, d, e}};
}

poker::holdem::PrivateState private_state(
    poker::HoleCards p0_hand,
    poker::HoleCards p1_hand
) {
    poker::holdem::PrivateState state;
    state.p0_hand = p0_hand;
    state.p1_hand = p1_hand;
    return state;
}

poker::holdem::PublicState make_base_river_state() {
    poker::holdem::PublicState state;

    state.street = poker::holdem::Street::River;

    state.board = board(
        c(poker::Rank::Ace, poker::Suit::Spades),
        c(poker::Rank::Seven, poker::Suit::Hearts),
        c(poker::Rank::Two, poker::Suit::Clubs),
        c(poker::Rank::Jack, poker::Suit::Diamonds),
        c(poker::Rank::Four, poker::Suit::Spades)
    );

    state.pot = 1000;

    state.p0_stack = 2000;
    state.p1_stack = 2000;

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
    state.folded_player = poker::Player::Terminal;
    state.winner = poker::Player::Terminal;

    state.action_history.clear();

    return state;
}

float utility_p0(
    const poker::holdem::PublicState& public_state,
    const poker::holdem::PrivateState& private_state
) {
    const poker::HandEvaluator evaluator;

    return poker::holdem::terminal_utility_p0(
        public_state,
        private_state,
        evaluator
    );
}

void test_p1_folds_to_p0_bet_p0_wins_current_pot_minus_own_commitment() {
    poker::holdem::PublicState state = make_base_river_state();

    // Initial pot before river action: 1000.
    // P0 bets 500.
    // P1 folds.
    //
    // Total pot at terminal = 1500.
    // From P0 net perspective for this subgame:
    //   P0 wins pot but already contributed 500 on this street.
    //   net = 1500 - 500 = +1000.
    state.pot = 1500;
    state.p0_committed_this_round = 500;
    state.p1_committed_this_round = 0;
    state.p0_stack = 1500;
    state.p1_stack = 2000;

    state.terminal = true;
    state.terminal_reason = poker::holdem::TerminalReason::Fold;
    state.folded_player = poker::Player::P1;
    state.winner = poker::Player::P0;

    const poker::holdem::PrivateState priv = private_state(
        hand(
            c(poker::Rank::King, poker::Suit::Hearts),
            c(poker::Rank::Queen, poker::Suit::Hearts)
        ),
        hand(
            c(poker::Rank::Ten, poker::Suit::Clubs),
            c(poker::Rank::Nine, poker::Suit::Clubs)
        )
    );

    check_near(
        utility_p0(state, priv),
        1000.0,
        kTol,
        "P0 utility after P1 folds to P0 bet is wrong."
    );

    std::cout << "[pass] test_p1_folds_to_p0_bet_p0_wins_current_pot_minus_own_commitment\n";
}

void test_p0_folds_to_p1_bet_p0_loses_own_commitment() {
    poker::holdem::PublicState state = make_base_river_state();

    // Initial pot before river action: 1000.
    // P0 checks, P1 bets 500, P0 folds.
    //
    // Total pot at terminal = 1500.
    // P0 contributed 0 on this street.
    // From subgame street-net perspective:
    //   net = -0.
    //
    // If your convention treats the starting pot as contestable equity, folding
    // after committing zero returns 0 for P0, not -1000.
    state.pot = 1500;
    state.p0_committed_this_round = 0;
    state.p1_committed_this_round = 500;
    state.p0_stack = 2000;
    state.p1_stack = 1500;

    state.terminal = true;
    state.terminal_reason = poker::holdem::TerminalReason::Fold;
    state.folded_player = poker::Player::P0;
    state.winner = poker::Player::P1;

    const poker::holdem::PrivateState priv = private_state(
        hand(
            c(poker::Rank::King, poker::Suit::Hearts),
            c(poker::Rank::Queen, poker::Suit::Hearts)
        ),
        hand(
            c(poker::Rank::Ten, poker::Suit::Clubs),
            c(poker::Rank::Nine, poker::Suit::Clubs)
        )
    );

    check_near(
        utility_p0(state, priv),
        0.0,
        kTol,
        "P0 utility after folding with no river contribution is wrong."
    );

    std::cout << "[pass] test_p0_folds_to_p1_bet_p0_loses_own_commitment\n";
}

void test_p0_folds_after_calling_or_raising_loses_own_commitment() {
    poker::holdem::PublicState state = make_base_river_state();

    // Example terminal state:
    // Initial pot before street: 1000.
    // P0 has committed 800 on the street.
    // P1 has committed 1200 on the street.
    // P0 folds.
    //
    // Terminal pot = 3000.
    // P0's net utility by street-net convention = -800.
    state.pot = 3000;
    state.p0_committed_this_round = 800;
    state.p1_committed_this_round = 1200;
    state.p0_stack = 1200;
    state.p1_stack = 800;

    state.terminal = true;
    state.terminal_reason = poker::holdem::TerminalReason::Fold;
    state.folded_player = poker::Player::P0;
    state.winner = poker::Player::P1;

    const poker::holdem::PrivateState priv = private_state(
        hand(
            c(poker::Rank::King, poker::Suit::Hearts),
            c(poker::Rank::Queen, poker::Suit::Hearts)
        ),
        hand(
            c(poker::Rank::Ten, poker::Suit::Clubs),
            c(poker::Rank::Nine, poker::Suit::Clubs)
        )
    );

    check_near(
        utility_p0(state, priv),
        -800.0,
        kTol,
        "P0 utility after folding with committed chips is wrong."
    );

    std::cout << "[pass] test_p0_folds_after_calling_or_raising_loses_own_commitment\n";
}

void test_p0_wins_showdown_gets_pot_minus_own_commitment() {
    poker::holdem::PublicState state = make_base_river_state();

    // Board: As 7h 2c Jd 4s.
    // P0: Ah Kh -> pair of aces.
    // P1: Tc 9c -> ace-high board only.
    //
    // Initial pot 1000, bet-call 500 each => terminal pot 2000.
    // P0 committed 500 on this street.
    // net = 2000 - 500 = +1500.
    state.pot = 2000;
    state.p0_committed_this_round = 500;
    state.p1_committed_this_round = 500;
    state.p0_stack = 1500;
    state.p1_stack = 1500;

    state.terminal = true;
    state.terminal_reason = poker::holdem::TerminalReason::Showdown;
    state.folded_player = poker::Player::Terminal;
    state.winner = poker::Player::Terminal;

    const poker::holdem::PrivateState priv = private_state(
        hand(
            c(poker::Rank::Ace, poker::Suit::Hearts),
            c(poker::Rank::King, poker::Suit::Hearts)
        ),
        hand(
            c(poker::Rank::Ten, poker::Suit::Clubs),
            c(poker::Rank::Nine, poker::Suit::Clubs)
        )
    );

    check_near(
        utility_p0(state, priv),
        1500.0,
        kTol,
        "P0 showdown win utility is wrong."
    );

    std::cout << "[pass] test_p0_wins_showdown_gets_pot_minus_own_commitment\n";
}

void test_p0_loses_showdown_loses_own_commitment() {
    poker::holdem::PublicState state = make_base_river_state();

    // Board: As 7h 2c Jd 4s.
    // P0: Kh Qh -> ace-high board.
    // P1: Ah Tc -> pair of aces.
    //
    // Terminal pot 2000, P0 committed 500.
    // net = -500.
    state.pot = 2000;
    state.p0_committed_this_round = 500;
    state.p1_committed_this_round = 500;
    state.p0_stack = 1500;
    state.p1_stack = 1500;

    state.terminal = true;
    state.terminal_reason = poker::holdem::TerminalReason::Showdown;
    state.folded_player = poker::Player::Terminal;
    state.winner = poker::Player::Terminal;

    const poker::holdem::PrivateState priv = private_state(
        hand(
            c(poker::Rank::King, poker::Suit::Hearts),
            c(poker::Rank::Queen, poker::Suit::Hearts)
        ),
        hand(
            c(poker::Rank::Ace, poker::Suit::Hearts),
            c(poker::Rank::Ten, poker::Suit::Clubs)
        )
    );

    check_near(
        utility_p0(state, priv),
        -500.0,
        kTol,
        "P0 showdown loss utility is wrong."
    );

    std::cout << "[pass] test_p0_loses_showdown_loses_own_commitment\n";
}

void test_showdown_tie_splits_pot() {
    poker::holdem::PublicState state = make_base_river_state();

    // Board makes broadway straight for both:
    // Ts Jd Qc Kh As.
    state.board = board(
        c(poker::Rank::Ten, poker::Suit::Spades),
        c(poker::Rank::Jack, poker::Suit::Diamonds),
        c(poker::Rank::Queen, poker::Suit::Clubs),
        c(poker::Rank::King, poker::Suit::Hearts),
        c(poker::Rank::Ace, poker::Suit::Spades)
    );

    // Terminal pot 2000, both committed 500.
    // Split = 1000 returned to P0.
    // net = 1000 - 500 = +500.
    //
    // This includes P0's half of the starting pot. With a street-net
    // convention, ties can be positive if there was already a pot to split.
    state.pot = 2000;
    state.p0_committed_this_round = 500;
    state.p1_committed_this_round = 500;

    state.terminal = true;
    state.terminal_reason = poker::holdem::TerminalReason::Showdown;
    state.folded_player = poker::Player::Terminal;
    state.winner = poker::Player::Terminal;

    const poker::holdem::PrivateState priv = private_state(
        hand(
            c(poker::Rank::Two, poker::Suit::Hearts),
            c(poker::Rank::Three, poker::Suit::Hearts)
        ),
        hand(
            c(poker::Rank::Four, poker::Suit::Clubs),
            c(poker::Rank::Five, poker::Suit::Clubs)
        )
    );

    check_near(
        utility_p0(state, priv),
        500.0,
        kTol,
        "Showdown split-pot utility is wrong."
    );

    std::cout << "[pass] test_showdown_tie_splits_pot\n";
}

void test_all_in_showdown_win_uses_full_terminal_pot() {
    poker::holdem::PublicState state = make_base_river_state();

    // Initial pot 1000, both players all-in for 2000.
    // Terminal pot = 5000.
    // P0 committed = 2000.
    // P0 wins net = 5000 - 2000 = +3000.
    state.pot = 5000;
    state.p0_committed_this_round = 2000;
    state.p1_committed_this_round = 2000;
    state.p0_stack = 0;
    state.p1_stack = 0;

    state.terminal = true;
    state.terminal_reason = poker::holdem::TerminalReason::Showdown;
    state.folded_player = poker::Player::Terminal;
    state.winner = poker::Player::Terminal;

    const poker::holdem::PrivateState priv = private_state(
        hand(
            c(poker::Rank::Ace, poker::Suit::Hearts),
            c(poker::Rank::King, poker::Suit::Hearts)
        ),
        hand(
            c(poker::Rank::Ten, poker::Suit::Clubs),
            c(poker::Rank::Nine, poker::Suit::Clubs)
        )
    );

    check_near(
        utility_p0(state, priv),
        3000.0,
        kTol,
        "All-in showdown win utility is wrong."
    );

    std::cout << "[pass] test_all_in_showdown_win_uses_full_terminal_pot\n";
}

void test_terminal_utility_rejects_nonterminal_state() {
    poker::holdem::PublicState state = make_base_river_state();

    state.terminal = false;
    state.terminal_reason = poker::holdem::TerminalReason::None;

    const poker::holdem::PrivateState priv = private_state(
        hand(
            c(poker::Rank::Ace, poker::Suit::Hearts),
            c(poker::Rank::King, poker::Suit::Hearts)
        ),
        hand(
            c(poker::Rank::Ten, poker::Suit::Clubs),
            c(poker::Rank::Nine, poker::Suit::Clubs)
        )
    );

    bool threw = false;

    try {
        (void)utility_p0(state, priv);
    } catch (const std::exception&) {
        threw = true;
    }

    check(
        threw,
        "terminal_utility_p0 should reject nonterminal state."
    );

    std::cout << "[pass] test_terminal_utility_rejects_nonterminal_state\n";
}

void test_fold_terminal_requires_folded_player() {
    poker::holdem::PublicState state = make_base_river_state();

    state.terminal = true;
    state.terminal_reason = poker::holdem::TerminalReason::Fold;
    state.folded_player = poker::Player::Terminal;
    state.winner = poker::Player::Terminal;

    const poker::holdem::PrivateState priv = private_state(
        hand(
            c(poker::Rank::Ace, poker::Suit::Hearts),
            c(poker::Rank::King, poker::Suit::Hearts)
        ),
        hand(
            c(poker::Rank::Ten, poker::Suit::Clubs),
            c(poker::Rank::Nine, poker::Suit::Clubs)
        )
    );

    bool threw = false;

    try {
        (void)utility_p0(state, priv);
    } catch (const std::exception&) {
        threw = true;
    }

    check(
        threw,
        "Fold terminal should require folded_player."
    );

    std::cout << "[pass] test_fold_terminal_requires_folded_player\n";
}

void test_showdown_terminal_requires_river_board() {
    poker::holdem::PublicState state = make_base_river_state();

    state.board = poker::Board{
        {
            c(poker::Rank::Ace, poker::Suit::Spades),
            c(poker::Rank::Seven, poker::Suit::Hearts),
            c(poker::Rank::Two, poker::Suit::Clubs),
            c(poker::Rank::Jack, poker::Suit::Diamonds)
        }
    };

    state.terminal = true;
    state.terminal_reason = poker::holdem::TerminalReason::Showdown;

    const poker::holdem::PrivateState priv = private_state(
        hand(
            c(poker::Rank::Ace, poker::Suit::Hearts),
            c(poker::Rank::King, poker::Suit::Hearts)
        ),
        hand(
            c(poker::Rank::Ten, poker::Suit::Clubs),
            c(poker::Rank::Nine, poker::Suit::Clubs)
        )
    );

    bool threw = false;

    try {
        (void)utility_p0(state, priv);
    } catch (const std::exception&) {
        threw = true;
    }

    check(
        threw,
        "Showdown terminal should require a five-card river board."
    );

    std::cout << "[pass] test_showdown_terminal_requires_river_board\n";
}

void run_all_tests() {
    test_p1_folds_to_p0_bet_p0_wins_current_pot_minus_own_commitment();
    test_p0_folds_to_p1_bet_p0_loses_own_commitment();
    test_p0_folds_after_calling_or_raising_loses_own_commitment();

    test_p0_wins_showdown_gets_pot_minus_own_commitment();
    test_p0_loses_showdown_loses_own_commitment();
    test_showdown_tie_splits_pot();
    test_all_in_showdown_win_uses_full_terminal_pot();

    test_terminal_utility_rejects_nonterminal_state();
    test_fold_terminal_requires_folded_player();
    test_showdown_terminal_requires_river_board();
}

} // namespace

int main() {
    try {
        run_all_tests();
    } catch (const std::exception& e) {
        std::cerr << "[fail] " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "[pass] all terminal utility tests passed\n";
    return EXIT_SUCCESS;
}