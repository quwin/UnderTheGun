#include "game.hpp"

#include "holdem/infoset_key.hpp"
#include "holdem/private_state.hpp"
#include "holdem/public_state.hpp"
#include "holdem/street.hpp"

#include "poker/board.hpp"
#include "poker/card.hpp"
#include "poker/hand.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

void check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void check_eq(
    const std::string& actual,
    const std::string& expected,
    const std::string& message
) {
    if (actual != expected) {
        std::ostringstream oss;
        oss << message
            << "\nactual:   " << actual
            << "\nexpected: " << expected;
        throw std::runtime_error(oss.str());
    }
}

void check_ne(
    const std::string& a,
    const std::string& b,
    const std::string& message
) {
    if (a == b) {
        std::ostringstream oss;
        oss << message
            << "\nboth: " << a;
        throw std::runtime_error(oss.str());
    }
}

poker::HoleCards hand(
    poker::CardId a,
    poker::CardId b
) {
    return poker::HoleCards{a, b};
}

poker::Board make_river_board() {
    return poker::Board{
        {
            poker::make_card(poker::Rank::Ace, poker::Suit::Spades),
            poker::make_card(poker::Rank::Seven, poker::Suit::Hearts),
            poker::make_card(poker::Rank::Two, poker::Suit::Clubs),
            poker::make_card(poker::Rank::Jack, poker::Suit::Diamonds),
            poker::make_card(poker::Rank::Four, poker::Suit::Spades)
        }
    };
}

poker::Board make_turn_board() {
    return poker::Board{
            {
                poker::make_card(poker::Rank::Ace, poker::Suit::Spades),
                poker::make_card(poker::Rank::Seven, poker::Suit::Hearts),
                poker::make_card(poker::Rank::Two, poker::Suit::Clubs),
                poker::make_card(poker::Rank::Jack, poker::Suit::Diamonds),
            }
    };
}

poker::Board make_different_river_board() {
    return poker::Board{
        {
            poker::make_card(poker::Rank::Ace, poker::Suit::Spades),
            poker::make_card(poker::Rank::Seven, poker::Suit::Hearts),
            poker::make_card(poker::Rank::Two, poker::Suit::Clubs),
            poker::make_card(poker::Rank::Jack, poker::Suit::Diamonds),
            poker::make_card(poker::Rank::King, poker::Suit::Spades)
        }
    };
}

poker::holdem::PublicState make_public_state() {
    poker::holdem::PublicState state;

    state.street = poker::holdem::Street::River;
    state.board = make_river_board();

    state.pot = 1000;
    state.p0_stack = 2000;
    state.p1_stack = 2000;


    state.player_to_act = poker::Player::P0;

    state.terminal = false;
    state.terminal_reason = poker::holdem::TerminalReason::None;

    state.action_history.clear();
    state.betting.reset_for_new_street();

    return state;
}

poker::holdem::PrivateState make_private_state(
    poker::HoleCards p0,
    poker::HoleCards p1
) {
    poker::holdem::PrivateState state;
    state.p0_hand = p0;
    state.p1_hand = p1;
    return state;
}

std::string key_string(
    poker::Player player,
    const poker::holdem::PublicState& public_state,
    const poker::holdem::PrivateState& private_state
) {
    return poker::holdem::to_string(
        poker::holdem::make_infoset_key(
            player,
            public_state,
            private_state,
            poker::holdem::ExactHandAbstraction()
        )
    );
}

void test_p0_key_ignores_p1_private_hand() {
    const poker::holdem::PublicState pub = make_public_state();

    const poker::HoleCards p0_hand = hand(
        poker::make_card(poker::Rank::King, poker::Suit::Hearts),
        poker::make_card(poker::Rank::Queen, poker::Suit::Hearts)
    );

    const poker::HoleCards p1_hand_a = hand(
        poker::make_card(poker::Rank::Nine, poker::Suit::Clubs),
        poker::make_card(poker::Rank::Nine, poker::Suit::Diamonds)
    );

    const poker::HoleCards p1_hand_b = hand(
        poker::make_card(poker::Rank::Three, poker::Suit::Clubs),
        poker::make_card(poker::Rank::Three, poker::Suit::Diamonds)
    );

    const std::string key_a = key_string(
        poker::Player::P0,
        pub,
        make_private_state(p0_hand, p1_hand_a)
    );

    const std::string key_b = key_string(
        poker::Player::P0,
        pub,
        make_private_state(p0_hand, p1_hand_b)
    );

    check_eq(
        key_a,
        key_b,
        "P0 infoset key must ignore P1 private hand."
    );

    std::cout << "[pass] test_p0_key_ignores_p1_private_hand\n";
}

void test_p1_key_ignores_p0_private_hand() {
    poker::holdem::PublicState pub = make_public_state();
    pub.player_to_act = poker::Player::P1;

    const poker::HoleCards p0_hand_a = hand(
        poker::make_card(poker::Rank::King, poker::Suit::Hearts),
        poker::make_card(poker::Rank::Queen, poker::Suit::Hearts)
    );

    const poker::HoleCards p0_hand_b = hand(
        poker::make_card(poker::Rank::Eight, poker::Suit::Clubs),
        poker::make_card(poker::Rank::Eight, poker::Suit::Diamonds)
    );

    const poker::HoleCards p1_hand = hand(
        poker::make_card(poker::Rank::Nine, poker::Suit::Clubs),
        poker::make_card(poker::Rank::Nine, poker::Suit::Diamonds)
    );

    const std::string key_a = key_string(
        poker::Player::P1,
        pub,
        make_private_state(p0_hand_a, p1_hand)
    );

    const std::string key_b = key_string(
        poker::Player::P1,
        pub,
        make_private_state(p0_hand_b, p1_hand)
    );

    check_eq(
        key_a,
        key_b,
        "P1 infoset key must ignore P0 private hand."
    );

    std::cout << "[pass] test_p1_key_ignores_p0_private_hand\n";
}

void test_p0_key_changes_when_p0_private_hand_changes() {
    const poker::holdem::PublicState pub = make_public_state();

    const poker::HoleCards p0_hand_a = hand(
        poker::make_card(poker::Rank::King, poker::Suit::Hearts),
        poker::make_card(poker::Rank::Queen, poker::Suit::Hearts)
    );

    const poker::HoleCards p0_hand_b = hand(
        poker::make_card(poker::Rank::Ten, poker::Suit::Clubs),
        poker::make_card(poker::Rank::Ten, poker::Suit::Diamonds)
    );

    const poker::HoleCards p1_hand = hand(
        poker::make_card(poker::Rank::Nine, poker::Suit::Clubs),
        poker::make_card(poker::Rank::Nine, poker::Suit::Diamonds)
    );

    const std::string key_a = key_string(
        poker::Player::P0,
        pub,
        make_private_state(p0_hand_a, p1_hand)
    );

    const std::string key_b = key_string(
        poker::Player::P0,
        pub,
        make_private_state(p0_hand_b, p1_hand)
    );

    check_ne(
        key_a,
        key_b,
        "P0 infoset key should change when P0's own private hand changes."
    );

    std::cout << "[pass] test_p0_key_changes_when_p0_private_hand_changes\n";
}

void test_p1_key_changes_when_p1_private_hand_changes() {
    poker::holdem::PublicState pub = make_public_state();
    pub.player_to_act = poker::Player::P1;

    const poker::HoleCards p0_hand = hand(
        poker::make_card(poker::Rank::King, poker::Suit::Hearts),
        poker::make_card(poker::Rank::Queen, poker::Suit::Hearts)
    );

    const poker::HoleCards p1_hand_a = hand(
        poker::make_card(poker::Rank::Nine, poker::Suit::Clubs),
        poker::make_card(poker::Rank::Nine, poker::Suit::Diamonds)
    );

    const poker::HoleCards p1_hand_b = hand(
        poker::make_card(poker::Rank::Three, poker::Suit::Clubs),
        poker::make_card(poker::Rank::Three, poker::Suit::Diamonds)
    );

    const std::string key_a = key_string(
        poker::Player::P1,
        pub,
        make_private_state(p0_hand, p1_hand_a)
    );

    const std::string key_b = key_string(
        poker::Player::P1,
        pub,
        make_private_state(p0_hand, p1_hand_b)
    );

    check_ne(
        key_a,
        key_b,
        "P1 infoset key should change when P1's own private hand changes."
    );

    std::cout << "[pass] test_p1_key_changes_when_p1_private_hand_changes\n";
}

void test_key_changes_when_public_board_changes() {
    poker::holdem::PublicState pub_a = make_public_state();

    poker::holdem::PublicState pub_b = pub_a;
    pub_b.board = make_different_river_board();

    const poker::holdem::PrivateState private_state = make_private_state(
        hand(
            poker::make_card(poker::Rank::King, poker::Suit::Hearts),
            poker::make_card(poker::Rank::Queen, poker::Suit::Hearts)
        ),
        hand(
            poker::make_card(poker::Rank::Nine, poker::Suit::Clubs),
            poker::make_card(poker::Rank::Nine, poker::Suit::Diamonds)
        )
    );

    const std::string key_a = key_string(
        poker::Player::P0,
        pub_a,
        private_state
    );

    const std::string key_b = key_string(
        poker::Player::P0,
        pub_b,
        private_state
    );

    check_ne(
        key_a,
        key_b,
        "Infoset key should change when public board changes."
    );

    std::cout << "[pass] test_key_changes_when_public_board_changes\n";
}

void test_key_changes_when_public_betting_history_changes() {
    poker::holdem::PublicState pub_a = make_public_state();

    poker::holdem::PublicState pub_b = pub_a;
    pub_b.action_history.push_back(
        poker::holdem::Action{
            poker::holdem::ActionType::Bet,
            500
        }
    );
    pub_b.betting.add_committed(poker::Player::P0, 500);
    pub_b.p0_stack = 1500;
    pub_b.pot = 1500;
    pub_b.player_to_act = poker::Player::P1;
    pub_b.betting.last_aggressor = poker::Player::P0;

    const poker::holdem::PrivateState private_state = make_private_state(
        hand(
            poker::make_card(poker::Rank::King, poker::Suit::Hearts),
            poker::make_card(poker::Rank::Queen, poker::Suit::Hearts)
        ),
        hand(
            poker::make_card(poker::Rank::Nine, poker::Suit::Clubs),
            poker::make_card(poker::Rank::Nine, poker::Suit::Diamonds)
        )
    );

    const std::string key_a = key_string(
        poker::Player::P0,
        pub_a,
        private_state
    );

    const std::string key_b = key_string(
        poker::Player::P1,
        pub_b,
        private_state
    );

    check_ne(
        key_a,
        key_b,
        "Infoset key should change when public betting history changes."
    );

    std::cout << "[pass] test_key_changes_when_public_betting_history_changes\n";
}

void test_key_changes_when_stack_or_pot_state_changes() {
    poker::holdem::PublicState pub_a = make_public_state();

    poker::holdem::PublicState pub_b = pub_a;
    pub_b.pot = 1500;
    pub_b.p0_stack = 1500;
    pub_b.betting.set_committed(poker::Player::P0, 500);

    const poker::holdem::PrivateState private_state = make_private_state(
        hand(
            poker::make_card(poker::Rank::King, poker::Suit::Hearts),
            poker::make_card(poker::Rank::Queen, poker::Suit::Hearts)
        ),
        hand(
            poker::make_card(poker::Rank::Nine, poker::Suit::Clubs),
            poker::make_card(poker::Rank::Nine, poker::Suit::Diamonds)
        )
    );

    const std::string key_a = key_string(
        poker::Player::P0,
        pub_a,
        private_state
    );

    const std::string key_b = key_string(
        poker::Player::P0,
        pub_b,
        private_state
    );

    check_ne(
        key_a,
        key_b,
        "Infoset key should change when public pot/stack state changes."
    );

    std::cout << "[pass] test_key_changes_when_stack_or_pot_state_changes\n";
}

void test_key_changes_when_street_changes() {
    poker::holdem::PublicState pub_a = make_public_state();

    poker::holdem::PublicState pub_b = pub_a;
    pub_b.street = poker::holdem::Street::Turn;
    pub_b.board = make_turn_board();

    const poker::holdem::PrivateState private_state = make_private_state(
        hand(
            poker::make_card(poker::Rank::King, poker::Suit::Hearts),
            poker::make_card(poker::Rank::Queen, poker::Suit::Hearts)
        ),
        hand(
            poker::make_card(poker::Rank::Nine, poker::Suit::Clubs),
            poker::make_card(poker::Rank::Nine, poker::Suit::Diamonds)
        )
    );

    const std::string key_a = key_string(
        poker::Player::P0,
        pub_a,
        private_state
    );

    const std::string key_b = key_string(
        poker::Player::P0,
        pub_b,
        private_state
    );

    check_ne(
        key_a,
        key_b,
        "Infoset key should change when street changes."
    );

    std::cout << "[pass] test_key_changes_when_street_changes\n";
}

void test_key_string_does_not_contain_opponent_hand_literals_for_p0() {
    const poker::holdem::PublicState pub = make_public_state();

    const poker::HoleCards p0_hand = hand(
        poker::make_card(poker::Rank::King, poker::Suit::Hearts),
        poker::make_card(poker::Rank::Queen, poker::Suit::Hearts)
    );

    const poker::HoleCards p1_hand = hand(
        poker::make_card(poker::Rank::Nine, poker::Suit::Clubs),
        poker::make_card(poker::Rank::Nine, poker::Suit::Diamonds)
    );

    const std::string key = key_string(
        poker::Player::P0,
        pub,
        make_private_state(p0_hand, p1_hand)
    );

    check(
        key.find("9c") == std::string::npos,
        "P0 key should not contain P1 first card literal."
    );

    check(
        key.find("9d") == std::string::npos,
        "P0 key should not contain P1 second card literal."
    );

    check(
        key.find("p1_hand") == std::string::npos,
        "P0 key should not contain p1_hand field."
    );

    std::cout << "[pass] test_key_string_does_not_contain_opponent_hand_literals_for_p0\n";
}

void test_key_string_does_not_contain_opponent_hand_literals_for_p1() {
    poker::holdem::PublicState pub = make_public_state();
    pub.player_to_act = poker::Player::P1;

    const poker::HoleCards p0_hand = hand(
        poker::make_card(poker::Rank::King, poker::Suit::Hearts),
        poker::make_card(poker::Rank::Queen, poker::Suit::Hearts)
    );

    const poker::HoleCards p1_hand = hand(
        poker::make_card(poker::Rank::Nine, poker::Suit::Clubs),
        poker::make_card(poker::Rank::Nine, poker::Suit::Diamonds)
    );

    const std::string key = key_string(
        poker::Player::P1,
        pub,
        make_private_state(p0_hand, p1_hand)
    );

    check(
        key.find("Kh") == std::string::npos,
        "P1 key should not contain P0 first card literal."
    );

    check(
        key.find("Qh") == std::string::npos,
        "P1 key should not contain P0 second card literal."
    );

    check(
        key.find("p0_hand") == std::string::npos,
        "P1 key should not contain p0_hand field."
    );

    std::cout << "[pass] test_key_string_does_not_contain_opponent_hand_literals_for_p1\n";
}

void test_key_rejects_non_acting_or_invalid_player() {
    const poker::holdem::PublicState pub = make_public_state();

    const poker::holdem::PrivateState private_state = make_private_state(
        hand(
            poker::make_card(poker::Rank::King, poker::Suit::Hearts),
            poker::make_card(poker::Rank::Queen, poker::Suit::Hearts)
        ),
        hand(
            poker::make_card(poker::Rank::Nine, poker::Suit::Clubs),
            poker::make_card(poker::Rank::Nine, poker::Suit::Diamonds)
        )
    );

    bool threw = false;
    try {
        (void)poker::holdem::make_infoset_key(
            poker::Player::Chance,
            pub,
            private_state,
            poker::holdem::ExactHandAbstraction()
        );
    } catch (const std::exception&) {
        threw = true;
    }

    check(
        threw,
        "Infoset key creation should reject Chance as acting player."
    );

    std::cout << "[pass] test_key_rejects_non_acting_or_invalid_player\n";
}

void run_all_tests() {
    test_p0_key_ignores_p1_private_hand();
    test_p1_key_ignores_p0_private_hand();

    test_p0_key_changes_when_p0_private_hand_changes();
    test_p1_key_changes_when_p1_private_hand_changes();

    test_key_changes_when_public_board_changes();
    test_key_changes_when_public_betting_history_changes();
    test_key_changes_when_stack_or_pot_state_changes();
    test_key_changes_when_street_changes();

    test_key_string_does_not_contain_opponent_hand_literals_for_p0();
    test_key_string_does_not_contain_opponent_hand_literals_for_p1();

    test_key_rejects_non_acting_or_invalid_player();
}

} // namespace

int main() {
    try {
        run_all_tests();
    } catch (const std::exception& e) {
        std::cerr << "[fail] " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "[pass] all infoset key privacy tests passed\n";
    return EXIT_SUCCESS;
}