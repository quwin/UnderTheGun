#include "poker/board.hpp"
#include "poker/card.hpp"
#include "poker/hand.hpp"
#include "poker/hand_evaluator.hpp"

#include <array>
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

void check_category(
    const poker::HandStrength& strength,
    poker::HandCategory expected,
    const std::string& message
) {
    check_eq(
        static_cast<int>(strength.category),
        static_cast<int>(expected),
        message
    );
}

void check_compare_positive(
    int result,
    const std::string& message
) {
    check(result > 0, message);
}

void check_compare_negative(
    int result,
    const std::string& message
) {
    check(result < 0, message);
}

void check_compare_zero(
    int result,
    const std::string& message
) {
    check(result == 0, message);
}

void test_high_card_category() {
    const poker::HandEvaluator evaluator;

    const poker::Board b = board(
        c(poker::Rank::Two, poker::Suit::Clubs),
        c(poker::Rank::Seven, poker::Suit::Diamonds),
        c(poker::Rank::Nine, poker::Suit::Hearts),
        c(poker::Rank::Jack, poker::Suit::Spades),
        c(poker::Rank::King, poker::Suit::Clubs)
    );

    const poker::HoleCards h = hand(
        c(poker::Rank::Ace, poker::Suit::Diamonds),
        c(poker::Rank::Three, poker::Suit::Hearts)
    );

    const poker::HandStrength strength = evaluator.evaluate_7(h, b);

    check_category(
        strength,
        poker::HandCategory::HighCard,
        "Expected high-card category."
    );

    check_eq(
        strength.ranks[0],
        static_cast<int>(poker::Rank::Ace),
        "High-card hand should use ace as top kicker."
    );

    std::cout << "[pass] test_high_card_category\n";
}

void test_one_pair_category() {
    const poker::HandEvaluator evaluator;

    const poker::Board b = board(
        c(poker::Rank::Two, poker::Suit::Clubs),
        c(poker::Rank::Seven, poker::Suit::Diamonds),
        c(poker::Rank::Nine, poker::Suit::Hearts),
        c(poker::Rank::Jack, poker::Suit::Spades),
        c(poker::Rank::King, poker::Suit::Clubs)
    );

    const poker::HoleCards h = hand(
        c(poker::Rank::Ace, poker::Suit::Diamonds),
        c(poker::Rank::Ace, poker::Suit::Hearts)
    );

    const poker::HandStrength strength = evaluator.evaluate_7(h, b);

    check_category(
        strength,
        poker::HandCategory::OnePair,
        "Expected one-pair category."
    );

    check_eq(
        strength.ranks[0],
        static_cast<int>(poker::Rank::Ace),
        "Pair should be aces."
    );

    std::cout << "[pass] test_one_pair_category\n";
}

void test_two_pair_category() {
    const poker::HandEvaluator evaluator;

    const poker::Board b = board(
        c(poker::Rank::Two, poker::Suit::Clubs),
        c(poker::Rank::Two, poker::Suit::Diamonds),
        c(poker::Rank::Nine, poker::Suit::Hearts),
        c(poker::Rank::Jack, poker::Suit::Spades),
        c(poker::Rank::King, poker::Suit::Clubs)
    );

    const poker::HoleCards h = hand(
        c(poker::Rank::Ace, poker::Suit::Diamonds),
        c(poker::Rank::Ace, poker::Suit::Hearts)
    );

    const poker::HandStrength strength = evaluator.evaluate_7(h, b);

    check_category(
        strength,
        poker::HandCategory::TwoPair,
        "Expected two-pair category."
    );

    check_eq(
        strength.ranks[0],
        static_cast<int>(poker::Rank::Ace),
        "Top pair should be aces."
    );

    check_eq(
        strength.ranks[1],
        static_cast<int>(poker::Rank::Two),
        "Second pair should be twos."
    );

    std::cout << "[pass] test_two_pair_category\n";
}

void test_three_of_a_kind_category() {
    const poker::HandEvaluator evaluator;

    const poker::Board b = board(
        c(poker::Rank::Ace, poker::Suit::Clubs),
        c(poker::Rank::Ace, poker::Suit::Diamonds),
        c(poker::Rank::Nine, poker::Suit::Hearts),
        c(poker::Rank::Jack, poker::Suit::Spades),
        c(poker::Rank::King, poker::Suit::Clubs)
    );

    const poker::HoleCards h = hand(
        c(poker::Rank::Ace, poker::Suit::Spades),
        c(poker::Rank::Three, poker::Suit::Hearts)
    );

    const poker::HandStrength strength = evaluator.evaluate_7(h, b);

    check_category(
        strength,
        poker::HandCategory::ThreeOfAKind,
        "Expected trips category."
    );

    check_eq(
        strength.ranks[0],
        static_cast<int>(poker::Rank::Ace),
        "Trips should be aces."
    );

    std::cout << "[pass] test_three_of_a_kind_category\n";
}

void test_straight_category_broadway() {
    const poker::HandEvaluator evaluator;

    const poker::Board b = board(
        c(poker::Rank::Ten, poker::Suit::Clubs),
        c(poker::Rank::Jack, poker::Suit::Diamonds),
        c(poker::Rank::Queen, poker::Suit::Hearts),
        c(poker::Rank::King, poker::Suit::Spades),
        c(poker::Rank::Two, poker::Suit::Clubs)
    );

    const poker::HoleCards h = hand(
        c(poker::Rank::Ace, poker::Suit::Diamonds),
        c(poker::Rank::Three, poker::Suit::Hearts)
    );

    const poker::HandStrength strength = evaluator.evaluate_7(h, b);

    check_category(
        strength,
        poker::HandCategory::Straight,
        "Expected broadway straight."
    );

    check_eq(
        strength.ranks[0],
        static_cast<int>(poker::Rank::Ace),
        "Broadway straight should be ace-high."
    );

    std::cout << "[pass] test_straight_category_broadway\n";
}

void test_straight_category_wheel() {
    const poker::HandEvaluator evaluator;

    const poker::Board b = board(
        c(poker::Rank::Ace, poker::Suit::Clubs),
        c(poker::Rank::Two, poker::Suit::Diamonds),
        c(poker::Rank::Three, poker::Suit::Hearts),
        c(poker::Rank::Four, poker::Suit::Spades),
        c(poker::Rank::Nine, poker::Suit::Clubs)
    );

    const poker::HoleCards h = hand(
        c(poker::Rank::Five, poker::Suit::Diamonds),
        c(poker::Rank::King, poker::Suit::Hearts)
    );

    const poker::HandStrength strength = evaluator.evaluate_7(h, b);

    check_category(
        strength,
        poker::HandCategory::Straight,
        "Expected wheel straight."
    );

    check_eq(
        strength.ranks[0],
        static_cast<int>(poker::Rank::Five),
        "Wheel straight should be five-high."
    );

    std::cout << "[pass] test_straight_category_wheel\n";
}

void test_flush_category() {
    const poker::HandEvaluator evaluator;

    const poker::Board b = board(
        c(poker::Rank::Two, poker::Suit::Spades),
        c(poker::Rank::Seven, poker::Suit::Spades),
        c(poker::Rank::Nine, poker::Suit::Spades),
        c(poker::Rank::Jack, poker::Suit::Spades),
        c(poker::Rank::King, poker::Suit::Clubs)
    );

    const poker::HoleCards h = hand(
        c(poker::Rank::Ace, poker::Suit::Spades),
        c(poker::Rank::Three, poker::Suit::Hearts)
    );

    const poker::HandStrength strength = evaluator.evaluate_7(h, b);

    check_category(
        strength,
        poker::HandCategory::Flush,
        "Expected flush category."
    );

    check_eq(
        strength.ranks[0],
        static_cast<int>(poker::Rank::Ace),
        "Flush should be ace-high."
    );

    std::cout << "[pass] test_flush_category\n";
}

void test_full_house_category() {
    const poker::HandEvaluator evaluator;

    const poker::Board b = board(
        c(poker::Rank::Ace, poker::Suit::Clubs),
        c(poker::Rank::Ace, poker::Suit::Diamonds),
        c(poker::Rank::King, poker::Suit::Hearts),
        c(poker::Rank::King, poker::Suit::Spades),
        c(poker::Rank::Two, poker::Suit::Clubs)
    );

    const poker::HoleCards h = hand(
        c(poker::Rank::Ace, poker::Suit::Spades),
        c(poker::Rank::Three, poker::Suit::Hearts)
    );

    const poker::HandStrength strength = evaluator.evaluate_7(h, b);

    check_category(
        strength,
        poker::HandCategory::FullHouse,
        "Expected full house category."
    );

    check_eq(
        strength.ranks[0],
        static_cast<int>(poker::Rank::Ace),
        "Full house trips should be aces."
    );

    check_eq(
        strength.ranks[1],
        static_cast<int>(poker::Rank::King),
        "Full house pair should be kings."
    );

    std::cout << "[pass] test_full_house_category\n";
}

void test_four_of_a_kind_category() {
    const poker::HandEvaluator evaluator;

    const poker::Board b = board(
        c(poker::Rank::Ace, poker::Suit::Clubs),
        c(poker::Rank::Ace, poker::Suit::Diamonds),
        c(poker::Rank::Ace, poker::Suit::Hearts),
        c(poker::Rank::King, poker::Suit::Spades),
        c(poker::Rank::Two, poker::Suit::Clubs)
    );

    const poker::HoleCards h = hand(
        c(poker::Rank::Ace, poker::Suit::Spades),
        c(poker::Rank::Three, poker::Suit::Hearts)
    );

    const poker::HandStrength strength = evaluator.evaluate_7(h, b);

    check_category(
        strength,
        poker::HandCategory::FourOfAKind,
        "Expected four-of-a-kind category."
    );

    check_eq(
        strength.ranks[0],
        static_cast<int>(poker::Rank::Ace),
        "Quads should be aces."
    );

    check_eq(
        strength.ranks[1],
        static_cast<int>(poker::Rank::King),
        "Quad kicker should be king."
    );

    std::cout << "[pass] test_four_of_a_kind_category\n";
}

void test_straight_flush_category() {
    const poker::HandEvaluator evaluator;

    const poker::Board b = board(
        c(poker::Rank::Nine, poker::Suit::Spades),
        c(poker::Rank::Ten, poker::Suit::Spades),
        c(poker::Rank::Jack, poker::Suit::Spades),
        c(poker::Rank::Queen, poker::Suit::Spades),
        c(poker::Rank::Two, poker::Suit::Clubs)
    );

    const poker::HoleCards h = hand(
        c(poker::Rank::King, poker::Suit::Spades),
        c(poker::Rank::Three, poker::Suit::Hearts)
    );

    const poker::HandStrength strength = evaluator.evaluate_7(h, b);

    check_category(
        strength,
        poker::HandCategory::StraightFlush,
        "Expected straight flush category."
    );

    check_eq(
        strength.ranks[0],
        static_cast<int>(poker::Rank::King),
        "Straight flush should be king-high."
    );

    std::cout << "[pass] test_straight_flush_category\n";
}

void test_category_ordering() {
    const poker::HandEvaluator evaluator;

    const poker::Board common_board = board(
        c(poker::Rank::Two, poker::Suit::Clubs),
        c(poker::Rank::Seven, poker::Suit::Diamonds),
        c(poker::Rank::Nine, poker::Suit::Hearts),
        c(poker::Rank::Jack, poker::Suit::Spades),
        c(poker::Rank::King, poker::Suit::Clubs)
    );

    const poker::HoleCards pair_aces = hand(
        c(poker::Rank::Ace, poker::Suit::Diamonds),
        c(poker::Rank::Ace, poker::Suit::Hearts)
    );

    const poker::HoleCards high_card = hand(
        c(poker::Rank::Queen, poker::Suit::Diamonds),
        c(poker::Rank::Three, poker::Suit::Hearts)
    );

    check_compare_positive(
        evaluator.compare_7(pair_aces, high_card, common_board),
        "One pair should beat high card."
    );

    check_compare_negative(
        evaluator.compare_7(high_card, pair_aces, common_board),
        "High card should lose to one pair."
    );

    std::cout << "[pass] test_category_ordering\n";
}

void test_pair_kicker_breaks_tie() {
    const poker::HandEvaluator evaluator;

    const poker::Board b = board(
        c(poker::Rank::Ace, poker::Suit::Clubs),
        c(poker::Rank::Seven, poker::Suit::Diamonds),
        c(poker::Rank::Nine, poker::Suit::Hearts),
        c(poker::Rank::Jack, poker::Suit::Spades),
        c(poker::Rank::Two, poker::Suit::Clubs)
    );

    const poker::HoleCards better = hand(
        c(poker::Rank::Ace, poker::Suit::Diamonds),
        c(poker::Rank::King, poker::Suit::Hearts)
    );

    const poker::HoleCards worse = hand(
        c(poker::Rank::Ace, poker::Suit::Spades),
        c(poker::Rank::Queen, poker::Suit::Hearts)
    );

    check_compare_positive(
        evaluator.compare_7(better, worse, b),
        "Same pair should be decided by kicker."
    );

    std::cout << "[pass] test_pair_kicker_breaks_tie\n";
}

void test_two_pair_kicker_breaks_tie() {
    const poker::HandEvaluator evaluator;

    const poker::Board b = board(
        c(poker::Rank::Ace, poker::Suit::Clubs),
        c(poker::Rank::Ace, poker::Suit::Diamonds),
        c(poker::Rank::King, poker::Suit::Hearts),
        c(poker::Rank::King, poker::Suit::Spades),
        c(poker::Rank::Two, poker::Suit::Clubs)
    );

    const poker::HoleCards better = hand(
        c(poker::Rank::Queen, poker::Suit::Diamonds),
        c(poker::Rank::Three, poker::Suit::Hearts)
    );

    const poker::HoleCards worse = hand(
        c(poker::Rank::Jack, poker::Suit::Diamonds),
        c(poker::Rank::Four, poker::Suit::Hearts)
    );

    check_compare_positive(
        evaluator.compare_7(better, worse, b),
        "Same two pair should be decided by kicker."
    );

    std::cout << "[pass] test_two_pair_kicker_breaks_tie\n";
}

void test_board_only_tie() {
    const poker::HandEvaluator evaluator;

    const poker::Board royal_board = board(
        c(poker::Rank::Ten, poker::Suit::Spades),
        c(poker::Rank::Jack, poker::Suit::Spades),
        c(poker::Rank::Queen, poker::Suit::Spades),
        c(poker::Rank::King, poker::Suit::Spades),
        c(poker::Rank::Ace, poker::Suit::Spades)
    );

    const poker::HoleCards p0 = hand(
        c(poker::Rank::Two, poker::Suit::Clubs),
        c(poker::Rank::Three, poker::Suit::Diamonds)
    );

    const poker::HoleCards p1 = hand(
        c(poker::Rank::Four, poker::Suit::Clubs),
        c(poker::Rank::Five, poker::Suit::Diamonds)
    );

    check_compare_zero(
        evaluator.compare_7(p0, p1, royal_board),
        "Both players should tie when board makes royal flush."
    );

    std::cout << "[pass] test_board_only_tie\n";
}

void test_best_five_of_seven_prefers_flush_over_straight() {
    const poker::HandEvaluator evaluator;

    const poker::Board b = board(
        c(poker::Rank::Two, poker::Suit::Spades),
        c(poker::Rank::Three, poker::Suit::Spades),
        c(poker::Rank::Four, poker::Suit::Spades),
        c(poker::Rank::Five, poker::Suit::Hearts),
        c(poker::Rank::King, poker::Suit::Spades)
    );

    const poker::HoleCards h = hand(
        c(poker::Rank::Ace, poker::Suit::Spades),
        c(poker::Rank::Six, poker::Suit::Diamonds)
    );

    const poker::HandStrength strength = evaluator.evaluate_7(h, b);

    check_category(
        strength,
        poker::HandCategory::Flush,
        "Evaluator should choose ace-high flush over six-high straight."
    );

    std::cout << "[pass] test_best_five_of_seven_prefers_flush_over_straight\n";
}

void test_best_full_house_uses_highest_trips_then_highest_pair() {
    const poker::HandEvaluator evaluator;

    const poker::Board b = board(
        c(poker::Rank::Ace, poker::Suit::Clubs),
        c(poker::Rank::Ace, poker::Suit::Diamonds),
        c(poker::Rank::King, poker::Suit::Hearts),
        c(poker::Rank::King, poker::Suit::Spades),
        c(poker::Rank::King, poker::Suit::Clubs)
    );

    const poker::HoleCards h = hand(
        c(poker::Rank::Ace, poker::Suit::Spades),
        c(poker::Rank::Two, poker::Suit::Hearts)
    );

    const poker::HandStrength strength = evaluator.evaluate_7(h, b);

    check_category(
        strength,
        poker::HandCategory::FullHouse,
        "Expected full house."
    );

    check_eq(
        strength.ranks[0],
        static_cast<int>(poker::Rank::Ace),
        "Best full house should use Aces as trips."
    );

    check_eq(
        strength.ranks[1],
        static_cast<int>(poker::Rank::King),
        "Best full house should use Kings as pair."
    );

    std::cout << "[pass] test_best_full_house_uses_highest_trips_then_highest_pair\n";
}

void test_duplicate_cards_throw() {
    const poker::HandEvaluator evaluator;

    const poker::CardId ace_spades =
        c(poker::Rank::Ace, poker::Suit::Spades);

    const poker::Board b = board(
        ace_spades,
        c(poker::Rank::Two, poker::Suit::Clubs),
        c(poker::Rank::Three, poker::Suit::Diamonds),
        c(poker::Rank::Four, poker::Suit::Hearts),
        c(poker::Rank::Five, poker::Suit::Clubs)
    );

    const poker::HoleCards h = hand(
        ace_spades,
        c(poker::Rank::King, poker::Suit::Hearts)
    );

    bool threw = false;

    try {
        (void)evaluator.evaluate_7(h, b);
    } catch (const std::exception&) {
        threw = true;
    }

    check(
        threw,
        "Evaluator should reject duplicate cards across hand and board."
    );

    std::cout << "[pass] test_duplicate_cards_throw\n";
}

void test_incomplete_board_throws_for_7_card_eval() {
    const poker::HandEvaluator evaluator;

    const poker::Board b{
        {
            c(poker::Rank::Two, poker::Suit::Clubs),
            c(poker::Rank::Three, poker::Suit::Diamonds),
            c(poker::Rank::Four, poker::Suit::Hearts)
        }
    };

    const poker::HoleCards h = hand(
        c(poker::Rank::Ace, poker::Suit::Spades),
        c(poker::Rank::King, poker::Suit::Hearts)
    );

    bool threw = false;

    try {
        (void)evaluator.evaluate_7(h, b);
    } catch (const std::exception&) {
        threw = true;
    }

    check(
        threw,
        "Seven-card evaluator should reject incomplete board."
    );

    std::cout << "[pass] test_incomplete_board_throws_for_7_card_eval\n";
}

void run_all_tests() {
    test_high_card_category();
    test_one_pair_category();
    test_two_pair_category();
    test_three_of_a_kind_category();
    test_straight_category_broadway();
    test_straight_category_wheel();
    test_flush_category();
    test_full_house_category();
    test_four_of_a_kind_category();
    test_straight_flush_category();
    test_category_ordering();
    test_pair_kicker_breaks_tie();
    test_two_pair_kicker_breaks_tie();
    test_board_only_tie();
    test_best_five_of_seven_prefers_flush_over_straight();
    test_best_full_house_uses_highest_trips_then_highest_pair();
    test_duplicate_cards_throw();
    test_incomplete_board_throws_for_7_card_eval();
}

} // namespace

int main() {
    try {
        run_all_tests();
    } catch (const std::exception& e) {
        std::cerr << "[fail] " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "[pass] all hand evaluator tests passed\n";
    return EXIT_SUCCESS;
}
