#include "poker/board.hpp"
#include "poker/card.hpp"
#include "poker/deck_mask.hpp"
#include "poker/hand.hpp"
#include "poker/range.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kTol = 1e-6;

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

poker::HandId h(
    poker::Rank r1,
    poker::Suit s1,
    poker::Rank r2,
    poker::Suit s2
) {
    return poker::make_hand(c(r1, s1), c(r2, s2));
}

poker::Board make_test_board() {
    return poker::Board{
        {
            c(poker::Rank::Ace, poker::Suit::Spades),
            c(poker::Rank::Seven, poker::Suit::Hearts),
            c(poker::Rank::Two, poker::Suit::Clubs),
            c(poker::Rank::Jack, poker::Suit::Diamonds),
            c(poker::Rank::Four, poker::Suit::Spades)
        }
    };
}

bool contains_hand(
    const std::vector<poker::HandId>& hands,
    poker::HandId hand
) {
    return std::find(hands.begin(), hands.end(), hand) != hands.end();
}

poker::Range make_range(std::initializer_list<std::pair<poker::HandId, float>> entries) {
    poker::Range range;
    range.clear();

    for (const auto& [hand, weight] : entries) {
        range.set_weight(hand, weight);
    }

    return range;
}

double probability_sum(const std::vector<poker::LegalHandPair>& pairs) {
    double sum = 0.0;

    for (const poker::LegalHandPair& pair : pairs) {
        sum += static_cast<double>(pair.probability);
    }

    return sum;
}

bool contains_pair(
    const std::vector<poker::LegalHandPair>& pairs,
    poker::HandId p0_hand,
    poker::HandId p1_hand
) {
    return std::any_of(
        pairs.begin(),
        pairs.end(),
        [p0_hand, p1_hand](const poker::LegalHandPair& pair) {
            return pair.p0_hand == p0_hand && pair.p1_hand == p1_hand;
        }
    );
}

float pair_probability(
    const std::vector<poker::LegalHandPair>& pairs,
    poker::HandId p0_hand,
    poker::HandId p1_hand
) {
    for (const poker::LegalHandPair& pair : pairs) {
        if (pair.p0_hand == p0_hand && pair.p1_hand == p1_hand) {
            return pair.probability;
        }
    }

    std::ostringstream oss;
    oss << "Could not find requested legal hand pair.";
    throw std::runtime_error(oss.str());
}

void test_empty_range_has_zero_weight() {
    poker::Range range;
    range.clear();

    check_eq(
        range.nonzero_count(),
        0,
        "Empty range should have zero nonzero combos."
    );

    check_near(
        range.total_weight(),
        0.0,
        kTol,
        "Empty range should have zero total weight."
    );

    std::cout << "[pass] test_empty_range_has_zero_weight\n";
}

void test_set_and_get_combo_weight() {
    poker::Range range;
    range.clear();

    const poker::HandId ak_hearts = h(
        poker::Rank::Ace,
        poker::Suit::Hearts,
        poker::Rank::King,
        poker::Suit::Hearts
    );

    range.set_weight(ak_hearts, 0.75f);

    check_near(
        range.weight(ak_hearts),
        0.75,
        kTol,
        "Range should return stored combo weight."
    );

    check_eq(
        range.nonzero_count(),
        1,
        "Range should have exactly one nonzero combo."
    );

    std::cout << "[pass] test_set_and_get_combo_weight\n";
}

void test_zero_weight_removes_combo_from_positive_list() {
    const poker::HandId aa = h(
        poker::Rank::Ace,
        poker::Suit::Hearts,
        poker::Rank::Ace,
        poker::Suit::Diamonds
    );

    poker::Range range;
    range.clear();

    range.set_weight(aa, 1.0f);
    range.set_weight(aa, 0.0f);

    check_eq(
        range.nonzero_count(),
        0,
        "Setting combo weight to zero should remove it from nonzero count."
    );

    check(
        !contains_hand(range.hands_with_positive_weight(), aa),
        "Zero-weight combo should not appear in positive list."
    );

    std::cout << "[pass] test_zero_weight_removes_combo_from_positive_list\n";
}

void test_remove_blocked_removes_board_card_combos() {
    const poker::Board board = make_test_board();
    const poker::DeckMask dead = poker::board_mask(board);

    const poker::HandId blocked_by_ace_spades = h(
        poker::Rank::Ace,
        poker::Suit::Spades,
        poker::Rank::King,
        poker::Suit::Hearts
    );

    const poker::HandId blocked_by_jack_diamonds = h(
        poker::Rank::Jack,
        poker::Suit::Diamonds,
        poker::Rank::Jack,
        poker::Suit::Clubs
    );

    const poker::HandId unblocked = h(
        poker::Rank::King,
        poker::Suit::Hearts,
        poker::Rank::Queen,
        poker::Suit::Hearts
    );

    const poker::Range range = make_range({
        {blocked_by_ace_spades, 1.0f},
        {blocked_by_jack_diamonds, 1.0f},
        {unblocked, 1.0f}
    });

    const poker::Range filtered = range.remove_blocked(dead);

    check_near(
        filtered.weight(blocked_by_ace_spades),
        0.0,
        kTol,
        "Combo containing board ace of spades should be removed."
    );

    check_near(
        filtered.weight(blocked_by_jack_diamonds),
        0.0,
        kTol,
        "Combo containing board jack of diamonds should be removed."
    );

    check_near(
        filtered.weight(unblocked),
        1.0,
        kTol,
        "Unblocked combo should remain."
    );

    check_eq(
        filtered.nonzero_count(),
        1,
        "Only one combo should remain after board card removal."
    );

    std::cout << "[pass] test_remove_blocked_removes_board_card_combos\n";
}

void test_remove_blocked_preserves_weights_for_unblocked_combos() {
    const poker::Board board = make_test_board();
    const poker::DeckMask dead = poker::board_mask(board);

    const poker::HandId kq_hearts = h(
        poker::Rank::King,
        poker::Suit::Hearts,
        poker::Rank::Queen,
        poker::Suit::Hearts
    );

    const poker::HandId tt = h(
        poker::Rank::Ten,
        poker::Suit::Clubs,
        poker::Rank::Ten,
        poker::Suit::Diamonds
    );

    const poker::Range range = make_range({
        {kq_hearts, 0.25f},
        {tt, 0.80f}
    });

    const poker::Range filtered = range.remove_blocked(dead);

    check_near(
        filtered.weight(kq_hearts),
        0.25,
        kTol,
        "Unblocked KQ hearts weight should be preserved."
    );

    check_near(
        filtered.weight(tt),
        0.80,
        kTol,
        "Unblocked TT weight should be preserved."
    );

    check_near(
        filtered.total_weight(),
        1.05,
        kTol,
        "Filtered range total should preserve unblocked weights."
    );

    std::cout << "[pass] test_remove_blocked_preserves_weights_for_unblocked_combos\n";
}

void test_normalized_range_sums_to_one() {
    const poker::HandId aa = h(
        poker::Rank::Ace,
        poker::Suit::Hearts,
        poker::Rank::Ace,
        poker::Suit::Diamonds
    );

    const poker::HandId kk = h(
        poker::Rank::King,
        poker::Suit::Hearts,
        poker::Rank::King,
        poker::Suit::Diamonds
    );

    const poker::Range range = make_range({
        {aa, 2.0f},
        {kk, 1.0f}
    });

    const poker::Range normalized = range.normalized();

    check_near(
        normalized.total_weight(),
        1.0,
        kTol,
        "Normalized range should sum to one."
    );

    check_near(
        normalized.weight(aa),
        2.0 / 3.0,
        kTol,
        "AA normalized weight mismatch."
    );

    check_near(
        normalized.weight(kk),
        1.0 / 3.0,
        kTol,
        "KK normalized weight mismatch."
    );

    std::cout << "[pass] test_normalized_range_sums_to_one\n";
}

void test_normalizing_empty_range_throws() {
    poker::Range range;
    range.clear();

    bool threw = false;

    try {
        (void)range.normalized();
    } catch (const std::exception&) {
        threw = true;
    }

    check(
        threw,
        "Normalizing empty range should throw."
    );

    std::cout << "[pass] test_normalizing_empty_range_throws\n";
}

void test_legal_hand_pairs_exclude_board_blockers() {
    const poker::Board board = make_test_board();
    const poker::DeckMask board_dead = poker::board_mask(board);

    const poker::HandId p0_blocked = h(
        poker::Rank::Ace,
        poker::Suit::Spades,
        poker::Rank::King,
        poker::Suit::Hearts
    );

    const poker::HandId p0_legal = h(
        poker::Rank::King,
        poker::Suit::Hearts,
        poker::Rank::Queen,
        poker::Suit::Hearts
    );

    const poker::HandId p1_legal = h(
        poker::Rank::Nine,
        poker::Suit::Clubs,
        poker::Rank::Nine,
        poker::Suit::Diamonds
    );

    const poker::Range p0_range = make_range({
        {p0_blocked, 1.0f},
        {p0_legal, 1.0f}
    });

    const poker::Range p1_range = make_range({
        {p1_legal, 1.0f}
    });

    const std::vector<poker::LegalHandPair> pairs = poker::legal_hand_pairs(p0_range, p1_range, board_dead);

    check_eq(
        static_cast<int>(pairs.size()),
        1,
        "Only one legal pair should remain after board blockers."
    );

    check(
        contains_pair(pairs, p0_legal, p1_legal),
        "Legal unblocked pair should be present."
    );

    check(
        !contains_pair(pairs, p0_blocked, p1_legal),
        "Pair containing board-blocked P0 hand should be absent."
    );

    std::cout << "[pass] test_legal_hand_pairs_exclude_board_blockers\n";
}

void test_legal_hand_pairs_exclude_shared_private_cards() {
    const poker::Board board = make_test_board();
    const poker::DeckMask board_dead = poker::board_mask(board);

    const poker::HandId p0_ak_hearts = h(
        poker::Rank::Ace,
        poker::Suit::Hearts,
        poker::Rank::King,
        poker::Suit::Hearts
    );

    const poker::HandId p1_shares_ace_hearts = h(
        poker::Rank::Ace,
        poker::Suit::Hearts,
        poker::Rank::Queen,
        poker::Suit::Diamonds
    );

    const poker::HandId p1_disjoint = h(
        poker::Rank::Nine,
        poker::Suit::Clubs,
        poker::Rank::Nine,
        poker::Suit::Diamonds
    );

    const poker::Range p0_range = make_range({
        {p0_ak_hearts, 1.0f}
    });

    const poker::Range p1_range = make_range({
        {p1_shares_ace_hearts, 1.0f},
        {p1_disjoint, 1.0f}
    });

    const std::vector<poker::LegalHandPair> pairs = poker::legal_hand_pairs(p0_range, p1_range, board_dead);

    check_eq(
        static_cast<int>(pairs.size()),
        1,
        "Only one pair should remain after private-card collision removal."
    );

    check(
        !contains_pair(pairs, p0_ak_hearts, p1_shares_ace_hearts),
        "Hands sharing ace of hearts should not form a legal pair."
    );

    check(
        contains_pair(pairs, p0_ak_hearts, p1_disjoint),
        "Disjoint private hands should form a legal pair."
    );

    std::cout << "[pass] test_legal_hand_pairs_exclude_shared_private_cards\n";
}

void test_legal_hand_pair_probabilities_normalize_to_one() {
    const poker::Board board = make_test_board();
    const poker::DeckMask board_dead = poker::board_mask(board);

    const poker::HandId p0_a = h(
        poker::Rank::King,
        poker::Suit::Hearts,
        poker::Rank::Queen,
        poker::Suit::Hearts
    );

    const poker::HandId p0_b = h(
        poker::Rank::Ten,
        poker::Suit::Clubs,
        poker::Rank::Ten,
        poker::Suit::Diamonds
    );

    const poker::HandId p1_a = h(
        poker::Rank::Nine,
        poker::Suit::Clubs,
        poker::Rank::Nine,
        poker::Suit::Diamonds
    );

    const poker::HandId p1_b = h(
        poker::Rank::Eight,
        poker::Suit::Clubs,
        poker::Rank::Eight,
        poker::Suit::Diamonds
    );

    const poker::Range p0_range = make_range({
        {p0_a, 2.0f},
        {p0_b, 1.0f}
    });

    const poker::Range p1_range = make_range({
        {p1_a, 3.0f},
        {p1_b, 1.0f}
    });

    const std::vector<poker::LegalHandPair> pairs =
        poker::legal_hand_pairs(p0_range, p1_range, board_dead);

    check_near(
        probability_sum(pairs),
        1.0,
        kTol,
        "Legal hand-pair probabilities should sum to one."
    );

    std::cout << "[pass] test_legal_hand_pair_probabilities_normalize_to_one\n";
}

void test_legal_hand_pair_probabilities_follow_product_weights() {
    const poker::Board board = make_test_board();
    const poker::DeckMask board_dead = poker::board_mask(board);

    const poker::HandId p0_a = h(
        poker::Rank::King,
        poker::Suit::Hearts,
        poker::Rank::Queen,
        poker::Suit::Hearts
    );

    const poker::HandId p0_b = h(
        poker::Rank::Ten,
        poker::Suit::Clubs,
        poker::Rank::Ten,
        poker::Suit::Diamonds
    );

    const poker::HandId p1_a = h(
        poker::Rank::Nine,
        poker::Suit::Clubs,
        poker::Rank::Nine,
        poker::Suit::Diamonds
    );

    const poker::HandId p1_b = h(
        poker::Rank::Eight,
        poker::Suit::Clubs,
        poker::Rank::Eight,
        poker::Suit::Diamonds
    );

    const poker::Range p0_range = make_range({
        {p0_a, 2.0f},
        {p0_b, 1.0f}
    });

    const poker::Range p1_range = make_range({
        {p1_a, 3.0f},
        {p1_b, 1.0f}
    });

    const std::vector<poker::LegalHandPair> pairs =
        poker::legal_hand_pairs(p0_range, p1_range, board_dead);

    // Product weights:
    //   p0_a/p1_a = 2 * 3 = 6
    //   p0_a/p1_b = 2 * 1 = 2
    //   p0_b/p1_a = 1 * 3 = 3
    //   p0_b/p1_b = 1 * 1 = 1
    // total = 12
    check_near(
        pair_probability(pairs, p0_a, p1_a),
        6.0 / 12.0,
        kTol,
        "p0_a/p1_a pair probability mismatch."
    );

    check_near(
        pair_probability(pairs, p0_a, p1_b),
        2.0 / 12.0,
        kTol,
        "p0_a/p1_b pair probability mismatch."
    );

    check_near(
        pair_probability(pairs, p0_b, p1_a),
        3.0 / 12.0,
        kTol,
        "p0_b/p1_a pair probability mismatch."
    );

    check_near(
        pair_probability(pairs, p0_b, p1_b),
        1.0 / 12.0,
        kTol,
        "p0_b/p1_b pair probability mismatch."
    );

    std::cout << "[pass] test_legal_hand_pair_probabilities_follow_product_weights\n";
}

void test_pair_enumeration_throws_when_no_legal_pairs_remain() {
    const poker::Board board = make_test_board();
    const poker::DeckMask board_dead = poker::board_mask(board);

    const poker::HandId p0_hand = h(
        poker::Rank::Ace,
        poker::Suit::Hearts,
        poker::Rank::King,
        poker::Suit::Hearts
    );

    const poker::HandId p1_shares_ace_hearts = h(
        poker::Rank::Ace,
        poker::Suit::Hearts,
        poker::Rank::Queen,
        poker::Suit::Diamonds
    );

    const poker::Range p0_range = make_range({
        {p0_hand, 1.0f}
    });

    const poker::Range p1_range = make_range({
        {p1_shares_ace_hearts, 1.0f}
    });

    bool threw = false;

    try {
        (void)poker::legal_hand_pairs(
            p0_range,
            p1_range,
            board_dead
        );
    } catch (const std::exception&) {
        threw = true;
    }

    check(
        threw,
        "Pair enumeration should throw when no legal hand pairs remain."
    );

    std::cout << "[pass] test_pair_enumeration_throws_when_no_legal_pairs_remain\n";
}

void test_negative_weight_throws() {
    poker::Range range;
    range.clear();

    bool threw = false;

    try {
        range.set_weight(
            h(
                poker::Rank::Ace,
                poker::Suit::Hearts,
                poker::Rank::King,
                poker::Suit::Hearts
            ),
            -0.25f
        );
    } catch (const std::exception&) {
        threw = true;
    }

    check(
        threw,
        "Setting a negative range weight should throw."
    );

    std::cout << "[pass] test_negative_weight_throws\n";
}

void test_duplicate_cards_in_hand_id_throw_or_are_rejected() {
    bool threw = false;

    try {
        const poker::CardId ace_hearts =
            c(poker::Rank::Ace, poker::Suit::Hearts);

        (void)poker::make_hand(ace_hearts, ace_hearts);
    } catch (const std::exception&) {
        threw = true;
    }

    check(
        threw,
        "make_hand should reject duplicate cards."
    );

    std::cout << "[pass] test_duplicate_cards_in_hand_id_throw_or_are_rejected\n";
}

void run_all_tests() {
    test_empty_range_has_zero_weight();
    test_set_and_get_combo_weight();
    test_zero_weight_removes_combo_from_positive_list();

    test_remove_blocked_removes_board_card_combos();
    test_remove_blocked_preserves_weights_for_unblocked_combos();
    test_normalized_range_sums_to_one();
    test_normalizing_empty_range_throws();

    test_legal_hand_pairs_exclude_board_blockers();
    test_legal_hand_pairs_exclude_shared_private_cards();
    test_legal_hand_pair_probabilities_normalize_to_one();
    test_legal_hand_pair_probabilities_follow_product_weights();
    test_pair_enumeration_throws_when_no_legal_pairs_remain();

    test_negative_weight_throws();
    test_duplicate_cards_in_hand_id_throw_or_are_rejected();
}

} // namespace

int main() {
    try {
        run_all_tests();
    } catch (const std::exception& e) {
        std::cerr << "[fail] " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "[pass] all range card-removal tests passed\n";
    return EXIT_SUCCESS;
}