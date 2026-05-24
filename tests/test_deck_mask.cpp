#include "poker/card.hpp"
#include "poker/deck_mask.hpp"
#include "poker/hand.hpp"
#include "poker/board.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

bool contains(
    const std::vector<poker::CardId>& cards,
    poker::CardId card
) {
    return std::find(cards.begin(), cards.end(), card) != cards.end();
}

void test_card_encoding_round_trip() {
    for (int suit = 0; suit < 4; ++suit) {
        for (int rank = 2; rank <= 14; ++rank) {
            const poker::CardId card = poker::make_card(
                static_cast<poker::Rank>(rank),
                static_cast<poker::Suit>(suit)
            );

            check_eq(
                static_cast<int>(poker::rank_of(card)),
                rank,
                "Card rank should round-trip."
            );

            check_eq(
                static_cast<int>(poker::suit_of(card)),
                suit,
                "Card suit should round-trip."
            );

            check(
                card < 52,
                "CardId should be in range 0..51."
            );
        }
    }

    std::cout << "[pass] test_card_encoding_round_trip\n";
}

void test_empty_mask_contains_no_cards() {
    const poker::DeckMask mask = poker::empty_deck_mask();

    check_eq(
        poker::popcount(mask),
        0,
        "Empty deck mask should have zero cards."
    );

    for (int c = 0; c < 52; ++c) {
        check(
            !poker::contains_card(mask, static_cast<poker::CardId>(c)),
            "Empty deck mask should not contain any card."
        );
    }

    std::cout << "[pass] test_empty_mask_contains_no_cards\n";
}

void test_full_mask_contains_all_cards() {
    const poker::DeckMask mask = poker::full_deck_mask();

    check_eq(
        poker::popcount(mask),
        52,
        "Full deck mask should contain 52 cards."
    );

    for (int c = 0; c < 52; ++c) {
        check(
            poker::contains_card(mask, static_cast<poker::CardId>(c)),
            "Full deck mask should contain every card."
        );
    }

    std::cout << "[pass] test_full_mask_contains_all_cards\n";
}

void test_card_mask_has_one_bit() {
    const poker::CardId ace_spades = poker::make_card(
        poker::Rank::Ace,
        poker::Suit::Spades
    );

    const poker::DeckMask mask = poker::card_mask(ace_spades);

    check_eq(
        poker::popcount(mask),
        1,
        "Single-card mask should contain exactly one card."
    );

    check(
        poker::contains_card(mask, ace_spades),
        "Single-card mask should contain the selected card."
    );

    std::cout << "[pass] test_card_mask_has_one_bit\n";
}

void test_add_and_remove_card() {
    const poker::CardId king_hearts = poker::make_card(
        poker::Rank::King,
        poker::Suit::Hearts
    );

    poker::DeckMask mask = poker::empty_deck_mask();

    mask = poker::add_card(mask, king_hearts);

    check_eq(
        poker::popcount(mask),
        1,
        "Adding one card should increase popcount to one."
    );

    check(
        poker::contains_card(mask, king_hearts),
        "Mask should contain added card."
    );

    mask = poker::remove_card(mask, king_hearts);

    check_eq(
        poker::popcount(mask),
        0,
        "Removing the card should return mask to empty."
    );

    check(
        !poker::contains_card(mask, king_hearts),
        "Mask should not contain removed card."
    );

    std::cout << "[pass] test_add_and_remove_card\n";
}

void test_adding_same_card_is_idempotent() {
    const poker::CardId queen_clubs = poker::make_card(
        poker::Rank::Queen,
        poker::Suit::Clubs
    );

    poker::DeckMask mask = poker::empty_deck_mask();

    mask = poker::add_card(mask, queen_clubs);
    mask = poker::add_card(mask, queen_clubs);

    check_eq(
        poker::popcount(mask),
        1,
        "Adding the same card twice should still leave one card."
    );

    std::cout << "[pass] test_adding_same_card_is_idempotent\n";
}

void test_removing_absent_card_is_idempotent() {
    const poker::CardId jack_diamonds = poker::make_card(
        poker::Rank::Jack,
        poker::Suit::Diamonds
    );

    poker::DeckMask mask = poker::empty_deck_mask();

    mask = poker::remove_card(mask, jack_diamonds);

    check_eq(
        poker::popcount(mask),
        0,
        "Removing an absent card should keep mask empty."
    );

    std::cout << "[pass] test_removing_absent_card_is_idempotent\n";
}

void test_hand_mask_contains_two_cards() {
    const poker::CardId ace_spades = poker::make_card(
        poker::Rank::Ace,
        poker::Suit::Spades
    );

    const poker::CardId ace_hearts = poker::make_card(
        poker::Rank::Ace,
        poker::Suit::Hearts
    );

    const poker::HoleCards hand{
        ace_spades,
        ace_hearts
    };

    const poker::DeckMask mask = poker::hand_mask(hand);

    check_eq(
        poker::popcount(mask),
        2,
        "Hole-card mask should contain two cards."
    );

    check(
        poker::contains_card(mask, ace_spades),
        "Hole-card mask should contain first card."
    );

    check(
        poker::contains_card(mask, ace_hearts),
        "Hole-card mask should contain second card."
    );

    std::cout << "[pass] test_hand_mask_contains_two_cards\n";
}

void test_board_mask_contains_board_cards() {
    const poker::Board board{
        {
            poker::make_card(poker::Rank::Ace, poker::Suit::Spades),
            poker::make_card(poker::Rank::King, poker::Suit::Spades),
            poker::make_card(poker::Rank::Queen, poker::Suit::Spades),
            poker::make_card(poker::Rank::Jack, poker::Suit::Spades),
            poker::make_card(poker::Rank::Ten, poker::Suit::Spades)
        }
    };

    const poker::DeckMask mask = poker::board_mask(board);

    check_eq(
        poker::popcount(mask),
        5,
        "Five-card board mask should contain five cards."
    );

    for (poker::CardId card : board.cards) {
        check(
            poker::contains_card(mask, card),
            "Board mask should contain every board card."
        );
    }

    std::cout << "[pass] test_board_mask_contains_board_cards\n";
}

void test_masks_overlap_when_sharing_card() {
    const poker::CardId ace_spades = poker::make_card(
        poker::Rank::Ace,
        poker::Suit::Spades
    );

    const poker::CardId king_spades = poker::make_card(
        poker::Rank::King,
        poker::Suit::Spades
    );

    const poker::DeckMask a =
        poker::add_card(poker::empty_deck_mask(), ace_spades);

    poker::DeckMask b = poker::empty_deck_mask();
    b = poker::add_card(b, ace_spades);
    b = poker::add_card(b, king_spades);

    check(
        poker::masks_overlap(a, b),
        "Masks sharing a card should overlap."
    );

    std::cout << "[pass] test_masks_overlap_when_sharing_card\n";
}

void test_masks_do_not_overlap_when_disjoint() {
    const poker::DeckMask a = poker::hand_mask(
        poker::HoleCards{
            poker::make_card(poker::Rank::Ace, poker::Suit::Spades),
            poker::make_card(poker::Rank::Ace, poker::Suit::Hearts)
        }
    );

    const poker::DeckMask b = poker::hand_mask(
        poker::HoleCards{
            poker::make_card(poker::Rank::King, poker::Suit::Clubs),
            poker::make_card(poker::Rank::King, poker::Suit::Diamonds)
        }
    );

    check(
        !poker::masks_overlap(a, b),
        "Disjoint masks should not overlap."
    );

    std::cout << "[pass] test_masks_do_not_overlap_when_disjoint\n";
}

void test_cards_in_mask_returns_exact_cards() {
    const poker::CardId ace_spades = poker::make_card(
        poker::Rank::Ace,
        poker::Suit::Spades
    );

    const poker::CardId two_clubs = poker::make_card(
        poker::Rank::Two,
        poker::Suit::Clubs
    );

    poker::DeckMask mask = poker::empty_deck_mask();
    mask = poker::add_card(mask, ace_spades);
    mask = poker::add_card(mask, two_clubs);

    const std::vector<poker::CardId> cards =
        poker::cards_in_mask(mask);

    check_eq(
        static_cast<int>(cards.size()),
        2,
        "cards_in_mask should return exactly two cards."
    );

    check(
        contains(cards, ace_spades),
        "cards_in_mask should include ace of spades."
    );

    check(
        contains(cards, two_clubs),
        "cards_in_mask should include two of clubs."
    );

    std::cout << "[pass] test_cards_in_mask_returns_exact_cards\n";
}

void test_remaining_cards_excludes_dead_cards() {
    const poker::CardId ace_spades = poker::make_card(
        poker::Rank::Ace,
        poker::Suit::Spades
    );

    const poker::CardId king_hearts = poker::make_card(
        poker::Rank::King,
        poker::Suit::Hearts
    );

    poker::DeckMask dead = poker::empty_deck_mask();
    dead = poker::add_card(dead, ace_spades);
    dead = poker::add_card(dead, king_hearts);

    const std::vector<poker::CardId> remaining =
        poker::remaining_cards(dead);

    check_eq(
        static_cast<int>(remaining.size()),
        50,
        "Removing two dead cards should leave 50 cards."
    );

    check(
        !contains(remaining, ace_spades),
        "Remaining cards should not include first dead card."
    );

    check(
        !contains(remaining, king_hearts),
        "Remaining cards should not include second dead card."
    );

    for (poker::CardId card : remaining) {
        check(
            !poker::contains_card(dead, card),
            "Remaining cards should not contain any dead card."
        );
    }

    std::cout << "[pass] test_remaining_cards_excludes_dead_cards\n";
}

void test_full_mask_minus_board_and_hand() {
    const poker::HoleCards hand{
        poker::make_card(poker::Rank::Ace, poker::Suit::Spades),
        poker::make_card(poker::Rank::Ace, poker::Suit::Hearts)
    };

    const poker::Board board{
        {
            poker::make_card(poker::Rank::Two, poker::Suit::Clubs),
            poker::make_card(poker::Rank::Seven, poker::Suit::Diamonds),
            poker::make_card(poker::Rank::King, poker::Suit::Spades),
            poker::make_card(poker::Rank::Four, poker::Suit::Hearts),
            poker::make_card(poker::Rank::Nine, poker::Suit::Clubs)
        }
    };

    const poker::DeckMask dead =
        poker::hand_mask(hand) | poker::board_mask(board);

    const poker::DeckMask available =
        poker::full_deck_mask() & ~dead;

    check_eq(
        poker::popcount(dead),
        7,
        "Two-card hand plus five-card board should have seven dead cards."
    );

    check_eq(
        poker::popcount(available),
        45,
        "Full deck minus seven dead cards should leave 45 available cards."
    );

    for (poker::CardId card : hand.cards()) {
        check(
            !poker::contains_card(available, card),
            "Available cards should exclude hole cards."
        );
    }

    for (poker::CardId card : board.cards) {
        check(
            !poker::contains_card(available, card),
            "Available cards should exclude board cards."
        );
    }

    std::cout << "[pass] test_full_mask_minus_board_and_hand\n";
}

void test_duplicate_hand_cards_throw_or_fail_validation() {
    const poker::CardId ace_spades = poker::make_card(
        poker::Rank::Ace,
        poker::Suit::Spades
    );

    bool threw = false;

    try {
        const poker::HoleCards hand{
            ace_spades,
            ace_spades
        };

        (void)poker::validate_hand(hand);
    } catch (const std::exception&) {
        threw = true;
    }

    check(
        threw,
        "Duplicate hole cards should fail validation."
    );

    std::cout << "[pass] test_duplicate_hand_cards_throw_or_fail_validation\n";
}

void test_duplicate_board_cards_throw_or_fail_validation() {
    const poker::CardId ace_spades = poker::make_card(
        poker::Rank::Ace,
        poker::Suit::Spades
    );

    bool threw = false;

    try {
        const poker::Board board{
            {
                ace_spades,
                ace_spades,
                poker::make_card(poker::Rank::King, poker::Suit::Spades)
            }
        };

        (void)poker::validate_board(board);
    } catch (const std::exception&) {
        threw = true;
    }

    check(
        threw,
        "Duplicate board cards should fail validation."
    );

    std::cout << "[pass] test_duplicate_board_cards_throw_or_fail_validation\n";
}

void run_all_tests() {
    test_card_encoding_round_trip();
    test_empty_mask_contains_no_cards();
    test_full_mask_contains_all_cards();
    test_card_mask_has_one_bit();
    test_add_and_remove_card();
    test_adding_same_card_is_idempotent();
    test_removing_absent_card_is_idempotent();
    test_hand_mask_contains_two_cards();
    test_board_mask_contains_board_cards();
    test_masks_overlap_when_sharing_card();
    test_masks_do_not_overlap_when_disjoint();
    test_cards_in_mask_returns_exact_cards();
    test_remaining_cards_excludes_dead_cards();
    test_full_mask_minus_board_and_hand();
    test_duplicate_hand_cards_throw_or_fail_validation();
    test_duplicate_board_cards_throw_or_fail_validation();
}

} // namespace

int main() {
    try {
        run_all_tests();
    } catch (const std::exception& e) {
        std::cerr << "[fail] " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "[pass] all deck mask tests passed\n";
    return EXIT_SUCCESS;
}