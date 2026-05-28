#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace poker {

using CardId = std::uint8_t;

enum class Suit : int {
    Clubs    = 0,
    Diamonds = 1,
    Hearts   = 2,
    Spades   = 3
};

enum class Rank : int {
    Two   = 2,
    Three = 3,
    Four  = 4,
    Five  = 5,
    Six   = 6,
    Seven = 7,
    Eight = 8,
    Nine  = 9,
    Ten   = 10,
    Jack  = 11,
    Queen = 12,
    King  = 13,
    Ace   = 14
};

inline bool is_valid_suit(Suit suit) {
    const int s = static_cast<int>(suit);
    return s >= 0 && s < 4;
}

inline bool is_valid_rank(Rank rank) {
    const int r = static_cast<int>(rank);
    return r >= 2 && r <= 14;
}

inline CardId make_card(Rank rank, Suit suit) {
    if (!is_valid_rank(rank)) {
        throw std::invalid_argument("Invalid card rank.");
    }

    if (!is_valid_suit(suit)) {
        throw std::invalid_argument("Invalid card suit.");
    }

    const int rank_index = static_cast<int>(rank) - 2; // 0..12
    const int suit_index = static_cast<int>(suit);     // 0..3

    return static_cast<CardId>(suit_index * 13 + rank_index);
}

inline void validate_card(CardId card) {
    if (card >= 52) {
        throw std::invalid_argument("CardId must be in range 0..51.");
    }
}

inline Rank rank_of(CardId card) {
    validate_card(card);

    const int rank_index = static_cast<int>(card) % 13;
    return static_cast<Rank>(rank_index + 2);
}

inline Suit suit_of(CardId card) {
    validate_card(card);

    const int suit_index = static_cast<int>(card) / 13;
    return static_cast<Suit>(suit_index);
}

inline int rank_value(CardId card) {
    return static_cast<int>(rank_of(card));
}

inline int suit_value(CardId card) {
    return static_cast<int>(suit_of(card));
}

inline std::string to_string(Suit suit) {
    switch (suit) {
        case Suit::Clubs:    return "c";
        case Suit::Diamonds: return "d";
        case Suit::Hearts:   return "h";
        case Suit::Spades:   return "s";
    }

    return "?";
}

inline std::string to_string(Rank rank) {
    switch (rank) {
        case Rank::Two:   return "2";
        case Rank::Three: return "3";
        case Rank::Four:  return "4";
        case Rank::Five:  return "5";
        case Rank::Six:   return "6";
        case Rank::Seven: return "7";
        case Rank::Eight: return "8";
        case Rank::Nine:  return "9";
        case Rank::Ten:   return "T";
        case Rank::Jack:  return "J";
        case Rank::Queen: return "Q";
        case Rank::King:  return "K";
        case Rank::Ace:   return "A";
    }

    return "?";
}

inline std::string to_string(CardId card) {
    return to_string(rank_of(card)) + to_string(suit_of(card));
}

inline CardId parse_card(const std::string& text) {
    if (text.size() != 2) {
        throw std::invalid_argument("Card string must have length 2, e.g. As or Td.");
    }

    Rank rank;

    switch (text[0]) {
        case '2': rank = Rank::Two;   break;
        case '3': rank = Rank::Three; break;
        case '4': rank = Rank::Four;  break;
        case '5': rank = Rank::Five;  break;
        case '6': rank = Rank::Six;   break;
        case '7': rank = Rank::Seven; break;
        case '8': rank = Rank::Eight; break;
        case '9': rank = Rank::Nine;  break;
        case 'T':
        case 't': rank = Rank::Ten;   break;
        case 'J':
        case 'j': rank = Rank::Jack;  break;
        case 'Q':
        case 'q': rank = Rank::Queen; break;
        case 'K':
        case 'k': rank = Rank::King;  break;
        case 'A':
        case 'a': rank = Rank::Ace;   break;
        default:
            throw std::invalid_argument("Invalid card rank character.");
    }

    Suit suit;

    switch (text[1]) {
        case 'c':
        case 'C': suit = Suit::Clubs;    break;
        case 'd':
        case 'D': suit = Suit::Diamonds; break;
        case 'h':
        case 'H': suit = Suit::Hearts;   break;
        case 's':
        case 'S': suit = Suit::Spades;   break;
        default:
            throw std::invalid_argument("Invalid card suit character.");
    }

    return make_card(rank, suit);
}

inline std::array<CardId, 52> full_deck() {
    std::array<CardId, 52> deck{};

    int index = 0;

    for (int suit = 0; suit < 4; ++suit) {
        for (int rank = 2; rank <= 14; ++rank) {
            deck[index++] = make_card(
                static_cast<Rank>(rank),
                static_cast<Suit>(suit)
            );
        }
    }

    return deck;
}

inline bool same_rank(CardId a, CardId b) {
    return rank_of(a) == rank_of(b);
}

inline bool same_suit(CardId a, CardId b) {
    return suit_of(a) == suit_of(b);
}

} // namespace poker