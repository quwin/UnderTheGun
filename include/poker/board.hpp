#pragma once

#include "card.hpp"
#include "deck_mask.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace poker {

enum class BoardStreet : int {
    Empty = 0,
    Flop  = 3,
    Turn  = 4,
    River = 5
};

struct Board {
    // Supports flop, turn, and river boards.
    //
    // Invariant:
    //   cards.size() must be 0, 3, 4, or 5 for normal Hold'em use.
    //
    // Tests may construct:
    //   Board{{card1, card2, card3}}
    //   Board{{card1, card2, card3, card4}}
    //   Board{{card1, card2, card3, card4, card5}}
    std::vector<CardId> cards;

    Board() = default;

    explicit Board(std::vector<CardId> input)
        : cards(std::move(input)) {
        validate();
    }

    Board(std::initializer_list<CardId> input)
        : cards(input) {
        validate();
    }

    int size() const {
        return static_cast<int>(cards.size());
    }

    bool empty() const {
        return cards.empty();
    }

    bool is_flop() const {
        return cards.size() == 3;
    }

    bool is_turn() const {
        return cards.size() == 4;
    }

    bool is_river() const {
        return cards.size() == 5;
    }

    BoardStreet street() const {
        switch (cards.size()) {
            case 0: return BoardStreet::Empty;
            case 3: return BoardStreet::Flop;
            case 4: return BoardStreet::Turn;
            case 5: return BoardStreet::River;
            default:
                throw std::logic_error("Invalid board size for Hold'em street.");
        }
    }

    CardId operator[](std::size_t index) const {
        return cards.at(index);
    }

    CardId& operator[](std::size_t index) {
        return cards.at(index);
    }

    bool contains(CardId card) const {
        return std::find(cards.begin(), cards.end(), card) != cards.end();
    }

    DeckMask mask() const {
        DeckMask result = empty_deck_mask();

        for (CardId card : cards) {
            result = add_card(result, card);
        }

        return result;
    }

    Board with_added_card(CardId card) const {
        if (cards.size() >= 5) {
            throw std::invalid_argument("Cannot add card to a complete river board.");
        }

        if (contains(card)) {
            throw std::invalid_argument("Cannot add duplicate card to board.");
        }

        Board next;
        next.cards = cards;
        next.cards.push_back(card);
        next.validate();

        return next;
    }

    void validate() const {
        if (!(cards.size() == 0 ||
              cards.size() == 3 ||
              cards.size() == 4 ||
              cards.size() == 5)) {
            throw std::invalid_argument(
                "Board must contain 0, 3, 4, or 5 cards."
            );
        }

        DeckMask seen = empty_deck_mask();

        for (CardId card : cards) {
            validate_card(card);

            if (contains_card(seen, card)) {
                throw std::invalid_argument("Board contains duplicate card.");
            }

            seen = add_card(seen, card);
        }
    }
};

inline bool operator==(const Board& a, const Board& b) {
    return a.cards == b.cards;
}

inline bool operator!=(const Board& a, const Board& b) {
    return !(a == b);
}

inline DeckMask board_mask(const Board& board) {
    return board.mask();
}

inline bool board_contains(const Board& board, CardId card) {
    return board.contains(card);
}

inline Board add_board_card(const Board& board, CardId card) {
    return board.with_added_card(card);
}

inline std::string to_string(BoardStreet street) {
    switch (street) {
        case BoardStreet::Empty: return "Empty";
        case BoardStreet::Flop:  return "Flop";
        case BoardStreet::Turn:  return "Turn";
        case BoardStreet::River: return "River";
    }

    return "Unknown";
}

inline std::string to_string(const Board& board) {
    std::string result;

    for (std::size_t i = 0; i < board.cards.size(); ++i) {
        if (i > 0) {
            result += " ";
        }

        result += to_string(board.cards[i]);
    }

    return result;
}

} // namespace poker