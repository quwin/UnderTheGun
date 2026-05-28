#pragma once

#include <stdexcept>
#include <string>

namespace poker::holdem {

enum class Street : int {
    Preflop = 0,
    Flop    = 1,
    Turn    = 2,
    River   = 3
};

inline bool is_valid_street(Street street) {
    switch (street) {
        case Street::Preflop:
        case Street::Flop:
        case Street::Turn:
        case Street::River:
            return true;
    }

    return false;
}

inline void validate_street(Street street) {
    if (!is_valid_street(street)) {
        throw std::invalid_argument("Invalid Hold'em street.");
    }
}

inline bool has_public_board(Street street) {
    switch (street) {
        case Street::Preflop:
            return false;

        case Street::Flop:
        case Street::Turn:
        case Street::River:
            return true;
    }

    return false;
}

inline int board_card_count_for_street(Street street) {
    switch (street) {
        case Street::Preflop:
            return 0;

        case Street::Flop:
            return 3;

        case Street::Turn:
            return 4;

        case Street::River:
            return 5;
    }

    throw std::invalid_argument("Invalid Hold'em street.");
}

inline bool is_preflop(Street street) {
    return street == Street::Preflop;
}

inline bool is_flop(Street street) {
    return street == Street::Flop;
}

inline bool is_turn(Street street) {
    return street == Street::Turn;
}

inline bool is_river(Street street) {
    return street == Street::River;
}

inline bool has_next_street(Street street) {
    switch (street) {
        case Street::Preflop:
        case Street::Flop:
        case Street::Turn:
            return true;

        case Street::River:
            return false;
    }

    return false;
}

inline Street next_street(Street street) {
    switch (street) {
        case Street::Preflop:
            return Street::Flop;

        case Street::Flop:
            return Street::Turn;

        case Street::Turn:
            return Street::River;

        case Street::River:
            throw std::invalid_argument("River has no next street.");
    }

    throw std::invalid_argument("Invalid Hold'em street.");
}

inline Street previous_street(Street street) {
    switch (street) {
        case Street::Preflop:
            throw std::invalid_argument("Preflop has no previous street.");

        case Street::Flop:
            return Street::Preflop;

        case Street::Turn:
            return Street::Flop;

        case Street::River:
            return Street::Turn;
    }

    throw std::invalid_argument("Invalid Hold'em street.");
}

inline int cards_to_deal_on_next_street(Street street) {
    switch (street) {
        case Street::Preflop:
            return 3; // flop

        case Street::Flop:
            return 1; // turn

        case Street::Turn:
            return 1; // river

        case Street::River:
            return 0;
    }

    throw std::invalid_argument("Invalid Hold'em street.");
}

inline bool board_size_matches_street(Street street, int board_size) {
    return board_size == board_card_count_for_street(street);
}

inline void validate_board_size_for_street(Street street, int board_size) {
    if (!board_size_matches_street(street, board_size)) {
        throw std::invalid_argument(
            "Board size does not match Hold'em street."
        );
    }
}

inline std::string to_string(Street street) {
    switch (street) {
        case Street::Preflop:
            return "preflop";

        case Street::Flop:
            return "flop";

        case Street::Turn:
            return "turn";

        case Street::River:
            return "river";
    }

    return "unknown";
}

inline Street parse_street(const std::string& text) {
    if (text == "preflop" || text == "Preflop" || text == "PREFLOP") {
        return Street::Preflop;
    }

    if (text == "flop" || text == "Flop" || text == "FLOP") {
        return Street::Flop;
    }

    if (text == "turn" || text == "Turn" || text == "TURN") {
        return Street::Turn;
    }

    if (text == "river" || text == "River" || text == "RIVER") {
        return Street::River;
    }

    throw std::invalid_argument("Could not parse Hold'em street.");
}

} // namespace poker::holdem