#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace poker::holdem {

enum class BetSizeType : int {
    PotFraction = 0,
    RaiseMultiplier = 1,
    FixedAmount = 2,
    AllIn = 3
};

struct BetSize {
    BetSizeType type = BetSizeType::PotFraction;
    double value = 1.0;

    static BetSize pot_fraction(double fraction) {
        BetSize size;
        size.type = BetSizeType::PotFraction;
        size.value = fraction;
        size.validate();
        return size;
    }

    static BetSize raise_multiplier(double multiplier) {
        BetSize size;
        size.type = BetSizeType::RaiseMultiplier;
        size.value = multiplier;
        size.validate();
        return size;
    }

    static BetSize fixed_amount(int amount) {
        BetSize size;
        size.type = BetSizeType::FixedAmount;
        size.value = static_cast<double>(amount);
        size.validate();
        return size;
    }

    static BetSize all_in() {
        BetSize size;
        size.type = BetSizeType::AllIn;
        size.value = 0.0;
        return size;
    }

    void validate() const {
        switch (type) {
            case BetSizeType::PotFraction:
                if (!std::isfinite(value) || value <= 0.0) {
                    throw std::invalid_argument(
                        "Pot-fraction bet size must be positive and finite."
                    );
                }
                break;

            case BetSizeType::RaiseMultiplier:
                if (!std::isfinite(value) || value <= 1.0) {
                    throw std::invalid_argument(
                        "Raise multiplier must be greater than 1 and finite."
                    );
                }
                break;

            case BetSizeType::FixedAmount:
                if (!std::isfinite(value) || value <= 0.0) {
                    throw std::invalid_argument(
                        "Fixed bet amount must be positive and finite."
                    );
                }
                break;

            case BetSizeType::AllIn:
                break;
        }
    }
};

struct BettingAbstraction {
    std::vector<BetSize> first_bet_sizes;
    std::vector<BetSize> raise_sizes;

    int max_raises_per_street = 1;
    int chip_unit = 1;
    bool always_allow_all_in = false;

    void validate() const {
        if (max_raises_per_street < 0) {
            throw std::invalid_argument(
                "max_raises_per_street must be nonnegative."
            );
        }

        if (chip_unit <= 0) {
            throw std::invalid_argument("chip_unit must be positive.");
        }

        for (const BetSize& size : first_bet_sizes) {
            size.validate();
        }

        for (const BetSize& size : raise_sizes) {
            size.validate();
        }
    }
};

inline std::string to_string(BetSizeType type) {
    switch (type) {
        case BetSizeType::PotFraction:     return "pot_fraction";
        case BetSizeType::RaiseMultiplier: return "raise_multiplier";
        case BetSizeType::FixedAmount:     return "fixed_amount";
        case BetSizeType::AllIn:           return "all_in";
    }

    return "unknown";
}

inline std::string to_string(const BetSize& size) {
    switch (size.type) {
        case BetSizeType::PotFraction:
            return "pot_fraction:" + std::to_string(size.value);

        case BetSizeType::RaiseMultiplier:
            return "raise_multiplier:" + std::to_string(size.value);

        case BetSizeType::FixedAmount:
            return "fixed_amount:" +
                   std::to_string(static_cast<int>(size.value));

        case BetSizeType::AllIn:
            return "all_in";
    }

    return "unknown";
}

inline BettingAbstraction make_tiny_betting_abstraction() {
    BettingAbstraction abstraction;

    abstraction.first_bet_sizes = {
        BetSize::pot_fraction(1.0)
    };

    abstraction.raise_sizes = {};
    abstraction.max_raises_per_street = 0;
    abstraction.chip_unit = 1;
    abstraction.always_allow_all_in = false;

    return abstraction;
}

inline BettingAbstraction make_standard_abstraction() {
    BettingAbstraction abstraction;

    abstraction.first_bet_sizes = {
        BetSize::pot_fraction(0.50),
        BetSize::pot_fraction(1.0),
        BetSize::all_in()
    };

    abstraction.raise_sizes = {
        BetSize::raise_multiplier(2.5),
        BetSize::all_in()
    };

    abstraction.max_raises_per_street = 1;
    abstraction.chip_unit = 1;
    abstraction.always_allow_all_in = false;

    return abstraction;
}

} // namespace poker::holdem