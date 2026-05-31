#pragma once

#include "game.hpp"

#include <stdexcept>
#include <string>

namespace poker::holdem {

struct BettingState {
    // Chips committed by each player during this street only.
    int p0_committed_this_round = 0;
    int p1_committed_this_round = 0;
    // Highest committed amount this street.
    //
    // Example:
    //   no bet:
    //     current_bet_to_call = 0
    //
    //   P0 bets 500:
    //     p0_committed_this_round = 500
    //     p1_committed_this_round = 0
    //     current_bet_to_call = 500
    //
    //   P1 raises to 1500:
    //     p0_committed_this_round = 500
    //     p1_committed_this_round = 1500
    //     current_bet_to_call = 1500
    int current_bet_to_call = 0;
    // Number of raises after the initial bet on this street.
    //
    // Suggested convention:
    //   first bet does not count as a raise
    //   bet-raise => num_raises_this_street = 1
    //   bet-raise-reraise => num_raises_this_street = 2
    int num_raises_this_street = 0;
    int actions_this_street = 0;
    int committed(Player player) const {
        switch (player) {
            case Player::P0:
                return p0_committed_this_round;

            case Player::P1:
                return p1_committed_this_round;

            default:
                throw std::invalid_argument(
                    "BettingState::committed requires P0 or P1."
                );
        }
    }
    void set_committed(Player player, int amount) {
        if (amount <= 0) {
            throw std::invalid_argument("Committed amount must be greater than zero.");
        }
        current_bet_to_call = amount;
        switch (player) {
            case Player::P0:
                p0_committed_this_round = amount;
                return;
            case Player::P1:
                p1_committed_this_round = amount;
                return;
            default:
                throw std::invalid_argument(
                    "BettingState::set_committed requires P0 or P1."
                );
        }
    }

    void add_committed(Player player, int additional_chips) {
        if (additional_chips < 0) {
            throw std::invalid_argument("Additional committed chips must be nonnegative.");
        }

        set_committed(
            player,
            committed(player) + additional_chips
        );
    }
    [[nodiscard]] bool both_players_checked() const {
        return unopened() &&
               actions_this_street >= 2 &&
               p0_committed_this_round == 0 &&
               p1_committed_this_round == 0;
    }

    [[nodiscard]] bool bet_was_called() const {
        return has_live_bet() && commitments_matched();
    }
    int amount_to_call(Player player) const {
        const int needed = current_bet_to_call - committed(player);
        return needed > 0 ? needed : 0;
    }

    bool player_is_facing_bet(Player player) const {
        return amount_to_call(player) > 0;
    }

    bool unopened() const {
        return current_bet_to_call == 0;
    }

    bool has_live_bet() const {
        return current_bet_to_call > 0;
    }

    bool commitments_matched() const {
        return p0_committed_this_round == p1_committed_this_round;
    }

    int highest_commitment() const {
        return p0_committed_this_round > p1_committed_this_round ? p0_committed_this_round : p1_committed_this_round;
    }

    int lowest_commitment() const {
        return p0_committed_this_round < p1_committed_this_round ? p0_committed_this_round : p1_committed_this_round;
    }

    int outstanding_call_amount() const {
        return highest_commitment() - lowest_commitment();
    }

    void reset_for_new_street() {
        p0_committed_this_round = 0;
        p1_committed_this_round = 0;

        current_bet_to_call = 0;
        num_raises_this_street = 0;
    }

    void validate() const {
        if (p0_committed_this_round < 0 ||
            p1_committed_this_round < 0) {
            throw std::invalid_argument(
                "Committed amounts must be nonnegative."
            );
        }

        if (current_bet_to_call < 0) {
            throw std::invalid_argument(
                "current_bet_to_call must be nonnegative."
            );
        }

        if (num_raises_this_street < 0) {
            throw std::invalid_argument(
                "num_raises_this_street must be nonnegative."
            );
        }
        if (current_bet_to_call < highest_commitment()) {
            throw std::invalid_argument(
                "current_bet_to_call cannot be less than highest commitment: " + std::to_string(current_bet_to_call) + " instead of " + std::to_string(highest_commitment())
            );
        }

        if (current_bet_to_call > highest_commitment()) {
            throw std::invalid_argument(
                "current_bet_to_call cannot exceed highest commitment: " + std::to_string(current_bet_to_call) + " instead of " + std::to_string(highest_commitment())
            );
        }
    }
};

inline std::string to_string(const BettingState& state) {
    return "p0_commit=" + std::to_string(state.p0_committed_this_round) +
           "|p1_commit=" + std::to_string(state.p1_committed_this_round) +
           "|bet_to_call=" + std::to_string(state.current_bet_to_call) +
           "|raises=" + std::to_string(state.num_raises_this_street);
}

} // namespace poker::holdem