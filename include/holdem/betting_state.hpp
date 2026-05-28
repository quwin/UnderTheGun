#pragma once

#include "game.hpp"

#include <stdexcept>
#include <string>

namespace poker::holdem {

struct BettingState {
    int actions_this_street = 0;
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
    // Size of the latest bet/raise increment.
    //
    // Example:
    //   P0 bets 500:
    //     last_raise_size = 500
    //
    //   P1 raises from 500 to 1500:
    //     last_raise_size = 1000
    int last_raise_size = 0;
    // Number of raises after the initial bet on this street.
    //
    // Suggested convention:
    //   first bet does not count as a raise
    //   bet-raise => num_raises_this_street = 1
    //   bet-raise-reraise => num_raises_this_street = 2
    int num_raises_this_street = 0;
    // True once any bet or raise has occurred this street.
    bool round_has_bet = false;
    // Used to detect check-check closure.
    bool last_action_was_check = false;
    // Last player to bet or raise.
    //
    // Chance means there has been no aggressor yet.
    Player last_aggressor = Player::Chance;
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
        round_has_bet = true;
        last_aggressor = player;
        current_bet_to_call = amount;
        switch (player) {
            case Player::P0:
                last_raise_size = amount - p0_committed_this_round;
                p0_committed_this_round = amount;
                return;
            case Player::P1:
                last_raise_size = amount - p0_committed_this_round;
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

    int amount_to_call(Player player) const {
        const int needed = current_bet_to_call - committed(player);
        return needed > 0 ? needed : 0;
    }

    bool player_is_facing_bet(Player player) const {
        return amount_to_call(player) > 0;
    }

    bool unopened() const {
        return !round_has_bet && current_bet_to_call == 0;
    }

    bool has_live_bet() const {
        return round_has_bet && current_bet_to_call > 0;
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
        last_raise_size = 0;
        num_raises_this_street = 0;

        round_has_bet = false;
        last_action_was_check = false;
        last_aggressor = Player::Chance;

        actions_this_street = 0;
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

        if (last_raise_size < 0) {
            throw std::invalid_argument(
                "last_raise_size must be nonnegative."
            );
        }

        if (num_raises_this_street < 0) {
            throw std::invalid_argument(
                "num_raises_this_street must be nonnegative."
            );
        }

        if (last_aggressor != Player::Chance &&
            last_aggressor != Player::P0 &&
            last_aggressor != Player::P1) {
            throw std::invalid_argument(
                "last_aggressor must be Chance, P0, or P1."
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

        if (!round_has_bet && current_bet_to_call != 0) {
            throw std::invalid_argument(
                "Unbet round cannot have current_bet_to_call > 0."
            );
        }

        if (!round_has_bet && last_aggressor != Player::Chance) {
            throw std::invalid_argument(
                "Unbet round cannot have a real last aggressor."
            );
        }
    }
};

inline std::string to_string(const BettingState& state) {
    return "p0_commit=" + std::to_string(state.p0_committed_this_round) +
           "|p1_commit=" + std::to_string(state.p1_committed_this_round) +
           "|bet_to_call=" + std::to_string(state.current_bet_to_call) +
           "|last_raise=" + std::to_string(state.last_raise_size) +
           "|raises=" + std::to_string(state.num_raises_this_street) +
           "|round_has_bet=" + std::to_string(state.round_has_bet ? 1 : 0) +
           "|last_check=" + std::to_string(state.last_action_was_check ? 1 : 0) +
           "|last_aggressor=" + poker::to_string(state.last_aggressor);
}

inline BettingState make_fresh_betting_state() {
    BettingState state;
    state.reset_for_new_street();
    return state;
}

} // namespace poker::holdem