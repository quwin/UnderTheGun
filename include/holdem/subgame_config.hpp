#pragma once

#include "game.hpp"

#include "betting_abstraction.hpp"
#include "board_abstraction.hpp"
#include "hand_abstraction.hpp"
#include "public_state.hpp"

#include "poker/board.hpp"
#include "poker/range.hpp"

#include <memory>
#include <stdexcept>

namespace poker::holdem {

enum class FirstToActRule : int {
    // Use config.player_to_act for the root.
    // On later streets, use oop_player.
    OopActsFirst = 0,
    // Always use P0 after a public street transition.
    // Useful while testing.
    P0ActsFirst = 1,
    // Always use P1 after a public street transition.
    P1ActsFirst = 2
};

struct HoldemSubgameConfig {
    Board board;
    // ---------------------------------------------------------------------
    // Starting public state
    // ---------------------------------------------------------------------
    // Total pot at the start of the subgame.
    //
    // Convention:
    //   pot_size already includes all chips committed before this subgame.
    int pot_size = 0;
    // Remaining stack for each player at the start of the subgame.
    //
    // If you want asymmetric stacks later, replace effective_stack with
    // p0_stack / p1_stack.
    int effective_stack = 0;
    // Player to act at the root public state.
    Player player_to_act = Player::P0;
    // Position metadata used when advancing to later streets.
    //
    // In heads-up postflop Hold'em, OOP acts first on each street.
    Player oop_player = Player::P0;
    Player ip_player = Player::P1;
    FirstToActRule first_to_act_rule = FirstToActRule::OopActsFirst;
    // ---------------------------------------------------------------------
    // Private hand ranges
    // ---------------------------------------------------------------------
    //
    // These are not expanded as root chance branches.
    //
    // The public-tree builder uses them to build:
    //
    //   Game::p0_hands
    //   Game::p1_hands
    //   Game::hand_pairs
    //
    // Solver/evaluator code then uses those side tables for terminal values,
    // card removal, and hand-aware strategy tensors.
    Range p0_range;
    Range p1_range;
    // ---------------------------------------------------------------------
    // Betting abstraction
    // ---------------------------------------------------------------------
    BettingAbstraction betting_abstraction;
    // ---------------------------------------------------------------------
    // Card abstractions
    // ---------------------------------------------------------------------
    // Exact by default.
    //
    // In exact mode, one strategy bucket corresponds to one legal combo in
    // the player's HandDomain.
    //
    // In bucketed mode, the builder may map multiple exact hands into one
    // action-state bucket.
    std::shared_ptr<const HandAbstraction> hand_abstraction =
        make_exact_hand_abstraction();
    // Exact by default.
    //
    // Public-board abstraction affects public chance transitions only.
    std::shared_ptr<const BoardAbstraction> board_abstraction =
        make_exact_board_abstraction();
    // ---------------------------------------------------------------------
    // All-in handling
    // ---------------------------------------------------------------------
    // If true, an all-in call before the river becomes a single terminal node.
    // Its EV must be computed by a terminal/all-in evaluator over hand pairs.
    //
    // Cannot be true at the same time as expand_all_in_runouts.
    bool collapse_all_in_runouts_to_ev = true;


    // ---------------------------------------------------------------------
    // Validation
    // ---------------------------------------------------------------------

    void validate() const {
        if (pot_size < 0) {
            throw std::invalid_argument(
                "HoldemSubgameConfig pot_size must be nonnegative."
            );
        }

        if (effective_stack < 0) {
            throw std::invalid_argument(
                "HoldemSubgameConfig effective_stack must be nonnegative."
            );
        }

        if (player_to_act != Player::P0 &&
            player_to_act != Player::P1) {
            throw std::invalid_argument(
                "HoldemSubgameConfig player_to_act must be P0 or P1."
            );
        }

        if (oop_player != Player::P0 &&
            oop_player != Player::P1) {
            throw std::invalid_argument(
                "HoldemSubgameConfig oop_player must be P0 or P1."
            );
        }

        if (ip_player != Player::P0 &&
            ip_player != Player::P1) {
            throw std::invalid_argument(
                "HoldemSubgameConfig ip_player must be P0 or P1."
            );
        }

        if (oop_player == ip_player) {
            throw std::invalid_argument(
                "HoldemSubgameConfig oop_player and ip_player must differ."
            );
        }

        switch (first_to_act_rule) {
            case FirstToActRule::OopActsFirst:
            case FirstToActRule::P0ActsFirst:
            case FirstToActRule::P1ActsFirst:
                break;

            default:
                throw std::invalid_argument(
                    "HoldemSubgameConfig first_to_act_rule is invalid."
                );
        }

        if (p0_range.empty()) {
            throw std::invalid_argument(
                "HoldemSubgameConfig p0_range cannot be empty."
            );
        }

        if (p1_range.empty()) {
            throw std::invalid_argument(
                "HoldemSubgameConfig p1_range cannot be empty."
            );
        }

        betting_abstraction.validate();

        if (!hand_abstraction) {
            throw std::invalid_argument(
                "HoldemSubgameConfig hand_abstraction cannot be null."
            );
        }

        if (!board_abstraction) {
            throw std::invalid_argument(
                "HoldemSubgameConfig board_abstraction cannot be null."
            );
        }
    }

    // ---------------------------------------------------------------------
    // Initial public state
    // ---------------------------------------------------------------------

    [[nodiscard]] PublicState initial_public_state() const {
        validate();
        PublicState state;
        state.board = board;
        state.p0_stack = effective_stack;
        state.p1_stack = effective_stack;
        state.pot = pot_size;
        state.player_to_act = player_to_act;
        state.betting = BettingState{};
        state.validate();
        return state;
    }

    Player first_player_to_act_after_street_transition() const {
        switch (first_to_act_rule) {
            case FirstToActRule::OopActsFirst:
                return oop_player;
            case FirstToActRule::P0ActsFirst:
                return Player::P0;
            case FirstToActRule::P1ActsFirst:
                return Player::P1;
        }
        throw std::logic_error(
            "Invalid FirstToActRule in first_player_to_act_after_street_transition."
        );
    }
};

} // namespace poker::holdem