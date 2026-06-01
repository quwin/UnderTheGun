#pragma once

#include "../game.hpp"
#include "betting_engine.hpp"
#include "public_state.hpp"
#include "subgame_config.hpp"
#include "all_in_equity_cache.hpp"

#include <vector>
#include <cstdint>

namespace poker::holdem {
    class HoldemSubgameBuilder {
    public:
        explicit HoldemSubgameBuilder(HoldemSubgameConfig  config);
        [[nodiscard]] Game build() const;
        [[nodiscard]] Player first_player_to_act_on_next_street() const;
    private:
        HoldemSubgameConfig config_;
        BettingEngine betting_engine_;
        mutable std::unique_ptr<AllInEquityCache> all_in_equity_cache_;
        void validate_config() const;

        void initialize_hand_data(Game& game, const Board &starting_board) const;
        static HandDomain build_hand_domain(const Range& range, const Board& starting_board) ;

        static HandPairTable build_hand_pair_table(
            const HandDomain& p0,
            const HandDomain& p1
        );

        void finalize_terminal_node(
            Game& game,
            int node_id,
            const PublicState& state
        ) const;

        [[nodiscard]] float terminal_value_for_pair(
            const Game& game,
            const PublicState& terminal_state,
            int hand_pair_id
        ) const;

        void expand_public_state(
            Game& game,
            int node_id,
            const PublicState& state
        ) const;

        void expand_action_node(
            Game& game,
            int node_id,
            const PublicState& state
        ) const;

        void expand_public_chance_node(
            Game& game,
            int node_id,
            const PublicState& state
        ) const;

        [[nodiscard]] PublicState normalize_post_action_state(
            PublicState state
        ) const;

        int add_node_for_state(
            Game& game,
            int parent_id,
            const PublicState& state
        ) const;
    };

} // namespace poker::holdem