#pragma once

#include "game.hpp"

#include "action.hpp"
#include "betting_engine.hpp"
#include "chance_model.hpp"
#include "infoset_key.hpp"
#include "private_state.hpp"
#include "public_state.hpp"
#include "subgame_config.hpp"
#include "terminal_utility.hpp"

#include "poker/hand_evaluator.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace poker::holdem {

using HoldemBuildContext = BuildContext;

class HoldemSubgameBuilder {
public:
    explicit HoldemSubgameBuilder(HoldemSubgameConfig config);

    // Builds the explicit extensive-form subgame tree.
    Game build() const;

private:
    HoldemSubgameConfig config_;

    BettingEngine betting_engine_;
    ChanceModel chance_model_;
    HandEvaluator hand_evaluator_;

    // ---------------------------------------------------------------------
    // Top-level construction
    // ---------------------------------------------------------------------

    int add_root_chance_node(
        HoldemBuildContext& ctx
    ) const;

    void add_private_deal_children(
        HoldemBuildContext& ctx,
        int root_id,
        const PublicState& initial_public_state
    ) const;

    // ---------------------------------------------------------------------
    // Recursive state expansion
    // ---------------------------------------------------------------------

    void expand_state(
        HoldemBuildContext& ctx,
        int node_id,
        const PublicState& public_state,
        const PrivateState& private_state
    ) const;

    void expand_decision_node(
        HoldemBuildContext& ctx,
        int node_id,
        const PublicState& public_state,
        const PrivateState& private_state
    ) const;

    void expand_public_chance_node(
        HoldemBuildContext& ctx,
        int node_id,
        const PublicState& public_state,
        const PrivateState& private_state
    ) const;

    void expand_all_in_called_state(
        HoldemBuildContext& ctx,
        int node_id,
        const PublicState& public_state,
        const PrivateState& private_state
    ) const;

    // ---------------------------------------------------------------------
    // Node creation
    // ---------------------------------------------------------------------

    int add_chance_node(
        HoldemBuildContext& ctx,
        int parent_id,
        const Action& incoming_action,
        float chance_probability,
        const PublicState& public_state,
        const PrivateState& private_state
    ) const;

    int add_decision_node(
        HoldemBuildContext& ctx,
        int parent_id,
        const Action& incoming_action,
        const PublicState& public_state,
        const PrivateState& private_state
    ) const;

    int add_terminal_node(
        HoldemBuildContext& ctx,
        int parent_id,
        const Action& incoming_action,
        const PublicState& public_state,
        const PrivateState& private_state
    ) const;

    int add_terminal_node(HoldemBuildContext &ctx, int parent_id, const GameAction &incoming_action,
                          const PublicState &public_state, const PrivateState &private_state) const;

    int add_terminal_node(HoldemBuildContext &ctx, int parent_id, const GameAction &incoming_action,
                          const PublicState &public_state, const PrivateState &private_state,
                          float chance_probability) const;

    int add_terminal_node_with_utility(
        HoldemBuildContext& ctx,
        int parent_id,
        const Action& incoming_action,
        const PublicState& public_state,
        float utility_p0
    ) const;

    int add_terminal_node_with_utility(HoldemBuildContext &ctx, int parent_id, const GameAction &incoming_action,
                                       const PublicState &public_state, float utility_p0) const;

    // ---------------------------------------------------------------------
    // Infoset management
    // ---------------------------------------------------------------------

    int get_or_create_infoset(
        HoldemBuildContext& ctx,
        Player player,
        const PublicState& public_state,
        const PrivateState& private_state,
        const std::vector<Action>& legal_actions
    ) const;

    InfoSetKey make_key(
        Player player,
        const PublicState& public_state,
        const PrivateState& private_state
    ) const;

    // ---------------------------------------------------------------------
    // Street / terminal helpers
    // ---------------------------------------------------------------------

    bool needs_public_chance_after_betting_round(
        const PublicState& public_state
    ) const;

    bool should_go_to_showdown_after_betting_round(
        const PublicState& public_state
    ) const;

    Player first_player_to_act_on_next_street(
        const PublicState& public_state
    ) const;

    PublicState make_showdown_state(
        PublicState state
    ) const;

    // ---------------------------------------------------------------------
    // Validation helpers
    // ---------------------------------------------------------------------

    void validate_config() const;

    void validate_built_game(
        const Game& game
    ) const;
};

} // namespace poker::holdem