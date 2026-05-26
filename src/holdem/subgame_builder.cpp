#include "holdem/subgame_builder.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace poker::holdem {
namespace {

    GameAction to_game_action(const Action& action) {
        return GameAction{
            static_cast<int>(action.type),
            action.amount,
            action_history_token(action)
        };
    }

    GameAction private_deal_game_action(
        const PrivateState& private_state
    ) {
        return GameAction{
            0,
            0,
            "private_deal:" + poker::holdem::to_string(private_state)
        };
    }

    GameAction public_board_game_action(
        const Board& board
    ) {
        return GameAction{
            0,
            0,
            "deal_board:" + poker::to_string(board)
        };
    }

    GameAction synthetic_action_label(const std::string& label) {
        return GameAction{0, 0, label};
    }

    std::vector<GameAction> to_game_actions(
        const std::vector<Action>& actions
    ) {
        std::vector<GameAction> result;
        result.reserve(actions.size());

        for (const Action& action : actions) {
            result.push_back(to_game_action(action));
        }

        return result;
    }

    PublicState clear_terminal_all_in_marker(PublicState state) {
        if (state.terminal_reason != TerminalReason::AllInCalled) {
            throw std::invalid_argument(
                "Expected AllInCalled state."
            );
        }

        state.terminal = false;
        state.terminal_reason = TerminalReason::None;
        state.player_to_act = Player::P0;
        state.folded_player = Player::Terminal;
        state.winner = Player::Terminal;

        return state;
    }

    int add_node_and_link(
        poker::BuildContext& ctx,
        int parent_id,
        poker::Node node
    ) {
                node.parent = parent_id;

                const int node_id = ctx.add_node(node);

                if (parent_id >= 0) {
                    ctx.add_child(parent_id, node_id);
                }

                return node_id;
    }

} // namespace

    InfoSetKey HoldemSubgameBuilder::make_key(
        Player player,
        const PublicState& public_state,
        const PrivateState& private_state
    ) const {
        return make_infoset_key(
            player,
            public_state,
            private_state,
            *config_.hand_abstraction,
            config_.board_abstraction.get()
        );
    }
    HoldemSubgameBuilder::HoldemSubgameBuilder(
        HoldemSubgameConfig config
    )
        : config_(std::move(config)),
          betting_engine_{},
          chance_model_(config_.board_abstraction),
          hand_evaluator_{} {}

    Game HoldemSubgameBuilder::build() const {
        validate_config();

        HoldemBuildContext ctx;

        const PublicState initial_public = config_.initial_public_state();

        const int root_id = add_root_chance_node(ctx);

        add_private_deal_children(
            ctx,
            root_id,
            initial_public
        );

        if (config_.validate_tree_during_build) {
            validate_built_game(ctx.game);
        }

        return std::move(ctx.game);
    }

    int HoldemSubgameBuilder::add_root_chance_node(
        BuildContext& ctx
    ) const {
        Node root;

        root.parent = -1;
        root.player = Player::Chance;
        root.infoset = -1;
        root.incoming_action = chance_deal_action();
        root.chance_prob = 0.0f;
        root.terminal = false;
        root.utility_p0 = 0.0f;

        const int root_id = ctx.add_node(root);

        ctx.game.root = root_id;
        ctx.game.num_players = 2;

        return root_id;
    }

    void HoldemSubgameBuilder::add_private_deal_children(
        HoldemBuildContext& ctx,
        int root_id,
        const PublicState& initial_public_state
    ) const {
        const std::vector<PrivateDealOutcome> deals =
            chance_model_.private_deal_outcomes(config_);

        for (const PrivateDealOutcome& deal : deals) {
            Node child;

            child.player = initial_public_state.player_to_act;
            child.infoset = -1;
            child.incoming_action = private_deal_game_action(deal.private_state);
            child.chance_prob = deal.probability;
            child.terminal = false;
            child.utility_p0 = 0.0f;

            const int child_id = add_node_and_link(ctx, root_id, child);

            expand_state(
                ctx,
                child_id,
                initial_public_state,
                deal.private_state
            );
        }
    }

    void HoldemSubgameBuilder::expand_state(
        HoldemBuildContext& ctx,
        int node_id,
        const PublicState& public_state,
        const PrivateState& private_state
    ) const {
        public_state.validate();
        private_state.validate();

        if (public_state.terminal) {
            return;
        }

        if (betting_engine_.betting_round_closed(public_state)) {
            if (should_go_to_showdown_after_betting_round(public_state)) {
                PublicState showdown = make_showdown_state(public_state);

                add_terminal_node(
                    ctx,
                    node_id,
                    synthetic_action_label("showdown"),
                    showdown,
                    private_state
                );

                return;
            }

            if (needs_public_chance_after_betting_round(public_state)) {
                Node chance_node;

                chance_node.player = Player::Chance;
                chance_node.infoset = -1;
                chance_node.incoming_action =
                    synthetic_action_label("public_chance");
                chance_node.chance_prob = 0.0f;
                chance_node.terminal = false;
                chance_node.utility_p0 = 0.0f;

                const int chance_id =
                    add_node_and_link(ctx, node_id, chance_node);

                expand_public_chance_node(
                    ctx,
                    chance_id,
                    public_state,
                    private_state
                );

                return;
            }
        }

        expand_decision_node(
            ctx,
            node_id,
            public_state,
            private_state
        );
    }

    void HoldemSubgameBuilder::expand_decision_node(
        HoldemBuildContext& ctx,
        int node_id,
        const PublicState& public_state,
        const PrivateState& private_state
    ) const {
        const Player node_player = ctx.game.node(node_id).player;

        if (node_player != Player::P0 && node_player != Player::P1) {
            throw std::logic_error(
                "expand_decision_node requires a real-player node."
            );
        }

        if (public_state.player_to_act != node_player) {
            throw std::logic_error(
                "PublicState player_to_act does not match decision node player."
            );
        }

        const std::vector<Action> legal_actions =
            betting_engine_.legal_actions(
                public_state,
                config_.betting_abstraction
            );

        if (legal_actions.empty()) {
            throw std::logic_error("Decision node has no legal actions.");
        }

        const int infoset_id =
            get_or_create_infoset(
                ctx,
                public_state.player_to_act,
                public_state,
                private_state,
                legal_actions
            );

        ctx.game.node(node_id).infoset = infoset_id;

        for (const Action& action : legal_actions) {
            PublicState next_public = betting_engine_.apply_action(public_state, action);

            if (next_public.terminal &&
                next_public.terminal_reason == TerminalReason::AllInCalled) {
                if (config_.collapse_all_in_runouts_to_ev) {
                    throw std::runtime_error(
                        "collapse_all_in_runouts_to_ev requires an AllInEquityResolver."
                    );
                }

                Node all_in_chance;

                all_in_chance.player = Player::Chance;
                all_in_chance.infoset = -1;
                all_in_chance.incoming_action = to_game_action(action);
                all_in_chance.chance_prob = 0.0f;
                all_in_chance.terminal = false;
                all_in_chance.utility_p0 = 0.0f;

                const int all_in_chance_id =
                    add_node_and_link(ctx, node_id, all_in_chance);

                expand_all_in_called_state(
                    ctx,
                    all_in_chance_id,
                    next_public,
                    private_state
                );

                continue;
            }

            if (next_public.terminal) {
                add_terminal_node(
                    ctx,
                    node_id,
                    action,
                    next_public,
                    private_state
                );

                continue;
            }

            // IMPORTANT: handle closed nonterminal betting rounds here.
            // River: action directly creates showdown terminal.
            // Flop/turn: action creates a public-card chance node.
            if (betting_engine_.betting_round_closed(next_public)) {
                if (should_go_to_showdown_after_betting_round(next_public)) {
                    PublicState showdown = make_showdown_state(next_public);

                    add_terminal_node(
                        ctx,
                        node_id,
                        action,
                        showdown,
                        private_state
                    );

                    continue;
                }

                if (needs_public_chance_after_betting_round(next_public)) {
                    Node chance_node;

                    chance_node.player = Player::Chance;
                    chance_node.infoset = -1;
                    chance_node.incoming_action = to_game_action(action);
                    chance_node.chance_prob = 0.0f;
                    chance_node.terminal = false;
                    chance_node.utility_p0 = 0.0f;

                    const int chance_id =
                        add_node_and_link(ctx, node_id, chance_node);

                    expand_public_chance_node(
                        ctx,
                        chance_id,
                        next_public,
                        private_state
                    );

                    continue;
                }
            }

            Node child;

            child.player = next_public.player_to_act;
            child.infoset = -1;
            child.incoming_action = to_game_action(action);
            child.chance_prob = 0.0f;
            child.terminal = false;
            child.utility_p0 = 0.0f;

            const int child_id =
                add_node_and_link(ctx, node_id, child);

            expand_state(
                ctx,
                child_id,
                next_public,
                private_state
            );
        }
    }

    void HoldemSubgameBuilder::expand_public_chance_node(
        HoldemBuildContext& ctx,
        int node_id,
        const PublicState& public_state,
        const PrivateState& private_state
    ) const {
        if (ctx.game.node(node_id).player != Player::Chance) {
            throw std::logic_error(
                "expand_public_chance_node requires a chance node."
            );
        }

        const Player next_to_act =
            first_player_to_act_on_next_street(public_state);

        const std::vector<PublicBoardOutcome> outcomes =
            chance_model_.public_board_outcomes(
                public_state,
                private_state,
                next_to_act
            );

        for (const PublicBoardOutcome& outcome : outcomes) {
            if (betting_engine_.betting_round_closed(outcome.public_state)) {
                throw std::logic_error(
                    "Public-card chance produced a next-street state whose betting "
                    "round is already closed. Did you forget to reset BettingState "
                    "when advancing streets?"
                );
            }

            Node child;

            child.player = outcome.public_state.player_to_act;
            child.infoset = -1;
            child.incoming_action =
                public_board_game_action(outcome.public_state.board);
            child.chance_prob = outcome.probability;
            child.terminal = false;
            child.utility_p0 = 0.0f;

            const int child_id =
                add_node_and_link(ctx, node_id, child);

            expand_state(
                ctx,
                child_id,
                outcome.public_state,
                private_state
            );
        }
    }

    void HoldemSubgameBuilder::expand_all_in_called_state(
        HoldemBuildContext& ctx,
        int node_id,
        const PublicState& public_state,
        const PrivateState& private_state
    ) const {
        if (!public_state.terminal ||
            public_state.terminal_reason != TerminalReason::AllInCalled) {
            throw std::logic_error(
                "expand_all_in_called_state requires AllInCalled terminal marker."
            );
        }

        if (ctx.game.node(node_id).player != Player::Chance) {
            throw std::logic_error(
                "All-in runout expansion must occur from a chance node."
            );
        }

        if (public_state.street == Street::River) {
            PublicState showdown = public_state;
            showdown.terminal = true;
            showdown.terminal_reason = TerminalReason::Showdown;
            showdown.player_to_act = Player::Terminal;
            showdown.folded_player = Player::Terminal;
            showdown.winner = Player::Terminal;

            add_terminal_node(
                ctx,
                node_id,
                synthetic_action_label("all_in_showdown"),
                showdown,
                private_state
            );

            return;
        }

        PublicState runout_state =
            clear_terminal_all_in_marker(public_state);

        const Player arbitrary_next_to_act = Player::P0;

        const std::vector<PublicBoardOutcome> outcomes =
            chance_model_.public_board_outcomes(
                runout_state,
                private_state,
                arbitrary_next_to_act
            );

        for (const PublicBoardOutcome& outcome : outcomes) {
            if (outcome.public_state.street == Street::River) {
                PublicState showdown = outcome.public_state;
                showdown.terminal = true;
                showdown.terminal_reason = TerminalReason::Showdown;
                showdown.player_to_act = Player::Terminal;
                showdown.folded_player = Player::Terminal;
                showdown.winner = Player::Terminal;

                add_terminal_node(
                    ctx,
                    node_id,
                    public_board_game_action(outcome.public_state.board),
                    showdown,
                    private_state,
                    outcome.probability
                );

                continue;
            }

            PublicState next_all_in = outcome.public_state;
            next_all_in.terminal = true;
            next_all_in.terminal_reason = TerminalReason::AllInCalled;
            next_all_in.player_to_act = Player::Terminal;
            next_all_in.folded_player = Player::Terminal;
            next_all_in.winner = Player::Terminal;

            Node child_chance;

            child_chance.player = Player::Chance;
            child_chance.infoset = -1;
            child_chance.incoming_action =
                public_board_game_action(outcome.public_state.board);
            child_chance.chance_prob = outcome.probability;
            child_chance.terminal = false;
            child_chance.utility_p0 = 0.0f;

            const int child_chance_id =
                add_node_and_link(ctx, node_id, child_chance);

            expand_all_in_called_state(
                ctx,
                child_chance_id,
                next_all_in,
                private_state
            );
        }
    }

    int HoldemSubgameBuilder::add_chance_node(
        HoldemBuildContext& ctx,
        int parent_id,
        const Action& incoming_action,
        float chance_probability,
        const PublicState& public_state,
        const PrivateState& private_state
    ) const {
        Node node;

        node.player = Player::Chance;
        node.infoset = -1;
        node.incoming_action = to_game_action(incoming_action);
        node.chance_prob = chance_probability;
        node.terminal = false;
        node.utility_p0 = 0.0f;

        (void)public_state;
        (void)private_state;

        return add_node_and_link(ctx, parent_id, node);
    }

    int HoldemSubgameBuilder::add_decision_node(
        HoldemBuildContext& ctx,
        int parent_id,
        const Action& incoming_action,
        const PublicState& public_state,
        const PrivateState& private_state
    ) const {
        Node node;

        node.player = public_state.player_to_act;
        node.infoset = -1;
        node.incoming_action = to_game_action(incoming_action);
        node.chance_prob = 0.0f;
        node.terminal = false;
        node.utility_p0 = 0.0f;

        (void)private_state;

        return add_node_and_link(ctx, parent_id, node);
    }

    int HoldemSubgameBuilder::add_terminal_node(
        HoldemBuildContext& ctx,
        int parent_id,
        const Action& incoming_action,
        const PublicState& public_state,
        const PrivateState& private_state
    ) const {
        return add_terminal_node(
            ctx,
            parent_id,
            to_game_action(incoming_action),
            public_state,
            private_state
        );
    }

    int HoldemSubgameBuilder::add_terminal_node(
        HoldemBuildContext& ctx,
        int parent_id,
        const GameAction& incoming_action,
        const PublicState& public_state,
        const PrivateState& private_state
    ) const {
        const float utility =
            terminal_utility_p0(
                public_state,
                private_state,
                hand_evaluator_
            );

        return add_terminal_node_with_utility(
            ctx,
            parent_id,
            incoming_action,
            public_state,
            utility
        );
    }

    int HoldemSubgameBuilder::add_terminal_node(
        HoldemBuildContext& ctx,
        int parent_id,
        const GameAction& incoming_action,
        const PublicState& public_state,
        const PrivateState& private_state,
        float chance_probability
    ) const {
        const float utility =
            terminal_utility_p0(
                public_state,
                private_state,
                hand_evaluator_
            );

        if (!public_state.terminal) {
            throw std::invalid_argument(
                "add_terminal_node requires terminal PublicState."
            );
        }

        Node node;

        node.player = Player::Terminal;
        node.infoset = -1;
        node.incoming_action = incoming_action;
        node.chance_prob = chance_probability;
        node.terminal = true;
        node.utility_p0 = utility;

        return add_node_and_link(ctx, parent_id, node);
    }

    int HoldemSubgameBuilder::add_terminal_node_with_utility(
        HoldemBuildContext& ctx,
        int parent_id,
        const Action& incoming_action,
        const PublicState& public_state,
        float utility_p0
    ) const {
        return add_terminal_node_with_utility(
            ctx,
            parent_id,
            to_game_action(incoming_action),
            public_state,
            utility_p0
        );
    }

    int HoldemSubgameBuilder::add_terminal_node_with_utility(
        HoldemBuildContext& ctx,
        int parent_id,
        const GameAction& incoming_action,
        const PublicState& public_state,
        float utility_p0
    ) const {
        if (!public_state.terminal) {
            throw std::invalid_argument(
                "add_terminal_node_with_utility requires terminal PublicState."
            );
        }

        Node node;

        node.player = Player::Terminal;
        node.infoset = -1;
        node.incoming_action = incoming_action;
        node.chance_prob = 0.0f;
        node.terminal = true;
        node.utility_p0 = utility_p0;

        return add_node_and_link(ctx, parent_id, node);
    }

    int HoldemSubgameBuilder::get_or_create_infoset(
        HoldemBuildContext& ctx,
        Player player,
        const PublicState& public_state,
        const PrivateState& private_state,
        const std::vector<Action>& legal_actions
    ) const {
        const InfoSetKey key =
            make_key(
                player,
                public_state,
                private_state
            );

        const std::string serialized_key =
            to_string(key);

        const std::vector<GameAction> actions =
            to_game_actions(legal_actions);

        return ctx.get_or_create_infoset(
            player,
            serialized_key,
            actions
        );
    }
    bool HoldemSubgameBuilder::needs_public_chance_after_betting_round(
        const PublicState& public_state
    ) const {
        return !public_state.terminal &&
               public_state.street != Street::River;
    }

    bool HoldemSubgameBuilder::should_go_to_showdown_after_betting_round(
        const PublicState& public_state
    ) const {
        return !public_state.terminal &&
               public_state.street == Street::River;
    }

    Player HoldemSubgameBuilder::first_player_to_act_on_next_street(
        const PublicState& /*public_state*/
    ) const {
        // For now, assume P0 is the out-of-position player and acts first
        // on every postflop street.
        //
        // Later, make this explicit:
        //   config_.out_of_position_player
        return Player::P0;
    }

    PublicState HoldemSubgameBuilder::make_showdown_state(
        PublicState state
    ) const {
        if (state.street != Street::River) {
            throw std::invalid_argument(
                "Showdown state can only be created on river."
            );
        }

        state.terminal = true;
        state.terminal_reason = TerminalReason::Showdown;
        state.player_to_act = Player::Terminal;
        state.folded_player = Player::Terminal;
        state.winner = Player::Terminal;

        state.validate();

        return state;
    }

    void HoldemSubgameBuilder::validate_config() const {
        config_.validate();
    }

    void HoldemSubgameBuilder::validate_built_game(
        const Game& game
    ) const {
        validate_game_basic_shape(game);
    }

} // namespace poker::holdem