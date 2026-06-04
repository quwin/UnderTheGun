// subgame_builder.cpp
//
// Public-tree Hold'em subgame builder.
//
// This version intentionally does NOT build:
//
//   root chance -> exact P0/P1 private hand pair -> full duplicated subtree
//
// Instead it builds:
//
//   public action/chance/terminal tree
//   + HandDomain for each player
//   + one HandPairTable
//   + ActionState tensor metadata per public decision node
//
// Private hands are not tree branches.

#include "holdem/subgame_builder.hpp"
#include "holdem/terminal_utility.hpp"
#include "holdem/action.hpp"
#include "holdem/board_abstraction.hpp"
#include "holdem/betting_engine.hpp"
#include "holdem/public_state.hpp"
#include "holdem/subgame_config.hpp"

#include "game.hpp"

#include "poker/board.hpp"
#include "poker/deck_mask.hpp"
#include "poker/hand.hpp"
#include "poker/range.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace poker::holdem {
namespace {
GameAction to_game_action(const Action& action) {
    return GameAction{
        static_cast<int>(action.type),
        action.amount
    };
}

PublicNode make_node_from_public_state(
    const PublicState& state,
    const PublicNodeType type,
    const Player player
) {
    PublicNode node;

    node.parent = -1;
    node.depth = 0;
    node.type = type;
    node.player = player;
    node.first_edge = 0;
    node.edge_count = 0;
    node.action_state_index = -1;
    return node;
}

bool should_go_to_showdown_after_betting_round(
    const BettingEngine& betting_engine,
    const PublicState& state
) {
    if (state.is_terminal()) {
        return false;
    }
    return state.board.is_river() && poker::holdem::BettingEngine::betting_round_closed(state);
}

bool needs_public_chance_after_betting_round(
    const BettingEngine& betting_engine,
    const PublicState& state
) {
    if (state.is_terminal()) {
        return false;
    }
    if (state.board.is_river()) {
        return false;
    }
    return poker::holdem::BettingEngine::betting_round_closed(state);
}

bool is_all_in_runout_state(const PublicState& state) {
    if (state.is_terminal()) {
        return false;
    }
    return state.either_player_all_in() && state.both_players_have_matched_current_bet();
}

} // namespace

// -----------------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------------

HoldemSubgameBuilder::HoldemSubgameBuilder(HoldemSubgameConfig  config)
    : config_(std::move(config)),
      betting_engine_{},
      all_in_equity_cache_() {
    if (!config_.board_abstraction) {
        throw std::invalid_argument(
            "HoldemSubgameBuilder requires board_abstraction."
        );
    }
    if (!config_.hand_abstraction) {
        throw std::invalid_argument(
            "HoldemSubgameBuilder requires hand_abstraction."
        );
    }
}

Game HoldemSubgameBuilder::build() const {
    Game game;
    game.num_players = 2;
    const PublicState root_state = config_.initial_public_state();
    game.starting_board = root_state.board;
    initialize_hand_data(game, root_state.board);
    all_in_equity_cache_ = std::make_unique<AllInEquityCache>();
    all_in_equity_cache_->initialize_for_subgame(root_state.board, config_.p0_range, config_.p1_range);
    const PublicNode root_node = make_node_from_public_state(
        root_state,
        PublicNodeType::Action,
        root_state.player_to_act
    );
    const int root_id = game.add_node(root_node);
    game.root = root_id;
    expand_public_state(game, root_id, root_state);
    return game;
}

void HoldemSubgameBuilder::validate_config() const {
    config_.validate();
    if (!config_.board_abstraction) {
        throw std::invalid_argument(
            "HoldemSubgameConfig board_abstraction is null."
        );
    }
    if (!config_.hand_abstraction) {
        throw std::invalid_argument(
            "HoldemSubgameConfig hand_abstraction is null."
        );
    }
}

// -----------------------------------------------------------------------------
// Hand domains / legal pairs
// -----------------------------------------------------------------------------
    Player HoldemSubgameBuilder::first_player_to_act_on_next_street() const {
        switch (config_.first_to_act_rule) {
            case FirstToActRule::OopActsFirst:
                return config_.oop_player;

            case FirstToActRule::P0ActsFirst:
                return Player::P0;

            case FirstToActRule::P1ActsFirst:
                return Player::P1;
        }

        throw std::logic_error("Unknown FirstToActRule.");
    }
    void HoldemSubgameBuilder::initialize_hand_data(Game& game,const Board &starting_board) const {
    HandDomain p0 = build_hand_domain(config_.p0_range, starting_board);
    HandDomain p1 = build_hand_domain(config_.p1_range, starting_board);
    HandPairTable pairs = build_hand_pair_table(p0, p1);
    game.set_hand_domains(
        std::move(p0),
        std::move(p1),
        std::move(pairs)
    );
}

HandDomain HoldemSubgameBuilder::build_hand_domain(
    const Range& range,
    const Board &starting_board
) {
    const DeckMask dead = board_mask(starting_board);
    HandDomain domain;
    domain.hands = range.legal_hands(dead);
    if (domain.hands.empty()) {
        throw std::invalid_argument(
            "Range has no legal hands after board blockers."
        );
    }

    return domain;
}

HandPairTable HoldemSubgameBuilder::build_hand_pair_table(
    const HandDomain& p0,
    const HandDomain& p1
) {
    if (p0.empty() || p1.empty()) {
        throw std::invalid_argument(
            "Cannot build HandPairTable from empty HandDomain."
        );
    }
    HandPairTable table;
    for (int i = 0; i < p0.hand_count(); ++i) {
        const HandId p0_hand = p0.hands[static_cast<std::size_t>(i)];
        const HoleCards p0_cards = hand_from_id(p0_hand);

        for (int j = 0; j < p1.hand_count(); ++j) {
            const HandId p1_hand = p1.hands[static_cast<std::size_t>(j)];
            const HoleCards p1_cards = hand_from_id(p1_hand);

            if (hands_overlap(p0_cards, p1_cards)) {
                continue;
            }

            table.p0_index.push_back(i);
            table.p1_index.push_back(j);
        }
    }

    if (table.empty()) {
        throw std::invalid_argument(
            "No legal non-overlapping hand pairs in subgame ranges."
        );
    }

    return table;
}

// -----------------------------------------------------------------------------
// Recursive public-tree expansion
// -----------------------------------------------------------------------------

void HoldemSubgameBuilder::expand_public_state(
    Game& game,
    const int node_id,
    const PublicState& state
) const {
    PublicNode& node = game.node(node_id);
    if (state.is_terminal()) {
        finalize_terminal_node(game, node_id, state);
        return;
    }
    if (node.type == PublicNodeType::Chance) {
        expand_public_chance_node(game, node_id, state);
        return;
    }
    if (node.type != PublicNodeType::Action) {
        throw std::logic_error(
            "Nonterminal public state must correspond to Action or Chance node."
        );
    }
    expand_action_node(game, node_id, state);
}

void HoldemSubgameBuilder::finalize_terminal_node(
    Game& game,
    const int node_id,
    const PublicState& state
) {
    PublicNode& node = game.node(node_id);
    node.type = PublicNodeType::Terminal;
    node.player = Player::Terminal;

    auto record = TerminalRecord();
    record.type = state.terminal_type;
    record.pot = state.pot;
    record.p0_committed = state.betting.p0_committed_this_round;
    record.board_index = make_board_index(game.starting_board, state.board);
    game.terminal_records.emplace_back(record);
}

float HoldemSubgameBuilder::terminal_value_for_pair(
    const Game& game,
    const PublicState& terminal_state,
    const int hand_pair_id
) const {
    if (hand_pair_id < 0 || hand_pair_id >= game.hand_pairs.pair_count()) {
        throw std::invalid_argument(
            "hand_pair_id is out of range."
        );
    }
    const int p0_domain_index = game.hand_pairs.p0_index.at(hand_pair_id);
    const int p1_domain_index = game.hand_pairs.p1_index.at(hand_pair_id);
    const HandId p0_hand_id = game.p0_hands.hands.at(p0_domain_index);
    const HandId p1_hand_id = game.p1_hands.hands.at(p1_domain_index);
    const PrivateState private_state{hand_from_id(p0_hand_id),hand_from_id(p1_hand_id)};
    // Exact public runouts may collide with a root hand pair.
    // That public/private combination is impossible.
    if (private_state.overlaps_mask(board_mask(terminal_state.board))) {
        return 0.0f;
    }
    return terminal_utility_p0(terminal_state, private_state, all_in_equity_cache_.get());
}

void HoldemSubgameBuilder::expand_action_node(
    Game& game,
    int node_id,
    const PublicState& state
) const {
    PublicNode& node = game.node(node_id);

    node.type = PublicNodeType::Action;
    node.player = state.player_to_act;
    const std::vector<Action> legal_actions =
        betting_engine_.legal_actions(
            state,
            config_.betting_abstraction
        );
    if (legal_actions.empty()) {
        throw std::logic_error(
            "Action node has no legal actions."
        );
    }
    struct PendingChild {
        int child_id = -1;
        int action_id = -1;
        PublicState child_state;
    };
    std::vector<PendingChild> pending;
    pending.reserve(legal_actions.size());

    // Important:
    //   Create all child nodes and attach all edges before recursively
    //   expanding any child. This preserves contiguous edge ranges in Game.
    for (const Action& action : legal_actions) {
        PublicState next = betting_engine_.apply_action(state, action);
        PublicState node_state = normalize_post_action_state(next);

        const int child_id = add_node_for_state(game, node_id, node_state);
        const int action_id = game.add_action(to_game_action(action));

        game.add_action_child(
            node_id,
            child_id,
            action_id
        );

        pending.push_back(
            PendingChild{
                child_id,
                action_id,
                std::move(node_state)
            }
        );
    }

    game.register_action_state(
        node_id,
        state.player_to_act
    );

    for (const PendingChild& child : pending) {
        expand_public_state(
            game,
            child.child_id,
            child.child_state
        );
    }
}

void HoldemSubgameBuilder::expand_public_chance_node(
    Game& game,
    int node_id,
    const PublicState& state
) const {


    if (state.board.is_river()) {
        throw std::logic_error(
            "River state cannot have public chance children."
        );
    }
    PublicNode& node = game.node(node_id);
    node.type = PublicNodeType::Chance;
    node.player = Player::Chance;
    const std::vector<BoardTransition> transitions =
        config_.board_abstraction->next_board_transitions(
            state.board,
            board_mask(state.board)
        );

    // const DeckMask unavailable = board_mask(state.board);
    // const int exact_count =
    //     static_cast<int>(cards_from_mask(remaining_cards(unavailable)).size());
    //
    // std::cerr
    //     << "[chance] board=" << poker::to_string(state.board)
    //     << " exact=" << exact_count
    //     << " transitions=" << transitions.size()
    //     << " is_exact=" << config_.board_abstraction->is_exact()
    //     << "\n";
    if (transitions.empty()) {
        throw std::logic_error(
            "Public chance node has no board transitions."
        );
    }

    struct PendingChanceChild {
        int child_id = -1;
        int public_card = -1;
        float probability = 0.0f;
        PublicState child_state;
    };

    std::vector<PendingChanceChild> pending;
    pending.reserve(transitions.size());

    for (const BoardTransition& transition : transitions) {
        transition.board.validate();

        if (transition.probability <= 0.0f ||
            transition.probability > 1.0f ||
            !std::isfinite(transition.probability)) {
            throw std::invalid_argument(
                "BoardTransition probability must be finite and in (0, 1]."
            );
        }

        if (transition.board.size() != state.board.size() + 1) {
            throw std::invalid_argument(
                "Public chance transition must add exactly one card."
            );
        }

        const phevaluator::Card dealt_card =
            transition.board.cards.back();

        PublicState next = make_next_street_public_state(
            state,
            transition.board,
            first_player_to_act_on_next_street()
        );

        // There should not be any chance nodes in an all in runout state, it should be terminal to EV
        if (is_all_in_runout_state(state)) {
            if (config_.collapse_all_in_runouts_to_ev) {
                throw std::invalid_argument(
                "Chance Nodes should no longer occur after an All-In state with EV Collapse enabled."
                );
            }
            if (next.board.is_river()) {
                next.make_showdown_terminal();
            } else {
                // Keep it nonterminal but all-in, so next expansion creates
                // another chance node instead of an action node.
                next.player_to_act = Player::P0;
                next.validate();
            }
        }

        const int child_id = add_node_for_state(game, node_id, next);

        game.add_chance_child(
            node_id,
            child_id,
            static_cast<int>(dealt_card),
            transition.probability
        );

        pending.push_back(
            PendingChanceChild{
                child_id,
                static_cast<int>(dealt_card),
                transition.probability,
                std::move(next)
            }
        );
    }

    for (const PendingChanceChild& child : pending) {
        expand_public_state(
            game,
            child.child_id,
            child.child_state
        );
    }
}

// -----------------------------------------------------------------------------
// State normalization / node creation
// -----------------------------------------------------------------------------

PublicState HoldemSubgameBuilder::normalize_post_action_state(
    PublicState state
) const {
    if (state.is_terminal()) {
        if (state.terminal_type == TerminalType::P0_Fold || state.terminal_type == TerminalType::P1_Fold || state.terminal_type == TerminalType::Showdown) {
            return state;
        }
        if (state.terminal_type == TerminalType::AllIn) {
            if (config_.collapse_all_in_runouts_to_ev) {
                return state;
            }
            if (state.board.is_river()) {
                state.make_showdown_terminal();
                return state;
            }
            state.clear_all_in_terminal_for_runout();
            return state;
        }
        throw std::logic_error(
            "Unknown terminal reason after action."
        );
    }
    if (should_go_to_showdown_after_betting_round(betting_engine_,state)) {
        state.make_showdown_terminal();
        return state;
    }
    // Non-river closed betting round becomes a chance node.
    if (needs_public_chance_after_betting_round(betting_engine_, state)) {
        return state;
    }

    return state;
}

int HoldemSubgameBuilder::add_node_for_state(
    Game& game,
    int parent_id,
    const PublicState& state
) const {

    auto type = PublicNodeType::Action;
    Player player;
    if (state.is_terminal()) {
        type = PublicNodeType::Terminal;
        player = Player::Terminal;
    } else if (is_all_in_runout_state(state)) {
        if (!state.board.is_river() && !config_.collapse_all_in_runouts_to_ev) {
            type = PublicNodeType::Chance;
            player = Player::Chance;
        } else {
            type = PublicNodeType::Terminal;
            player = Player::Terminal;
        }
    } else if (needs_public_chance_after_betting_round(
                   betting_engine_,
                   state
               )) {
        type = PublicNodeType::Chance;
        player = Player::Chance;
    } else {
        type = PublicNodeType::Action;
        player = state.player_to_act;
    }
    PublicNode node = make_node_from_public_state(
        state,
        type,
        player
    );
    node.parent = parent_id;
    return game.add_node(node);
}
} // namespace poker::holdem