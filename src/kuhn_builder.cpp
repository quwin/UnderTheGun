#include "kuhn_builder.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "../external/PokerHandEvaluator/cpp/include/phevaluator/card.h"

namespace poker {

namespace {

bool p0_wins_showdown(phevaluator::Card p0_card, phevaluator::Card p1_card) {
    return static_cast<int>(p0_card) > static_cast<int>(p1_card);
}

std::string kuhn_infoset_key(
    Player player,
    phevaluator::Card private_card,
    const std::string& public_history
) {
    return to_string(player) + "|" +
           private_card.describeRank() + "|" +
           public_history;
}

GameAction make_kuhn_game_action(holdem::ActionType action) {
    using holdem::ActionType;

    switch (action) {
        case ActionType::Check:
            return GameAction{
                static_cast<int>(ActionType::Check),
                0,
                "check"
            };

        case ActionType::Bet:
            return GameAction{
                static_cast<int>(ActionType::Bet),
                1,
                "bet"
            };

        case ActionType::Call:
            return GameAction{
                static_cast<int>(ActionType::Call),
                1,
                "call"
            };

        case ActionType::Fold:
            return GameAction{
                static_cast<int>(ActionType::Fold),
                0,
                "fold"
            };

        case ActionType::Raise:
        case ActionType::AllIn:
            break;
    }

    throw std::runtime_error(
        "Invalid Kuhn action type."
    );
}

std::vector<GameAction> make_kuhn_game_actions(
    const std::vector<holdem::ActionType>& actions
) {
    std::vector<GameAction> result;
    result.reserve(actions.size());

    for (holdem::ActionType action : actions) {
        result.push_back(make_kuhn_game_action(action));
    }

    return result;
}

GameAction make_private_deal_action(phevaluator::Card p0_card, phevaluator::Card p1_card) {
    return GameAction{
        0,
        0,
        "deal:p0=" + std::to_string(p0_card.describeRank()) + ",p1=" + p1_card.describeRank()
    };
}

} // namespace

KuhnGameBuilder::KuhnGameBuilder()
    : config_{} {}

KuhnGameBuilder::KuhnGameBuilder(KuhnConfig config)
    : config_(std::move(config)) {}

Game KuhnGameBuilder::build() const {
    BuildContext ctx;

    Node root;
    root.parent = -1;
    root.depth = 0;
    root.player = Player::Chance;
    root.infoset = -1;
    root.incoming_action = chance_deal_action();
    root.chance_prob = 0.0f;
    root.terminal = false;
    root.utility_p0 = 0.0f;

    const int root_id = ctx.add_node(root);

    ctx.game.root = root_id;
    ctx.game.num_players = 2;

    add_chance_deals(ctx, root_id);

    validate_game_basic_shape(ctx.game);

    return std::move(ctx.game);
}

void KuhnGameBuilder::add_chance_deals(
    BuildContext& ctx,
    int root_id
) const {
    int num_deals = 0;

    for (phevaluator::Card p0_card : config_.deck) {
        for (phevaluator::Card p1_card : config_.deck) {
            if (!config_.allow_redeal_same_card && p0_card == p1_card) {
                continue;
            }

            ++num_deals;
        }
    }

    if (num_deals <= 0) {
        throw std::runtime_error(
            "Kuhn deck produced no legal private-card deals."
        );
    }

    const float deal_probability =
        1.0f / static_cast<float>(num_deals);

    for (phevaluator::Card p0_card : config_.deck) {
        for (phevaluator::Card p1_card : config_.deck) {
            if (!config_.allow_redeal_same_card && p0_card == p1_card) {
                continue;
            }

            Node deal_node;
            deal_node.parent = root_id;
            deal_node.player = Player::P0;
            deal_node.infoset = -1;
            deal_node.incoming_action =
                make_private_deal_action(p0_card, p1_card);
            deal_node.chance_prob = deal_probability;
            deal_node.terminal = false;
            deal_node.utility_p0 = 0.0f;

            const int deal_id = ctx.add_node(deal_node);
            ctx.add_child(root_id, deal_id);

            add_betting_subtree(
                ctx,
                deal_id,
                Player::P0,
                p0_card,
                p1_card,
                ""
            );
        }
    }
}

void KuhnGameBuilder::add_betting_subtree(
    BuildContext& ctx,
    int parent_id,
    Player player_to_act,
    phevaluator::Card p0_card,
    phevaluator::Card p1_card,
    const std::string& history
) const {
    if (player_to_act != Player::P0 && player_to_act != Player::P1) {
        throw std::runtime_error(
            "Kuhn betting subtree requires P0 or P1 to act."
        );
    }

    Node& parent = ctx.game.node(parent_id);
    parent.player = player_to_act;

    const std::vector<holdem::ActionType> actions =
        legal_actions(history);

    const std::vector<GameAction> game_actions =
        make_kuhn_game_actions(actions);

    const std::string key =
        kuhn_infoset_key(
            player_to_act,
            private_card_for(player_to_act, p0_card, p1_card),
            history
        );

    parent.infoset =
        ctx.get_or_create_infoset(
            player_to_act,
            key,
            game_actions
        );

    for (holdem::ActionType action : actions) {
        const std::string next_history =
            append_action(history, action);

        if (is_terminal_history(next_history)) {
            add_terminal_node(
                ctx,
                parent_id,
                action,
                p0_card,
                p1_card,
                next_history
            );

            continue;
        }

        const Player next_player =
            next_player_after(player_to_act, next_history);

        const int child_id =
            add_decision_node(
                ctx,
                parent_id,
                action,
                next_player,
                p0_card,
                p1_card,
                next_history
            );

        add_betting_subtree(
            ctx,
            child_id,
            next_player,
            p0_card,
            p1_card,
            next_history
        );
    }
}

int KuhnGameBuilder::add_terminal_node(
    BuildContext& ctx,
    int parent_id,
    holdem::ActionType incoming_action,
    phevaluator::Card p0_card,
    phevaluator::Card p1_card,
    const std::string& terminal_history
) const {
    const Node& parent = ctx.game.node(parent_id);

    Node node;
    node.parent = parent.id;
    node.player = Player::Terminal;
    node.infoset = -1;
    node.incoming_action = make_kuhn_game_action(incoming_action);
    node.chance_prob = 0.0f;
    node.terminal = true;
    node.utility_p0 =
        terminal_utility_p0(
            p0_card,
            p1_card,
            terminal_history
        );

    const int node_id = ctx.add_node(node);
    ctx.add_child(parent_id, node_id);

    return node_id;
}

int KuhnGameBuilder::add_decision_node(
    BuildContext& ctx,
    int parent_id,
    holdem::ActionType incoming_action,
    Player player_to_act,
    phevaluator::Card /*p0_card*/,
    phevaluator::Card /*p1_card*/,
    const std::string& /*history*/
) const {
    if (player_to_act != Player::P0 && player_to_act != Player::P1) {
        throw std::runtime_error(
            "Kuhn decision node requires P0 or P1."
        );
    }

    const Node& parent = ctx.game.node(parent_id);

    Node node;
    node.parent = parent.id;
    node.player = player_to_act;
    node.infoset = -1;
    node.incoming_action = make_kuhn_game_action(incoming_action);
    node.chance_prob = 0.0f;
    node.terminal = false;
    node.utility_p0 = 0.0f;

    const int node_id = ctx.add_node(node);
    ctx.add_child(parent_id, node_id);

    return node_id;
}

std::vector<holdem::ActionType> KuhnGameBuilder::legal_actions(
    const std::string& history
) const {
    using holdem::ActionType;

    if (history == "") {
        return {
            ActionType::Check,
            ActionType::Bet
        };
    }

    if (history == "c") {
        return {
            ActionType::Check,
            ActionType::Bet
        };
    }

    if (history == "b") {
        return {
            ActionType::Call,
            ActionType::Fold
        };
    }

    if (history == "cb") {
        return {
            ActionType::Call,
            ActionType::Fold
        };
    }

    if (is_terminal_history(history)) {
        return {};
    }

    throw std::runtime_error(
        "Unknown Kuhn public history: " + history
    );
}

bool KuhnGameBuilder::is_terminal_history(
    const std::string& history
) const {
    return history == "cc"  ||
           history == "bc"  ||
           history == "bf"  ||
           history == "cbc" ||
           history == "cbf";
}

float KuhnGameBuilder::terminal_utility_p0(
    phevaluator::Card p0_card,
    phevaluator::Card p1_card,
    const std::string& terminal_history
) const {
    if (!is_terminal_history(terminal_history)) {
        throw std::runtime_error(
            "Cannot compute terminal utility for nonterminal history: " +
            terminal_history
        );
    }

    const bool p0_wins =
        p0_wins_showdown(p0_card, p1_card);

    if (terminal_history == "cc") {
        return p0_wins
            ? +config_.ante
            : -config_.ante;
    }

    if (terminal_history == "bc") {
        return p0_wins
            ? +(config_.ante + config_.bet_size)
            : -(config_.ante + config_.bet_size);
    }

    if (terminal_history == "cbc") {
        return p0_wins
            ? +(config_.ante + config_.bet_size)
            : -(config_.ante + config_.bet_size);
    }

    if (terminal_history == "bf") {
        // P0 bet, P1 folded.
        return +config_.ante;
    }

    if (terminal_history == "cbf") {
        // P0 checked, P1 bet, P0 folded.
        return -config_.ante;
    }

    throw std::runtime_error(
        "Unhandled terminal history: " + terminal_history
    );
}

Player KuhnGameBuilder::next_player_after(
    Player current_player,
    const std::string& new_history
) const {
    if (current_player != Player::P0 && current_player != Player::P1) {
        throw std::runtime_error(
            "next_player_after requires P0 or P1."
        );
    }

    if (is_terminal_history(new_history)) {
        return Player::Terminal;
    }

    if (new_history == "c") {
        return Player::P1;
    }

    if (new_history == "b") {
        return Player::P1;
    }

    if (new_history == "cb") {
        return Player::P0;
    }

    throw std::runtime_error(
        "Cannot determine next player after history: " +
        new_history
    );
}

phevaluator::Card KuhnGameBuilder::private_card_for(
    Player player,
    phevaluator::Card p0_card,
    phevaluator::Card p1_card
) const {
    if (player == Player::P0) {
        return p0_card;
    }

    if (player == Player::P1) {
        return p1_card;
    }

    throw std::runtime_error(
        "Chance/terminal nodes do not have a private card."
    );
}

std::string KuhnGameBuilder::append_action(
    const std::string& history,
    holdem::ActionType action
) const {
    using holdem::ActionType;

    switch (action) {
        case ActionType::Check:
            return history + "c";

        case ActionType::Bet:
            return history + "b";

        case ActionType::Call:
            return history + "c";

        case ActionType::Fold:
            return history + "f";

        case ActionType::Raise:
        case ActionType::AllIn:
            break;
    }

    throw std::runtime_error(
        "Cannot append invalid Kuhn action."
    );
}

Game build_kuhn_game() {
    KuhnGameBuilder builder;
    return builder.build();
}

Game build_kuhn_game(const KuhnConfig& config) {
    KuhnGameBuilder builder(config);
    return builder.build();
}

} // namespace poker