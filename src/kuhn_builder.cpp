#include "kuhn_builder.hpp"

#include <cassert>
#include <stdexcept>
#include <utility>

namespace poker {

namespace {

int card_rank(Card card) {
    switch (card) {
        case Card::Jack:  return 0;
        case Card::Queen: return 1;
        case Card::King:  return 2;
        case Card::None:  break;
    }

    throw std::runtime_error("Invalid card rank.");
}

bool p0_wins_showdown(Card p0_card, Card p1_card) {
    return card_rank(p0_card) > card_rank(p1_card);
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
    root.incoming_action = Action::Deal;
    root.chance_prob = 0.0f;
    root.p0_card = Card::None;
    root.p1_card = Card::None;
    root.history = "";
    root.terminal = false;
    root.utility_p0 = 0.0f;

    const int root_id = ctx.add_node(root);
    ctx.game.root = root_id;
    ctx.game.num_players = 2;

    add_chance_deals(ctx, root_id);

    return std::move(ctx.game);
}

void KuhnGameBuilder::add_chance_deals(
    BuildContext& ctx,
    int root_id
) const {
    int num_deals = 0;

    for (Card p0_card : config_.deck) {
        for (Card p1_card : config_.deck) {
            if (!config_.allow_redeal_same_card && p0_card == p1_card) {
                continue;
            }

            ++num_deals;
        }
    }

    assert(num_deals > 0);

    const float deal_probability = 1.0f / static_cast<float>(num_deals);

    for (Card p0_card : config_.deck) {
        for (Card p1_card : config_.deck) {
            if (!config_.allow_redeal_same_card && p0_card == p1_card) {
                continue;
            }

            Node deal_node;
            deal_node.depth = ctx.game.node(root_id).depth + 1;
            deal_node.player = Player::P0;
            deal_node.infoset = -1; // set by add_decision_node below
            deal_node.incoming_action = Action::Deal;
            deal_node.chance_prob = deal_probability;
            deal_node.p0_card = p0_card;
            deal_node.p1_card = p1_card;
            deal_node.history = "";
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
    Card p0_card,
    Card p1_card,
    const std::string& history
) const {
    assert(player_to_act == Player::P0 || player_to_act == Player::P1);

    Node& parent = ctx.game.node(parent_id);
    parent.player = player_to_act;
    parent.history = history;
    parent.p0_card = p0_card;
    parent.p1_card = p1_card;

    const std::vector<Action> actions = legal_actions(history);

    parent.infoset = ctx.get_or_create_infoset(
        player_to_act,
        private_card_for(player_to_act, p0_card, p1_card),
        history,
        actions
    );

    for (Action action : actions) {
        const std::string next_history = append_action(history, action);

        if (is_terminal_history(next_history)) {
            add_terminal_node(
                ctx,
                parent_id,
                action,
                p0_card,
                p1_card,
                next_history
            );
        } else {
            const Player next_player =
                next_player_after(player_to_act, next_history);

            const int child_id = add_decision_node(
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
}

int KuhnGameBuilder::add_terminal_node(
    BuildContext& ctx,
    int parent_id,
    Action incoming_action,
    Card p0_card,
    Card p1_card,
    const std::string& terminal_history
) const {
    const Node& parent = ctx.game.node(parent_id);

    Node node;
    node.depth = parent.depth + 1;
    node.player = Player::Terminal;
    node.infoset = -1;
    node.incoming_action = incoming_action;
    node.chance_prob = 0.0f;
    node.p0_card = p0_card;
    node.p1_card = p1_card;
    node.history = terminal_history;
    node.terminal = true;
    node.utility_p0 = terminal_utility_p0(
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
    Action incoming_action,
    Player player_to_act,
    Card p0_card,
    Card p1_card,
    const std::string& history
) const {
    const Node& parent = ctx.game.node(parent_id);

    Node node;
    node.depth = parent.depth + 1;
    node.player = player_to_act;
    node.infoset = -1; // assigned when subtree expands this node
    node.incoming_action = incoming_action;
    node.chance_prob = 0.0f;
    node.p0_card = p0_card;
    node.p1_card = p1_card;
    node.history = history;
    node.terminal = false;
    node.utility_p0 = 0.0f;

    const int node_id = ctx.add_node(node);
    ctx.add_child(parent_id, node_id);

    return node_id;
}

std::vector<Action> KuhnGameBuilder::legal_actions(
    const std::string& history
) const {
    if (history.empty()) {
        return {Action::Check, Action::Bet};
    }

    if (history == "c") {
        return {Action::Check, Action::Bet};
    }

    if (history == "b") {
        return {Action::Call, Action::Fold};
    }

    if (history == "cb") {
        return {Action::Call, Action::Fold};
    }

    if (is_terminal_history(history)) {
        return {};
    }

    throw std::runtime_error("Invalid Kuhn history: " + history);
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
    Card p0_card,
    Card p1_card,
    const std::string& terminal_history
) const {
    assert(p0_card != Card::None);
    assert(p1_card != Card::None);
    assert(p0_card != p1_card || config_.allow_redeal_same_card);

    const bool p0_wins = p0_wins_showdown(p0_card, p1_card);

    // Net utility convention:
    //
    // check-check:
    //   winner gains 1 net unit.
    //
    // bet-call:
    //   winner gains 2 net units.
    //
    // bet-fold:
    //   bettor gains 1 net unit.
    if (terminal_history == "cc") {
        return p0_wins ? +config_.ante : -config_.ante;
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
        "Cannot compute payoff for nonterminal history: " +
        terminal_history
    );
}

Player KuhnGameBuilder::next_player_after(
    Player current_player,
    const std::string& new_history
) const {
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

Card KuhnGameBuilder::private_card_for(
    Player player,
    Card p0_card,
    Card p1_card
) const {
    if (player == Player::P0) {
        return p0_card;
    }

    if (player == Player::P1) {
        return p1_card;
    }

    throw std::runtime_error("Chance/terminal nodes have no private card.");
}

std::string KuhnGameBuilder::append_action(
    const std::string& history,
    Action action
) const {
    switch (action) {
        case Action::Check:
            return history + "c";

        case Action::Bet:
            return history + "b";

        case Action::Call:
            return history + "c";

        case Action::Fold:
            return history + "f";

        case Action::Deal:
            break;
    }

    throw std::runtime_error("Cannot append invalid Kuhn action.");
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