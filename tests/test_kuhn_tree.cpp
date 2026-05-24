#include "kuhn_builder.hpp"

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kTol = 1e-6;

void check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void check_near(
    double actual,
    double expected,
    double tolerance,
    const std::string& message
) {
    if (std::abs(actual - expected) > tolerance) {
        std::ostringstream oss;
        oss << message
            << " actual=" << actual
            << " expected=" << expected
            << " tolerance=" << tolerance;
        throw std::runtime_error(oss.str());
    }
}

int card_rank(poker::Card card) {
    switch (card) {
        case poker::Card::Jack:  return 0;
        case poker::Card::Queen: return 1;
        case poker::Card::King:  return 2;
        case poker::Card::None:  break;
    }

    throw std::runtime_error("Invalid card.");
}

bool p0_wins_showdown(poker::Card p0_card, poker::Card p1_card) {
    return card_rank(p0_card) > card_rank(p1_card);
}

std::string card_name(poker::Card card) {
    switch (card) {
        case poker::Card::Jack:  return "J";
        case poker::Card::Queen: return "Q";
        case poker::Card::King:  return "K";
        case poker::Card::None:  return "-";
    }

    return "?";
}

std::string player_name(poker::Player player) {
    switch (player) {
        case poker::Player::Chance:   return "Chance";
        case poker::Player::P0:       return "P0";
        case poker::Player::P1:       return "P1";
        case poker::Player::Terminal: return "Terminal";
    }

    return "?";
}

std::string infoset_key(
    poker::Player player,
    poker::Card private_card,
    const std::string& history
) {
    return player_name(player) + "|" + card_name(private_card) + "|" + history;
}

poker::Card private_card_for(
    poker::Player player,
    poker::Card p0_card,
    poker::Card p1_card
) {
    if (player == poker::Player::P0) {
        return p0_card;
    }

    if (player == poker::Player::P1) {
        return p1_card;
    }

    throw std::runtime_error("No private card for non-real player.");
}

bool is_terminal_history(const std::string& history) {
    return history == "cc"  ||
           history == "bc"  ||
           history == "bf"  ||
           history == "cbc" ||
           history == "cbf";
}

std::vector<poker::Action> expected_legal_actions(
    const std::string& history
) {
    if (history.empty()) {
        return {poker::Action::Check, poker::Action::Bet};
    }

    if (history == "c") {
        return {poker::Action::Check, poker::Action::Bet};
    }

    if (history == "b") {
        return {poker::Action::Call, poker::Action::Fold};
    }

    if (history == "cb") {
        return {poker::Action::Call, poker::Action::Fold};
    }

    return {};
}

double expected_terminal_utility_p0(
    poker::Card p0_card,
    poker::Card p1_card,
    const std::string& history
) {
    const bool p0_wins = p0_wins_showdown(p0_card, p1_card);

    if (history == "cc") {
        return p0_wins ? +1.0 : -1.0;
    }

    if (history == "bc" || history == "cbc") {
        return p0_wins ? +2.0 : -2.0;
    }

    if (history == "bf") {
        return +1.0;
    }

    if (history == "cbf") {
        return -1.0;
    }

    throw std::runtime_error("Invalid terminal history.");
}

int count_terminal_nodes(const poker::Game& game) {
    int count = 0;

    for (const poker::Node& node : game.nodes) {
        if (node.terminal) {
            ++count;
        }
    }

    return count;
}

void test_root_and_chance_deals() {
    const poker::Game game = poker::build_kuhn_game();

    check(game.root == 0, "Root should have id 0.");
    check(game.num_nodes() > 0, "Game should contain nodes.");

    const poker::Node& root = game.node(game.root);

    check(root.player == poker::Player::Chance, "Root should be chance.");
    check(!root.terminal, "Root should not be terminal.");
    check(root.parent == -1, "Root should not have a parent.");
    check(root.children.size() == 6, "Root should have 6 ordered card deals.");

    double prob_sum = 0.0;
    std::set<std::string> deals;

    for (int child_id : root.children) {
        const poker::Node& deal = game.node(child_id);

        check(deal.parent == root.id, "Deal node parent should be root.");
        check(deal.depth == 1, "Deal node depth should be 1.");
        check(deal.p0_card != poker::Card::None, "P0 card should be dealt.");
        check(deal.p1_card != poker::Card::None, "P1 card should be dealt.");
        check(deal.p0_card != deal.p1_card, "Players should not receive same card.");

        check_near(
            deal.chance_prob,
            1.0 / 6.0,
            kTol,
            "Each chance deal should have probability 1/6."
        );

        prob_sum += deal.chance_prob;

        deals.insert(card_name(deal.p0_card) + "/" + card_name(deal.p1_card));
    }

    check_near(prob_sum, 1.0, kTol, "Chance deal probabilities should sum to 1.");

    check(deals.count("J/Q") == 1, "Missing deal J/Q.");
    check(deals.count("J/K") == 1, "Missing deal J/K.");
    check(deals.count("Q/J") == 1, "Missing deal Q/J.");
    check(deals.count("Q/K") == 1, "Missing deal Q/K.");
    check(deals.count("K/J") == 1, "Missing deal K/J.");
    check(deals.count("K/Q") == 1, "Missing deal K/Q.");

    std::cout << "[pass] test_root_and_chance_deals\n";
}

void test_expected_counts() {
    const poker::Game game = poker::build_kuhn_game();

    check(
        count_terminal_nodes(game) == 30,
        "Standard two-player Kuhn should have 30 terminal nodes."
    );

    check(
        game.num_infosets() == 12,
        "Compact two-player Kuhn should have 12 rational-player infosets."
    );

    check(
        game.num_q() == 24,
        "Compact two-player Kuhn should have 24 infoset-action entries."
    );

    std::cout << "[pass] test_expected_counts\n";
}

void test_parent_child_consistency() {
    const poker::Game game = poker::build_kuhn_game();

    for (const poker::Node& node : game.nodes) {
        check(node.id >= 0, "Every node should have nonnegative id.");

        if (node.id == game.root) {
            check(node.parent == -1, "Root should have parent -1.");
        } else {
            check(node.parent >= 0, "Non-root node should have parent.");
            check(node.parent < game.num_nodes(), "Parent id out of range.");

            const poker::Node& parent = game.node(node.parent);

            bool found = false;
            for (int child_id : parent.children) {
                if (child_id == node.id) {
                    found = true;
                    break;
                }
            }

            check(found, "Parent should contain child id.");
            check(node.depth == parent.depth + 1, "Child depth should be parent depth + 1.");
        }

        for (int child_id : node.children) {
            check(child_id >= 0, "Child id should be nonnegative.");
            check(child_id < game.num_nodes(), "Child id out of range.");
            check(game.node(child_id).parent == node.id, "Child parent pointer mismatch.");
        }
    }

    std::cout << "[pass] test_parent_child_consistency\n";
}

void test_decision_nodes_have_valid_infosets() {
    const poker::Game game = poker::build_kuhn_game();

    for (const poker::Node& node : game.nodes) {
        if (node.player != poker::Player::P0 &&
            node.player != poker::Player::P1) {
            continue;
        }

        check(!node.terminal, "Real-player node should not be terminal.");
        check(node.infoset >= 0, "Real-player node should have infoset.");
        check(node.infoset < game.num_infosets(), "Infoset id out of range.");

        const poker::InfoSet& infoset = game.infoset(node.infoset);

        check(infoset.player == node.player, "Infoset player should match node player.");
        check(infoset.private_card == private_card_for(node.player, node.p0_card, node.p1_card),
              "Infoset private card should match acting player's card.");
        check(infoset.public_history == node.history,
              "Infoset public history should match node history.");

        const std::vector<poker::Action> expected_actions =
            expected_legal_actions(node.history);

        check(
            infoset.actions == expected_actions,
            "Infoset actions should match legal Kuhn actions."
        );

        check(
            node.children.size() == infoset.actions.size(),
            "Decision node child count should match infoset action count."
        );

        check(
            infoset.q_indices.size() == infoset.actions.size(),
            "Infoset q_indices count should match action count."
        );

        for (std::size_t i = 0; i < node.children.size(); ++i) {
            const poker::Node& child = game.node(node.children[i]);
            check(
                child.incoming_action == infoset.actions[i],
                "Child incoming action should match infoset action order."
            );

            const int q = infoset.q_indices[i];

            check(q >= 0, "q index should be nonnegative.");
            check(q < game.num_q(), "q index out of range.");

            const poker::InfoSetAction& q_entry = game.q_entries[q];

            check(q_entry.q == q, "q entry id mismatch.");
            check(q_entry.infoset == infoset.id, "q entry infoset mismatch.");
            check(q_entry.local_action == static_cast<int>(i), "q local action mismatch.");
            check(q_entry.action == infoset.actions[i], "q action mismatch.");
        }
    }

    std::cout << "[pass] test_decision_nodes_have_valid_infosets\n";
}

void test_infoset_merging() {
    const poker::Game game = poker::build_kuhn_game();

    std::map<std::string, int> key_to_infoset;

    for (const poker::Node& node : game.nodes) {
        if (node.player != poker::Player::P0 &&
            node.player != poker::Player::P1) {
            continue;
        }

        const std::string key = infoset_key(
            node.player,
            private_card_for(node.player, node.p0_card, node.p1_card),
            node.history
        );

        auto it = key_to_infoset.find(key);

        if (it == key_to_infoset.end()) {
            key_to_infoset.emplace(key, node.infoset);
        } else {
            check(
                it->second == node.infoset,
                "Nodes with same player/card/history should share infoset."
            );
        }
    }

    check(
        static_cast<int>(key_to_infoset.size()) == game.num_infosets(),
        "Number of unique infoset keys should match game.num_infosets()."
    );

    check(key_to_infoset.count("P0|J|") == 1, "Missing P0 J initial infoset.");
    check(key_to_infoset.count("P0|Q|") == 1, "Missing P0 Q initial infoset.");
    check(key_to_infoset.count("P0|K|") == 1, "Missing P0 K initial infoset.");

    check(key_to_infoset.count("P1|J|c") == 1, "Missing P1 J after check infoset.");
    check(key_to_infoset.count("P1|Q|c") == 1, "Missing P1 Q after check infoset.");
    check(key_to_infoset.count("P1|K|c") == 1, "Missing P1 K after check infoset.");

    check(key_to_infoset.count("P1|J|b") == 1, "Missing P1 J after bet infoset.");
    check(key_to_infoset.count("P1|Q|b") == 1, "Missing P1 Q after bet infoset.");
    check(key_to_infoset.count("P1|K|b") == 1, "Missing P1 K after bet infoset.");

    check(key_to_infoset.count("P0|J|cb") == 1, "Missing P0 J after check-bet infoset.");
    check(key_to_infoset.count("P0|Q|cb") == 1, "Missing P0 Q after check-bet infoset.");
    check(key_to_infoset.count("P0|K|cb") == 1, "Missing P0 K after check-bet infoset.");

    std::cout << "[pass] test_infoset_merging\n";
}

void test_terminal_histories_and_utilities() {
    const poker::Game game = poker::build_kuhn_game();

    int terminal_count = 0;

    for (const poker::Node& node : game.nodes) {
        if (!node.terminal) {
            continue;
        }

        ++terminal_count;

        check(
            node.player == poker::Player::Terminal,
            "Terminal node should have Terminal player."
        );

        check(
            node.infoset == -1,
            "Terminal node should not have infoset."
        );

        check(
            node.children.empty(),
            "Terminal node should not have children."
        );

        check(
            is_terminal_history(node.history),
            "Terminal node has invalid terminal history: " + node.history
        );

        const double expected = expected_terminal_utility_p0(
            node.p0_card,
            node.p1_card,
            node.history
        );

        check_near(
            node.utility_p0,
            expected,
            kTol,
            "Terminal utility mismatch."
        );
    }

    check(terminal_count == 30, "Expected 30 terminal nodes.");

    std::cout << "[pass] test_terminal_histories_and_utilities\n";
}

void test_nonterminal_histories() {
    const poker::Game game = poker::build_kuhn_game();

    const std::set<std::string> allowed_nonterminal_histories = {
        "",
        "c",
        "b",
        "cb"
    };

    for (const poker::Node& node : game.nodes) {
        if (node.terminal || node.player == poker::Player::Chance) {
            continue;
        }

        check(
            allowed_nonterminal_histories.count(node.history) == 1,
            "Invalid nonterminal betting history: " + node.history
        );
    }

    std::cout << "[pass] test_nonterminal_histories\n";
}

void run_all_tests() {
    test_root_and_chance_deals();
    test_expected_counts();
    test_parent_child_consistency();
    test_decision_nodes_have_valid_infosets();
    test_infoset_merging();
    test_terminal_histories_and_utilities();
    test_nonterminal_histories();
}

} // namespace

int main() {
    try {
        run_all_tests();
    } catch (const std::exception& e) {
        std::cerr << "[fail] " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "[pass] all Kuhn tree tests passed\n";
    return EXIT_SUCCESS;
}