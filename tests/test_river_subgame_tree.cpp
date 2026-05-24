#include "game.hpp"

#include "holdem/betting_abstraction.hpp"
#include "holdem/subgame_builder.hpp"
#include "holdem/subgame_config.hpp"
#include "holdem/street.hpp"

#include "poker/board.hpp"
#include "poker/card.hpp"
#include "poker/range.hpp"

#include <algorithm>
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

constexpr double kProbTol = 1e-6;

void check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void check_eq(
    int actual,
    int expected,
    const std::string& message
) {
    if (actual != expected) {
        std::ostringstream oss;
        oss << message
            << " actual=" << actual
            << " expected=" << expected;
        throw std::runtime_error(oss.str());
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

poker::CardId c(poker::Rank rank, poker::Suit suit) {
    return poker::make_card(rank, suit);
}

poker::Board make_test_river_board() {
    return poker::Board{
        {
            c(poker::Rank::Ace, poker::Suit::Spades),
            c(poker::Rank::Seven, poker::Suit::Hearts),
            c(poker::Rank::Two, poker::Suit::Clubs),
            c(poker::Rank::Jack, poker::Suit::Diamonds),
            c(poker::Rank::Four, poker::Suit::Spades)
        }
    };
}

poker::Range make_tiny_p0_range() {
    poker::Range range;
    range.clear();

    range.set_weight(
        poker::make_hand(
            c(poker::Rank::King, poker::Suit::Hearts),
            c(poker::Rank::Queen, poker::Suit::Hearts)
        ),
        1.0f
    );

    range.set_weight(
        poker::make_hand(
            c(poker::Rank::King, poker::Suit::Spades),
            c(poker::Rank::King, poker::Suit::Diamonds)
        ),
        1.0f
    );

    return range;
}

poker::Range make_tiny_p1_range() {
    poker::Range range;
    range.clear();

    range.set_weight(
        poker::make_hand(
            c(poker::Rank::Queen, poker::Suit::Clubs),
            c(poker::Rank::Queen, poker::Suit::Diamonds)
        ),
        1.0f
    );

    range.set_weight(
        poker::make_hand(
            c(poker::Rank::Ten, poker::Suit::Hearts),
            c(poker::Rank::Nine, poker::Suit::Hearts)
        ),
        1.0f
    );

    return range;
}

poker::holdem::BettingAbstraction make_tiny_betting_abstraction() {
    poker::holdem::BettingAbstraction abstraction;

    // Tiny structural tree:
    //
    // Unopened:
    //   check
    //   bet pot
    //
    // Facing bet:
    //   fold
    //   call
    //
    // No raises.
    abstraction.first_bet_sizes = {
        poker::holdem::BetSize::pot_fraction(1.0)
    };

    abstraction.raise_sizes = {};
    abstraction.max_raises_per_street = 0;

    return abstraction;
}

poker::holdem::HoldemSubgameConfig make_test_config() {
    poker::holdem::HoldemSubgameConfig config;

    config.start_street = poker::holdem::Street::River;
    config.board = make_test_river_board();

    config.pot_size = 1000;
    config.effective_stack = 2000;
    config.player_to_act = poker::Player::P0;

    config.p0_range = make_tiny_p0_range();
    config.p1_range = make_tiny_p1_range();

    config.betting_abstraction = make_tiny_betting_abstraction();

    config.use_card_abstraction = false;
    config.use_action_abstraction = true;

    return config;
}

poker::Game build_test_game() {
    const poker::holdem::HoldemSubgameConfig config = make_test_config();
    return poker::holdem::HoldemSubgameBuilder(config).build();
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

int count_nodes_with_player(
    const poker::Game& game,
    poker::Player player
) {
    int count = 0;

    for (const poker::Node& node : game.nodes) {
        if (node.player == player) {
            ++count;
        }
    }

    return count;
}

bool has_action_type(
    const poker::InfoSet& infoset,
    poker::Action action
) {
    return std::find(
        infoset.actions.begin(),
        infoset.actions.end(),
        action
    ) != infoset.actions.end();
}

void test_river_subgame_builds_nonempty_tree() {
    const poker::Game game = build_test_game();

    check(game.root == 0, "Expected root id to be 0.");
    check(game.num_nodes() > 0, "River subgame should contain nodes.");
    check(game.num_infosets() > 0, "River subgame should contain infosets.");
    check(game.num_q() > 0, "River subgame should contain q entries.");
    check(game.max_depth > 0, "River subgame should have positive depth.");

    check(
        count_terminal_nodes(game) > 0,
        "River subgame should contain terminal nodes."
    );

    std::cout << "[pass] test_river_subgame_builds_nonempty_tree\n";
}

void test_root_is_private_hand_chance_node() {
    const poker::Game game = build_test_game();

    const poker::Node& root = game.node(game.root);

    check(root.player == poker::Player::Chance, "Root should be chance.");
    check(!root.terminal, "Root should not be terminal.");
    check(root.parent == -1, "Root should not have a parent.");
    check(root.infoset == -1, "Root should not have an infoset.");
    check(!root.children.empty(), "Root should have private hand-pair outcomes.");

    double probability_sum = 0.0;

    for (int child_id : root.children) {
        const poker::Node& child = game.node(child_id);

        check(child.parent == root.id, "Root child should point back to root.");
        check(child.depth == 1, "Root children should have depth 1.");
        check(
            child.chance_prob > 0.0f,
            "Private hand-pair child should have positive probability."
        );
        check(
            child.chance_prob <= 1.0f,
            "Private hand-pair probability should not exceed one."
        );

        probability_sum += static_cast<double>(child.chance_prob);
    }

    check_near(
        probability_sum,
        1.0,
        kProbTol,
        "Root private hand-pair chance probabilities should sum to 1."
    );

    std::cout << "[pass] test_root_is_private_hand_chance_node\n";
}

void test_river_subgame_has_no_public_card_chance_after_root() {
    const poker::Game game = build_test_game();

    int non_root_chance_nodes = 0;

    for (const poker::Node& node : game.nodes) {
        if (node.id != game.root && node.player == poker::Player::Chance) {
            ++non_root_chance_nodes;
        }
    }

    check_eq(
        non_root_chance_nodes,
        0,
        "River subgame should not contain public-card chance nodes after root."
    );

    check_eq(
        count_nodes_with_player(game, poker::Player::Chance),
        1,
        "River subgame should only have the private-hand root chance node."
    );

    std::cout << "[pass] test_river_subgame_has_no_public_card_chance_after_root\n";
}

void test_parent_child_consistency() {
    const poker::Game game = build_test_game();

    for (const poker::Node& node : game.nodes) {
        check(node.id >= 0, "Every node should have nonnegative id.");
        check(node.id < game.num_nodes(), "Every node id should be in range.");

        if (node.id == game.root) {
            check(node.parent == -1, "Root should have parent -1.");
        } else {
            check(node.parent >= 0, "Non-root node should have parent.");
            check(node.parent < game.num_nodes(), "Parent id out of range.");

            const poker::Node& parent = game.node(node.parent);

            const bool found =
                std::find(
                    parent.children.begin(),
                    parent.children.end(),
                    node.id
                ) != parent.children.end();

            check(found, "Parent should contain child id.");

            check(
                node.depth == parent.depth + 1,
                "Child depth should equal parent depth + 1."
            );
        }

        for (int child_id : node.children) {
            check(child_id >= 0, "Child id should be nonnegative.");
            check(child_id < game.num_nodes(), "Child id out of range.");

            const poker::Node& child = game.node(child_id);

            check(
                child.parent == node.id,
                "Child parent pointer should point back to parent."
            );
        }
    }

    std::cout << "[pass] test_parent_child_consistency\n";
}

void test_every_nonterminal_node_has_children() {
    const poker::Game game = build_test_game();

    for (const poker::Node& node : game.nodes) {
        if (node.terminal) {
            continue;
        }

        check(
            !node.children.empty(),
            "Every nonterminal node should have at least one child."
        );
    }

    std::cout << "[pass] test_every_nonterminal_node_has_children\n";
}

void test_terminal_nodes_are_valid() {
    const poker::Game game = build_test_game();

    bool saw_nonzero_terminal_value = false;

    for (const poker::Node& node : game.nodes) {
        if (!node.terminal) {
            continue;
        }

        check(
            node.children.empty(),
            "Terminal node should have no children."
        );

        check(
            node.player == poker::Player::Terminal,
            "Terminal node should have Terminal player."
        );

        check(
            node.infoset == -1,
            "Terminal node should not have infoset."
        );

        if (std::abs(static_cast<double>(node.utility_p0)) > 0.0) {
            saw_nonzero_terminal_value = true;
        }
    }

    check(
        saw_nonzero_terminal_value,
        "Expected at least one terminal node with nonzero P0 utility."
    );

    std::cout << "[pass] test_terminal_nodes_are_valid\n";
}

void test_decision_nodes_have_valid_infosets() {
    const poker::Game game = build_test_game();

    for (const poker::Node& node : game.nodes) {
        if (node.player != poker::Player::P0 &&
            node.player != poker::Player::P1) {
            continue;
        }

        check(!node.terminal, "Decision node should not be terminal.");
        check(node.infoset >= 0, "Decision node should have infoset.");
        check(node.infoset < game.num_infosets(), "Infoset id out of range.");

        const poker::InfoSet& infoset = game.infoset(node.infoset);

        check(
            infoset.player == node.player,
            "Infoset player should match decision-node player."
        );

        check(
            !infoset.actions.empty(),
            "Infoset should have legal actions."
        );

        check(
            node.children.size() == infoset.actions.size(),
            "Decision-node children should match infoset action count."
        );

        check(
            infoset.q_indices.size() == infoset.actions.size(),
            "Infoset q_indices should match action count."
        );

        for (std::size_t i = 0; i < infoset.q_indices.size(); ++i) {
            const int q = infoset.q_indices[i];

            check(q >= 0, "q index should be nonnegative.");
            check(q < game.num_q(), "q index should be in range.");

            const poker::InfoSetAction& q_entry = game.q_entries[q];

            check(q_entry.q == q, "q entry id mismatch.");
            check(q_entry.infoset == infoset.id, "q entry infoset mismatch.");

            check(
                q_entry.local_action == static_cast<int>(i),
                "q entry local action mismatch."
            );

            check(
                q_entry.action == infoset.actions[i],
                "q entry action should match infoset action."
            );
        }
    }

    std::cout << "[pass] test_decision_nodes_have_valid_infosets\n";
}

void test_infoset_action_sets_are_reasonable() {
    const poker::Game game = build_test_game();

    bool saw_unopened_check_bet_infoset = false;
    bool saw_facing_bet_infoset = false;

    for (const poker::InfoSet& infoset : game.infosets) {
        const bool has_check =
            has_action_type(infoset, poker::Action::Check);

        const bool has_bet =
            has_action_type(infoset, poker::Action::Bet);

        const bool has_call =
            has_action_type(infoset, poker::Action::Call);

        const bool has_fold =
            has_action_type(infoset, poker::Action::Fold);

        if (has_check || has_bet) {
            check(
                has_check && has_bet,
                "Unopened river infoset should have both check and bet."
            );

            check(
                !has_call && !has_fold,
                "Unopened river infoset should not have call or fold."
            );

            saw_unopened_check_bet_infoset = true;
        }

        if (has_call || has_fold) {
            check(
                has_call && has_fold,
                "Facing-bet river infoset should have both call and fold."
            );

            check(
                !has_check && !has_bet,
                "Facing-bet river infoset should not have check or fresh bet."
            );

            saw_facing_bet_infoset = true;
        }
    }

    check(
        saw_unopened_check_bet_infoset,
        "Expected at least one unopened check/bet infoset."
    );

    check(
        saw_facing_bet_infoset,
        "Expected at least one facing-bet fold/call infoset."
    );

    std::cout << "[pass] test_infoset_action_sets_are_reasonable\n";
}

void test_q_entries_are_contiguous_by_infoset() {
    const poker::Game game = build_test_game();

    for (const poker::InfoSet& infoset : game.infosets) {
        check(
            !infoset.q_indices.empty(),
            "Infoset should have q indices."
        );

        const int q_begin = infoset.q_indices.front();

        for (int local = 0;
             local < static_cast<int>(infoset.q_indices.size());
             ++local) {
            const int q = infoset.q_indices[local];

            check_eq(
                q,
                q_begin + local,
                "q indices should be contiguous within each infoset."
            );

            const poker::InfoSetAction& q_entry = game.q_entries[q];

            check_eq(
                q_entry.infoset,
                infoset.id,
                "q entry should point back to owning infoset."
            );

            check_eq(
                q_entry.local_action,
                local,
                "q entry local action should match position in infoset."
            );
        }
    }

    std::cout << "[pass] test_q_entries_are_contiguous_by_infoset\n";
}

void test_infosets_merge_across_opponent_private_hands() {
    const poker::Game game = build_test_game();

    // This assumes your Hold'em builder stores canonical infoset key material
    // in InfoSet::public_history, similar to the current Kuhn implementation.
    //
    // The key property:
    //   For the same acting player, own hand/bucket, board, stack state, and
    //   public betting history, changing only the opponent hand should not
    //   create a separate infoset.
    //
    // This smoke test checks that at least one key appears at multiple nodes,
    // which should happen in this 2-combo-vs-2-combo toy game.

    std::map<int, int> node_count_by_infoset;

    for (const poker::Node& node : game.nodes) {
        if (node.player != poker::Player::P0 &&
            node.player != poker::Player::P1) {
            continue;
        }

        ++node_count_by_infoset[node.infoset];
    }

    bool saw_infoset_with_multiple_nodes = false;

    for (const auto& [infoset_id, count] : node_count_by_infoset) {
        if (count > 1) {
            saw_infoset_with_multiple_nodes = true;
            break;
        }
    }

    check(
        saw_infoset_with_multiple_nodes,
        "Expected at least one infoset to merge multiple private states."
    );

    std::cout << "[pass] test_infosets_merge_across_opponent_private_hands\n";
}

void test_infoset_keys_do_not_obviously_encode_opponent_hand() {
    const poker::Game game = build_test_game();

    for (const poker::InfoSet& infoset : game.infosets) {
        const std::string& key = infoset.public_history;

        check(
            key.find("opponent") == std::string::npos,
            "Infoset key should not contain literal 'opponent'."
        );

        check(
            key.find("p0_hand") == std::string::npos ||
            infoset.player == poker::Player::P0,
            "P1 infoset key should not expose p0_hand."
        );

        check(
            key.find("p1_hand") == std::string::npos ||
            infoset.player == poker::Player::P1,
            "P0 infoset key should not expose p1_hand."
        );
    }

    std::cout << "[pass] test_infoset_keys_do_not_obviously_encode_opponent_hand\n";
}

void run_all_tests() {
    test_river_subgame_builds_nonempty_tree();
    test_root_is_private_hand_chance_node();
    test_river_subgame_has_no_public_card_chance_after_root();
    test_parent_child_consistency();
    test_every_nonterminal_node_has_children();
    test_terminal_nodes_are_valid();
    test_decision_nodes_have_valid_infosets();
    test_infoset_action_sets_are_reasonable();
    test_q_entries_are_contiguous_by_infoset();
    test_infosets_merge_across_opponent_private_hands();
    test_infoset_keys_do_not_obviously_encode_opponent_hand();
}

} // namespace

int main() {
    try {
        run_all_tests();
    } catch (const std::exception& e) {
        std::cerr << "[fail] " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "[pass] all river subgame tree tests passed\n";
    return EXIT_SUCCESS;
}