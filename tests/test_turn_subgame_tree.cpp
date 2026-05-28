#include "game.hpp"

#include "holdem/betting_abstraction.hpp"
#include "holdem/subgame_builder.hpp"
#include "holdem/subgame_config.hpp"
#include "holdem/street.hpp"

#include "poker/board.hpp"
#include "poker/range.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cfr_cpu.hpp"
#include "cfr_gpu.hpp"

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
    const double actual,
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

poker::Board make_test_turn_board() {
    return poker::Board{
            {
                phevaluator::Card("As"),
                phevaluator::Card("7h"),
                phevaluator::Card("Jh"),
                phevaluator::Card("Ts"),
            }
    };
}

poker::Range make_tiny_p0_range() {
    poker::Range range;
    range.clear();

    range.set_weight(
        poker::make_hand(
            phevaluator::Card("Kh"),
            phevaluator::Card("Qh")
        ),
        1.0f
    );

    range.set_weight(
        poker::make_hand(
            phevaluator::Card("Ks"),
            phevaluator::Card("Kd")
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
            phevaluator::Card("Qc"),
            phevaluator::Card("Qd")
        ),
        1.0f
    );

    range.set_weight(
        poker::make_hand(
            phevaluator::Card("Th"),
            phevaluator::Card("9h")
        ),
        1.0f
    );

    return range;
}

poker::holdem::HoldemSubgameConfig make_test_config() {
    poker::holdem::HoldemSubgameConfig config;

    config.start_street = poker::holdem::Street::Turn;
    config.board = make_test_turn_board();

    config.pot_size = 1000;
    config.effective_stack = 20000;
    config.player_to_act = poker::Player::P0;

    config.p0_range = make_tiny_p0_range();
    config.p1_range = make_tiny_p1_range();

    config.collapse_all_in_runouts_to_ev = true;
    config.validate_tree_during_build = false;

    config.betting_abstraction = poker::holdem::make_standard_abstraction();

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

bool has_action_type(const poker::InfoSet& infoset, poker::holdem::ActionType action_type) {
    for (const poker::GameAction action : infoset.actions) {
        if (action.action_type == static_cast<int>(action_type)) {
            return true;
        }
    }
    return false;
}
    using Clock = std::chrono::steady_clock;

    struct BenchResult {
        std::string name;
        double seconds = 0.0;
        double iters_per_sec = 0.0;
        std::vector<float> avg_strategy;
    };

    double max_abs_diff(
        const std::vector<float>& a,
        const std::vector<float>& b
    ) {
        if (a.size() != b.size()) {
            throw std::runtime_error("Strategy sizes differ.");
        }

        double diff = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            diff = std::max(diff, std::abs(
                static_cast<double>(a[i]) - static_cast<double>(b[i])
            ));
        }
        return diff;
    }
BenchResult run_cpu_benchmark(
        const poker::Game& game,
        int iterations
    ) {
    poker::CpuCfrSolver solver(game);
    const auto t0 = Clock::now();
    solver.run_iterations(iterations);
    const auto t1 = Clock::now();
    BenchResult r;
    r.name = "CPU";
    r.seconds = std::chrono::duration<double>(t1 - t0).count();
    r.iters_per_sec = iterations / r.seconds;
    r.avg_strategy = solver.average_strategy();
    return r;
}

BenchResult run_gpu_benchmark(
    const poker::Game& game,
    int iterations
) {
    poker::GpuCfrSolver solver(game);
    const auto t0 = Clock::now();
    solver.run_iterations(iterations);
    const auto t1 = Clock::now();

    BenchResult r;
    r.name = "GPU";
    r.seconds = std::chrono::duration<double>(t1 - t0).count();
    r.iters_per_sec = iterations / r.seconds;
    r.avg_strategy = solver.average_strategy();

    return r;
}

void print_result(const BenchResult& r) {
    std::cout << "[info] "
        << std::left << std::setw(8) << r.name
        << " time=" << std::setw(10) << r.seconds << "s"
        << " iter/s=" << std::setw(12) << r.iters_per_sec
        << "\n";
}
void test_turn_subgame_builds_nonempty_tree() {
    const poker::Game game = build_test_game();

    check(game.root == 0, "Expected root id to be 0.");
    check(game.num_nodes() > 0, "Turn subgame should contain nodes.");
    check(game.num_infosets() > 0, "Turn subgame should contain infosets.");
    check(game.num_q() > 0, "Turn subgame should contain q entries.");
    check(game.max_depth > 0, "Turn subgame should have positive depth.");

    check(
        count_terminal_nodes(game) > 0,
        "Turn subgame should contain terminal nodes."
    );

    check(
        count_nodes_with_player(game, poker::Player::Chance) > 1,
        "Turn subgame should contain private chance and river-card chance."
    );

    std::cout << "[pass] test_turn_subgame_builds_nonempty_tree\n";
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

void test_turn_subgame_contains_public_river_chance_nodes() {
    const poker::Game game = build_test_game();

    int non_root_chance_nodes = 0;

    for (const poker::Node& node : game.nodes) {
        if (node.id != game.root && node.player == poker::Player::Chance) {
            ++non_root_chance_nodes;

            check(
                !node.terminal,
                "Public river-card chance node should not be terminal."
            );

            check(
                node.infoset == -1,
                "Public river-card chance node should not have an infoset."
            );

            check(
                !node.children.empty(),
                "Public river-card chance node should have river-card outcomes."
            );
        }
    }

    check(
        non_root_chance_nodes > 0,
        "Turn subgame should contain public river-card chance nodes after betting round closes."
    );

    std::cout << "[pass] test_turn_subgame_contains_public_river_chance_nodes\n";
}

void test_chance_node_probabilities_sum_to_one() {
    const poker::Game game = build_test_game();

    for (const poker::Node& node : game.nodes) {
        if (node.player != poker::Player::Chance) {
            continue;
        }

        check(
            !node.children.empty(),
            "Chance node should have children."
        );

        double probability_sum = 0.0;

        for (int child_id : node.children) {
            const poker::Node& child = game.node(child_id);

            check(
                child.chance_prob > 0.0f,
                "Chance child should have positive probability."
            );

            check(
                child.chance_prob <= 1.0f,
                "Chance child probability should not exceed one."
            );

            probability_sum += static_cast<double>(child.chance_prob);
        }

        check_near(
            probability_sum,
            1.0,
            kProbTol,
            "Chance-node child probabilities should sum to 1."
        );
    }

    std::cout << "[pass] test_chance_node_probabilities_sum_to_one\n";
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
        const bool has_check = has_action_type(infoset, poker::holdem::ActionType::Check);

        const bool has_bet = has_action_type(infoset, poker::holdem::ActionType::Bet);

        const bool has_call = has_action_type(infoset, poker::holdem::ActionType::Call);

        const bool has_fold = has_action_type(infoset, poker::holdem::ActionType::Fold);

        if (has_check || has_bet) {
            check(
                has_check && has_bet,
                "Unopened turn/river infoset should have both check and bet."
            );

            check(
                !has_call && !has_fold,
                "Unopened turn/river infoset should not have call or fold."
            );

            saw_unopened_check_bet_infoset = true;
        }

        if (has_call || has_fold) {
            check(
                has_call && has_fold,
                "Facing-bet turn/river infoset should have both call and fold."
            );

            check(
                !has_check && !has_bet,
                "Facing-bet turn/river infoset should not have check or fresh bet."
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
        const std::string& key = infoset.key;

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

void test_turn_game_benchmark() {
    const poker::Game game = build_test_game();
    const int iterations = 100;
    std::cout << "[info] Testing " << iterations << " CFR iteration(s)\n";
    std::cout << "[info] Nodes: " << game.num_nodes()
              << " Infosets: " << game.num_infosets()
              << " Q: " << game.num_q() << "\n";

    BenchResult cpu = run_cpu_benchmark(game, iterations);
    print_result(cpu);

    BenchResult gpu =run_gpu_benchmark(game, iterations);
    print_result(gpu);

    std::cout << "[info] GPU speed relative to CPU: " << cpu.seconds / gpu.seconds << "x\n";
    std::cout << "[info] Max avg-strategy abs diff: " << max_abs_diff(cpu.avg_strategy, gpu.avg_strategy) << std::endl;
    std::cout << "[pass] test_turn_game_runs\n";
}

void run_all_tests() {
    test_turn_subgame_builds_nonempty_tree();
    test_root_is_private_hand_chance_node();
    test_turn_subgame_contains_public_river_chance_nodes();
    test_chance_node_probabilities_sum_to_one();
    test_parent_child_consistency();
    test_every_nonterminal_node_has_children();
    test_terminal_nodes_are_valid();
    test_decision_nodes_have_valid_infosets();
    test_infoset_action_sets_are_reasonable();
    test_q_entries_are_contiguous_by_infoset();
    test_infosets_merge_across_opponent_private_hands();
    test_infoset_keys_do_not_obviously_encode_opponent_hand();
    test_turn_game_benchmark();
}

} // namespace

int main() {
    try {
        run_all_tests();
    } catch (const std::exception& e) {
        std::cerr << "[fail] " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "[pass] all turn subgame tree tests passed\n";
    return EXIT_SUCCESS;
}