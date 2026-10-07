#pragma once

#include "test_support.hpp"
#include <set>

namespace test_support {

inline void check_subgame_tree(int board_cards) {
    auto config = tiny_config(board_cards);
    // A pot-sized bet is all-in, keeping exact flop runouts bounded.
    config.effective_stack = config.pot_size;
    const poker::Game game = poker::holdem::HoldemSubgameBuilder(config).build();
    game.validate();
    check(game.root == 0 && game.num_nodes() > 1 && game.max_depth > 0,
          "Subgame must build a nonempty public tree.");
    check(game.node(game.root).type == poker::PublicNodeType::Action &&
              game.node(game.root).player == config.player_to_act &&
              game.node(game.root).parent == -1,
          "Root must be the first public decision; private deals live in side tables.");
    check(game.p0_hands.hand_count() == 2 && game.p1_hands.hand_count() == 2 &&
              game.hand_pairs.pair_count() == 4,
          "Expected four legal pairs in the two-by-two fixture.");
    for (int pair = 0; pair < game.hand_pairs.pair_count(); ++pair) {
        const auto p0 = game.p0_hands.hands.at(game.hand_pairs.p0_index.at(pair));
        const auto p1 = game.p1_hands.hands.at(game.hand_pairs.p1_index.at(pair));
        check(!poker::hands_overlap(p0, p1), "Private pair must not overlap.");
        check(!poker::hand_overlaps_mask(p0, poker::board_mask(config.board)) &&
                  !poker::hand_overlaps_mask(p1, poker::board_mask(config.board)),
              "Private domains must respect starting board blockers.");
    }

    std::vector<int> incoming(game.num_nodes(), 0);
    int terminals = 0;
    int chances = 0;
    int action_states = 0;
    bool saw_check_bet = false;
    bool saw_fold_call = false;
    std::set<int> chance_board_sizes;
    for (int id = 0; id < game.num_nodes(); ++id) {
        const auto& node = game.node(id);
        check(node.depth >= 0 && node.depth <= game.max_depth, "Invalid node depth.");
        if (node.type == poker::PublicNodeType::Terminal) {
            ++terminals;
            check(node.edge_count == 0 && node.player == poker::Player::Terminal &&
                      node.action_state_index == -1,
                  "Terminal must have no children or action state.");
            continue;
        }
        check(node.edge_count > 0, "Every nonterminal needs children.");
        for (int local = 0; local < node.edge_count; ++local) {
            const auto& edge = game.edge(node.first_edge + local);
            check(edge.child >= 0 && edge.child < game.num_nodes(), "Child out of range.");
            const auto& child = game.node(edge.child);
            check(child.parent == id && child.depth == node.depth + 1,
                  "Parent/child links or depths disagree.");
            ++incoming.at(edge.child);
        }
        if (node.type == poker::PublicNodeType::Chance) {
            ++chances;
            check(node.player == poker::Player::Chance && node.action_state_index == -1,
                  "Public chance must not have an action state.");
            // Count cards dealt along the path, since PublicNode contains no Board.
            auto board = config.board;
            std::vector<int> path_cards;
            for (int child = id; game.node(child).parent >= 0;) {
                const int parent = game.node(child).parent;
                const auto& p = game.node(parent);
                if (p.type == poker::PublicNodeType::Chance)
                    for (int e = 0; e < p.edge_count; ++e)
                        if (game.edge(p.first_edge + e).child == child)
                            path_cards.push_back(game.edge(p.first_edge + e).public_card);
                child = parent;
            }
            for (auto card = path_cards.rbegin(); card != path_cards.rend(); ++card)
                board = board.with_added_card(phevaluator::Card(*card));
            chance_board_sizes.insert(static_cast<int>(board.size()));
            check(node.edge_count == 52 - static_cast<int>(board.size()),
                  "Exact public chance must enumerate every board-unblocked card.");
            double sum = 0.0;
            std::set<int> dealt;
            for (int e = 0; e < node.edge_count; ++e) {
                const auto& edge = game.edge(node.first_edge + e);
                check(edge.local_action == -1 && edge.action_index == -1 &&
                          edge.public_card >= 0 && edge.public_card < 52,
                      "Chance edge metadata is invalid.");
                check(!poker::contains_card(poker::board_mask(board), phevaluator::Card(edge.public_card)),
                      "Public chance cannot deal a board card again.");
                check(dealt.insert(edge.public_card).second, "Duplicate public chance card.");
                check_near(edge.chance_prob, 1.0 / node.edge_count, 1e-6,
                           "Exact public chance probabilities must be uniform.");
                sum += edge.chance_prob;
            }
            check_near(sum, 1.0, 1e-6, "Chance probabilities must sum to one.");
        } else {
            ++action_states;
            const auto& state = game.action_state(node.action_state_index);
            check(state.node == id && state.player == static_cast<int>(node.player) &&
                      state.action_count == node.edge_count &&
                      state.bucket_count == game.hand_domain(node.player).hand_count(),
                  "Action state must describe its public node and own hand domain.");
            for (int local = 0; local < node.edge_count; ++local) {
                const auto& edge = game.edge(node.first_edge + local);
                check(edge.local_action == local && edge.action_index == state.first_action + local &&
                          edge.public_card == -1 && edge.chance_prob == 1.0f,
                      "Action edges must match their contiguous action block.");
            }
            const bool check_action = local_action(game, state, poker::holdem::ActionType::Check) >= 0;
            const bool bet = local_action(game, state, poker::holdem::ActionType::Bet) >= 0 ||
                             local_action(game, state, poker::holdem::ActionType::AllIn) >= 0;
            const bool fold = local_action(game, state, poker::holdem::ActionType::Fold) >= 0;
            const bool call = local_action(game, state, poker::holdem::ActionType::Call) >= 0;
            check((check_action && bet && !fold && !call) ||
                      (!check_action && !bet && fold && call),
                  "Tiny betting abstraction must offer check/bet or fold/call.");
            saw_check_bet |= check_action && bet;
            saw_fold_call |= fold && call;
        }
    }
    check(incoming.at(game.root) == 0, "Root must have no incoming edges.");
    for (int id = 0; id < game.num_nodes(); ++id)
        if (id != game.root) check(incoming.at(id) == 1, "Each public node needs exactly one parent.");
    check(terminals > 0 && action_states == game.num_action_states() &&
              saw_check_bet && saw_fold_call, "Missing terminal or betting branches.");
    check((board_cards == 5 && chances == 0) || (board_cards < 5 && chances > 0),
          "Public chance nodes must match the starting street.");
    for (int size = board_cards; size < 5; ++size)
        check(chance_board_sizes.contains(size), "Missing public street transition.");

    std::size_t tensor_end = 0;
    std::size_t bucket_end = 0;
    for (const auto& state : game.action_states) {
        check(state.tensor_offset == tensor_end && state.state_bucket_offset == bucket_end,
              "Action-state tensors must be contiguous and disjoint.");
        tensor_end += static_cast<std::size_t>(state.bucket_count) * state.action_count;
        bucket_end += state.bucket_count;
    }
    check(tensor_end == game.cfr_tensor_entries() && bucket_end == game.state_bucket_entries(),
          "Tensor layout totals disagree.");
    check(game.terminal_value_p0.size() == static_cast<std::size_t>(terminals) * game.hand_pairs.pair_count(),
          "Terminal table must have one value per terminal and legal pair.");
    bool saw_nonzero = false;
    for (float value : game.terminal_value_p0) {
        check(std::isfinite(value), "Terminal utility must be finite.");
        saw_nonzero |= value != 0.0f;
    }
    check(saw_nonzero, "Terminal fixture must provide a learning signal.");
    std::cout << "[pass] public tree, chance, private domains, actions, tensors and terminals ("
              << board_cards << " board cards)\n";
}

} // namespace test_support
