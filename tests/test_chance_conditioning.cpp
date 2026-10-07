#include "test_support.hpp"
#include "cfr_cpu.hpp"
#include "cfr_gpu.hpp"
#include "chance_probability.hpp"
#include <array>
#include <phevaluator/phevaluator.h>

using namespace test_support;

namespace {

poker::holdem::HoldemSubgameConfig royal_config(int board_size) {
    auto config = tiny_config(board_size);
    config.board = poker::Board{{phevaluator::Card("Qh"), phevaluator::Card("Jh"), phevaluator::Card("Th")}};
    if (board_size == 4) config.board.cards.push_back(phevaluator::Card("2c"));
    config.p0_range.clear();
    config.p1_range.clear();
    config.p0_range.set_weight(poker::make_hand(phevaluator::Card("Ah"), phevaluator::Card("Kh")), 1.0f);
    config.p1_range.set_weight(poker::make_hand(phevaluator::Card("Qc"), phevaluator::Card("Qd")), 1.0f);
    config.betting_abstraction.first_bet_sizes.clear();
    config.terminal_mode = poker::TerminalMode::RecordComputed;
    return config;
}

void test_certain_win_retains_probability_mass() {
    for (int board_size : {4, 3}) {
        const auto config = royal_config(board_size);
        const auto game = poker::holdem::HoldemSubgameBuilder(config).build();
        poker::TerminalValueProvider terminals;
        poker::PublicExploitabilityEvaluator evaluator(game, terminals);
        const auto strategy = uniform_strategy(game);
        check_near(evaluator.expected_value_p0(strategy), 1000.0, 1e-4,
                   "A royal flush must retain the full pot over legal runouts.");
        check_near(evaluator.best_response(strategy, poker::Player::P0).value, 1000.0, 1e-4,
                   "Best response must preserve private-conditioned chance mass.");
        poker::CpuCfrSolver cpu(game, terminals);
        cpu.run_one_iteration();
        check_near(cpu.stats().last_root_value_p0, 1000.0, 1e-4,
                   "CPU chance traversal must preserve the full pot.");
        poker::GpuCfrConfig gpu_config;
        gpu_config.evaluator_data_dir = UTG_TEST_EVALUATOR_DIR;
        gpu_config.synchronize_each_iteration = true;
        poker::GpuCfrSolver gpu(game, gpu_config);
        gpu.run_one_iteration();
        check_near(evaluator.expected_value_p0(gpu.average_strategy()), 1000.0, 1e-4,
                   "GPU check-only policy must preserve the full pot.");
        std::cout << "[pass] certain-win EV over legal runouts (" << board_size << " board cards)\n";
    }
}

// Enumerate final boards directly, independently of public-tree edge weights.
double enumerated_equity(const poker::Game& game) {
    double total = 0.0;
    for (int pair = 0; pair < game.hand_pairs.pair_count(); ++pair) {
        const auto p0 = poker::hand_from_id(game.p0_hands.hands[game.hand_pairs.p0_index[pair]]);
        const auto p1 = poker::hand_from_id(game.p1_hands.hands[game.hand_pairs.p1_index[pair]]);
        const auto dead = poker::board_mask(game.starting_board) | p0.mask() | p1.mask();
        double equity = 0.0;
        int count = 0;
        for (int a = 0; a < 52; ++a) {
            if (dead & (poker::DeckMask{1} << a)) continue;
            for (int b = 0; b < (game.starting_board.size() == 3 ? a : 1); ++b) {
                if (game.starting_board.size() == 3 && (dead & (poker::DeckMask{1} << b))) continue;
                auto board = game.starting_board.cards;
                board.push_back(phevaluator::Card(a));
                if (board.size() == 4) board.push_back(phevaluator::Card(b));
                const auto r0 = phevaluator::EvaluateCards(p0.a, p0.b, board[0], board[1], board[2], board[3], board[4]);
                const auto r1 = phevaluator::EvaluateCards(p1.a, p1.b, board[0], board[1], board[2], board[3], board[4]);
                equity += r0 > r1 ? 1.0 : (r0 == r1 ? 0.5 : 0.0);
                ++count;
            }
        }
        check(count == (game.starting_board.size() == 3 ? 990 : 44), "Enumerate every legal final board.");
        total += equity / count;
    }
    return total / game.hand_pairs.pair_count();
}

void test_enumerated_runouts() {
    for (int board_size : {4, 3}) {
        auto config = tiny_config(board_size);
        config.betting_abstraction.first_bet_sizes.clear();
        config.terminal_mode = poker::TerminalMode::RecordComputed;
        const auto game = poker::holdem::HoldemSubgameBuilder(config).build();
        const double expected = 1000.0 * enumerated_equity(game);
        check(expected > 0.0 && expected < 1000.0, "Fixture must contain wins and losses.");
        poker::TerminalValueProvider terminals;
        poker::PublicExploitabilityEvaluator evaluator(game, terminals);
        check_near(evaluator.expected_value_p0(uniform_strategy(game)), expected, 1e-4,
                   "Tree EV must equal direct enumeration of legal boards.");
        poker::CpuCfrSolver cpu(game, terminals);
        cpu.run_one_iteration();
        check_near(cpu.stats().last_root_value_p0, expected, 1e-4, "CPU EV must equal enumerated equity.");
        for (int chunk : {1, 3}) {
            poker::GpuCfrConfig gpu_config;
            gpu_config.evaluator_data_dir = UTG_TEST_EVALUATOR_DIR;
            gpu_config.pair_chunk_size = chunk;
            poker::GpuCfrSolver gpu(game, gpu_config);
            gpu.run_one_iteration();
            check_near(evaluator.expected_value_p0(gpu.average_strategy()), expected, 1e-4,
                       "Chunked GPU check-only policy must equal enumerated equity.");
        }
        std::cout << "[pass] enumerated " << board_size << "-card EV=" << expected << '\n';
    }
}

void test_weighted_chance_reach() {
    poker::Game game;
    game.starting_board = tiny_config().board;
    game.set_hand_domains(
        poker::HandDomain{{poker::make_hand(phevaluator::Card("Kh"), phevaluator::Card("Qh")),
                           poker::make_hand(phevaluator::Card("Ks"), phevaluator::Card("Kd"))}},
        poker::HandDomain{{poker::make_hand(phevaluator::Card("Qc"), phevaluator::Card("Qd"))}},
        poker::HandPairTable{{0, 1}, {0, 0}});
    poker::PublicNode chance;
    chance.type = poker::PublicNodeType::Chance;
    chance.player = poker::Player::Chance;
    game.root = game.add_node(chance);
    const std::array<phevaluator::Card, 3> cards = {phevaluator::Card("Kh"), phevaluator::Card("2c"), phevaluator::Card("2d")};
    const std::array<float, 3> probabilities = {0.2f, 0.3f, 0.5f};
    const std::array<int, 3> payoffs = {1000, 100, 200};
    std::array<int, 3> decisions;
    for (int i = 0; i < 3; ++i) {
        poker::PublicNode node;
        node.type = poker::PublicNodeType::Action;
        node.player = poker::Player::P0;
        decisions[i] = game.add_node(node);
        game.add_chance_child(game.root, decisions[i], cards[i], probabilities[i]);
    }
    for (int i = 0; i < 3; ++i) {
        for (int a = 0; a < 2; ++a) {
            const int leaf = game.add_node(poker::PublicNode{});
            game.add_action_child(decisions[i], leaf, game.add_action(poker::GameAction{a, 0}));
            game.terminal_records.push_back({poker::TerminalType::P1_Fold, 0, a == 0 ? payoffs[i] : 0, 0});
        }
        game.register_action_state(decisions[i], poker::Player::P0);
    }
    game.validate();
    poker::TerminalValueProvider terminals;
    poker::PublicExploitabilityEvaluator evaluator(game, terminals);
    check_near(evaluator.expected_value_p0(uniform_strategy(game)), 123.125, 1e-4,
               "Nonuniform chance must preserve relative weights after blocking.");
    check_near(evaluator.best_response(uniform_strategy(game), poker::Player::P0).value, 246.25, 1e-4,
               "Best response must use pair-conditioned chance values.");
    poker::CpuCfrSolver cpu(game, terminals);
    cpu.run_one_iteration();
    for (int i = 0; i < 3; ++i) {
        const auto& state = game.action_state(game.node(decisions[i]).action_state_index);
        const double conditional = i == 0 ? 0.0 : probabilities[i] / 0.8;
        check_near(cpu.strategy_weight_sum()[state.state_bucket_index(0)], conditional / 2.0, 1e-6,
                   "Average weight must include chance and exclude blocked pairs.");
        check_near(cpu.regret_sum()[state.tensor_index(0, 0)], conditional * payoffs[i] / 4.0, 1e-4,
                   "Regret must include conditioned counterfactual reach.");
    }
    const auto first_regrets = cpu.regret_sum();
    cpu.run_iterations(9);
    for (int chunk : {1, 2}) {
        poker::GpuCfrConfig gpu_config;
        gpu_config.evaluator_data_dir = UTG_TEST_EVALUATOR_DIR;
        gpu_config.pair_chunk_size = chunk;
        poker::GpuCfrSolver gpu(game, gpu_config);
        gpu.run_one_iteration();
        const auto regrets = gpu.regret_sum();
        // GPU accumulates unnormalized pair weights; CPU uses pair probabilities.
        // This constant scale cancels during regret matching.
        for (std::size_t i = 0; i < regrets.size(); ++i)
            check_near(regrets[i] / game.hand_pairs.pair_count(), first_regrets[i], 1e-4, "Weighted CPU/GPU regret mismatch.");
        gpu.run_iterations(9);
        const auto cpu_avg = cpu.average_strategy(), gpu_avg = gpu.average_strategy();
        for (std::size_t i = 0; i < cpu_avg.size(); ++i)
            check_near(gpu_avg[i], cpu_avg[i], 1e-5, "Blocked and legal average strategies must agree.");
    }
    std::cout << "[pass] weighted chance values, regrets and average strategies\n";
}

void test_street_learning_parity() {
    for (int board_size : {4, 3}) {
        auto config = tiny_config(board_size);
        config.effective_stack = config.pot_size;
        config.terminal_mode = poker::TerminalMode::RecordComputed;
        const auto game = poker::holdem::HoldemSubgameBuilder(config).build();
        poker::TerminalValueProvider terminals;
        poker::CpuCfrSolver cpu(game, terminals);
        cpu.run_one_iteration();
        const auto first_regrets = cpu.regret_sum();
        cpu.run_iterations(19);
        const auto cpu_avg = cpu.average_strategy();
        for (int chunk : {1, 3}) {
            poker::GpuCfrConfig gpu_config;
            gpu_config.evaluator_data_dir = UTG_TEST_EVALUATOR_DIR;
            gpu_config.pair_chunk_size = chunk;
            poker::GpuCfrSolver gpu(game, gpu_config);
            gpu.run_one_iteration();
            const auto regrets = gpu.regret_sum();
            double maximum_regret = 0.0;
            for (std::size_t i = 0; i < regrets.size(); ++i) {
                check_near(regrets[i] / game.hand_pairs.pair_count(), first_regrets[i], 0.005,
                           "Turn/flop CPU/GPU learning signals must agree.");
                maximum_regret = std::max(maximum_regret, std::abs(static_cast<double>(regrets[i])));
            }
            check(maximum_regret > 1.0, "Betting fixture must exercise GPU value and reach passes.");
            gpu.run_iterations(19);
            const auto gpu_avg = gpu.average_strategy();
            for (std::size_t i = 0; i < gpu_avg.size(); ++i)
                check_near(gpu_avg[i], cpu_avg[i], 1e-3, "Turn/flop learned average strategies must agree.");
        }
        std::cout << "[pass] " << board_size << "-card CPU/GPU learning across pair chunks\n";
    }
}

void test_collapsed_all_in_equity() {
    for (int board_size : {4, 3}) {
        auto config = tiny_config(board_size);
        config.betting_abstraction.first_bet_sizes.clear();
        config.terminal_mode = poker::TerminalMode::RecordComputed;
        const auto source = poker::holdem::HoldemSubgameBuilder(config).build();
        poker::Game game;
        game.starting_board = source.starting_board;
        game.set_hand_domains(source.p0_hands, source.p1_hands, source.hand_pairs);
        poker::PublicNode decision;
        decision.type = poker::PublicNodeType::Action;
        decision.player = poker::Player::P0;
        game.root = game.add_node(decision);
        for (int a = 0; a < 2; ++a) {
            const int leaf = game.add_node(poker::PublicNode{});
            game.add_action_child(game.root, leaf, game.add_action(poker::GameAction{a, 0}));
            game.terminal_records.push_back({a == 0 ? poker::TerminalType::AllIn : poker::TerminalType::P1_Fold,
                                             0, a == 0 ? 1000 : 0, 0});
        }
        game.register_action_state(game.root, poker::Player::P0);
        game.validate();
        const double expected = 500.0 * enumerated_equity(game);
        poker::TerminalValueProvider terminals;
        poker::CpuCfrSolver cpu(game, terminals);
        cpu.run_one_iteration();
        check_near(cpu.stats().last_root_value_p0, expected, 0.001, "Collapsed CPU all-in must match enumeration.");
        poker::GpuCfrConfig gpu_config;
        gpu_config.evaluator_data_dir = UTG_TEST_EVALUATOR_DIR;
        gpu_config.pair_chunk_size = 3;
        poker::GpuCfrSolver gpu(game, gpu_config);
        gpu.run_one_iteration();
        const auto regrets = gpu.regret_sum();
        for (std::size_t i = 0; i < regrets.size(); ++i)
            check_near(regrets[i] / game.hand_pairs.pair_count(), cpu.regret_sum()[i], 0.001,
                       "Collapsed GPU all-in must match enumerated CPU equity.");
    }
    std::cout << "[pass] collapsed flop/turn all-in equities match final-board enumeration\n";
}

void test_reject_unsafe_suit_compression() {
    auto config = royal_config(4);
    config.board_abstraction = poker::holdem::make_isomorphic_board_abstraction(config.p0_range, config.p1_range);
    bool rejected = false;
    try { (void)poker::holdem::HoldemSubgameBuilder(config).build(); }
    catch (const std::invalid_argument& error) {
        rejected = std::string(error.what()).find("private-hand remapping") != std::string::npos;
    }
    check(rejected, "Compressed runouts must not silently produce incorrect pair values.");
}

} // namespace

int main() { return run([] {
    test_certain_win_retains_probability_mass();
    test_enumerated_runouts();
    test_weighted_chance_reach();
    test_street_learning_parity();
    test_collapsed_all_in_equity();
    test_reject_unsafe_suit_compression();
}); }
