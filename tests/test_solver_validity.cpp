#include "game.hpp"
#include "cfr_cpu.hpp"
#include "test_support.hpp"

#include "holdem/action.hpp"
#include "holdem/betting_abstraction.hpp"
#include "holdem/subgame_builder.hpp"
#include "holdem/subgame_config.hpp"

#include "poker/board.hpp"
#include "poker/range.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kEvTol = 1e-4;

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
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        std::ostringstream oss;
        oss << message
            << " actual=" << actual
            << " expected=" << expected
            << " tolerance=" << tolerance;
        throw std::runtime_error(oss.str());
    }
}

poker::Board make_river_board() {
    return poker::Board{
        {
            phevaluator::Card("As"),
            phevaluator::Card("7h"),
            phevaluator::Card("Jh"),
            phevaluator::Card("Ts"),
            phevaluator::Card("3d"),
        }
    };
}

poker::Range make_p0_tiny_range() {
    poker::Range range;
    range.clear();

    // Pair of aces on the fixture board.
    range.set_weight(
        poker::make_hand(phevaluator::Card("Ah"), phevaluator::Card("Kh")),
        1.0f
    );

    // Pair of kings.
    range.set_weight(
        poker::make_hand(phevaluator::Card("Ks"), phevaluator::Card("Kd")),
        1.0f
    );

    return range;
}

poker::Range make_p1_tiny_range() {
    poker::Range range;
    range.clear();

    // Pair of queens.
    range.set_weight(
        poker::make_hand(phevaluator::Card("Qc"), phevaluator::Card("Qd")),
        1.0f
    );

    // Pair of tens on the fixture board.
    range.set_weight(
        poker::make_hand(phevaluator::Card("Th"), phevaluator::Card("9h")),
        1.0f
    );

    return range;
}

poker::holdem::BettingAbstraction make_check_only_betting() {
    poker::holdem::BettingAbstraction abstraction;

    // No bet sizes means the only legal unopened action should be check.
    abstraction.first_bet_sizes = {};
    abstraction.raise_sizes = {};
    abstraction.max_raises_per_street = 0;
    abstraction.always_allow_all_in = false;

    return abstraction;
}

poker::holdem::BettingAbstraction make_pot_bet_no_raise_betting() {
    poker::holdem::BettingAbstraction abstraction;

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
    abstraction.always_allow_all_in = false;

    return abstraction;
}

poker::holdem::HoldemSubgameConfig make_base_config() {
    poker::holdem::HoldemSubgameConfig config;

    config.board = make_river_board();

    config.pot_size = 1000;
    config.effective_stack = 2000;
    config.player_to_act = poker::Player::P0;

    config.p0_range = make_p0_tiny_range();
    config.p1_range = make_p1_tiny_range();

    config.collapse_all_in_runouts_to_ev = true;
    config.terminal_mode = poker::TerminalMode::ValuePrecomputed;
    return config;
}

poker::Game build_check_only_game() {
    poker::holdem::HoldemSubgameConfig config = make_base_config();
    config.betting_abstraction = make_check_only_betting();

    return poker::holdem::HoldemSubgameBuilder(config).build();
}

poker::Game build_pot_bet_game() {
    poker::holdem::HoldemSubgameConfig config = make_base_config();
    config.betting_abstraction = make_pot_bet_no_raise_betting();

    return poker::holdem::HoldemSubgameBuilder(config).build();
}

void test_check_only_river_ev_constant() {
    const auto game = build_check_only_game();
    check(game.hand_pairs.pair_count() == 4, "Expected four private matchups.");
    const auto uniform = test_support::uniform_strategy(game);
    test_support::check_strategy(game, uniform);
    poker::TerminalValueProvider terminal_values;
    poker::PublicExploitabilityEvaluator evaluator(game, terminal_values);
    // AhKh has a pair of aces; KK beats both QQ and Th9h (a pair of tens).
    check_near(evaluator.expected_value_p0(uniform), 1000.0, kEvTol,
               "Check-only EV should be 1000.");
    check_near(evaluator.exploitability(uniform).exploitability, 0.0, kEvTol,
               "A game with no decisions should have zero exploitability.");
    poker::CpuCfrSolver solver(game, terminal_values);
    solver.run_iterations(10);
    test_support::check_strategy(game, solver.average_strategy());
    check_near(solver.stats().last_root_value_p0, 1000.0, kEvTol,
               "CPU traversal must reproduce analytical EV.");
    std::cout << "[pass] check-only river EV\n";
}

void test_default_record_terminal_river_ev() {
    auto config = make_base_config();
    config.terminal_mode = poker::holdem::HoldemSubgameConfig{}.terminal_mode;
    config.betting_abstraction = make_check_only_betting();
    const auto game = poker::holdem::HoldemSubgameBuilder(config).build();
    check(game.terminal_value_p0.empty() && !game.terminal_records.empty(),
          "Default game must store terminal records without a dense payoff table.");
    poker::TerminalValueProvider terminal_values;
    poker::CpuCfrSolver solver(game, terminal_values);
    solver.run_iterations(10);
    test_support::check_strategy(game, solver.average_strategy());
    check_near(solver.stats().last_root_value_p0, 1000.0, kEvTol,
               "Default record-computed CPU solve must reproduce known river EV.");
    poker::PublicExploitabilityEvaluator evaluator(game, terminal_values);
    check_near(evaluator.expected_value_p0(solver.average_strategy()), 1000.0, kEvTol,
               "Record-computed strategy evaluation must reproduce known river EV.");
    std::cout << "[pass] default record-computed river EV\n";
}

void test_record_and_precomputed_terminals_agree() {
    for (int board_cards : {5, 4, 3}) {
        auto config = test_support::tiny_config(board_cards);
        config.effective_stack = config.pot_size;
        config.terminal_mode = poker::TerminalMode::ValuePrecomputed;
        const auto dense = poker::holdem::HoldemSubgameBuilder(config).build();
        config.terminal_mode = poker::TerminalMode::RecordComputed;
        const auto records = poker::holdem::HoldemSubgameBuilder(config).build();
        check(records.terminal_value_p0.empty(), "Record mode must not materialize a dense payoff table.");
        check(records.num_nodes() == dense.num_nodes(), "Terminal modes must preserve tree structure.");
        poker::TerminalValueProvider dense_values, record_values;
        bool saw_all_in = false;
        bool saw_p0_fold = false;
        bool saw_p1_fold = false;
        bool saw_showdown = false;
        bool saw_extended_board = false;
        for (const auto& record : records.terminal_records) {
            saw_all_in |= record.type == poker::TerminalType::AllIn;
            saw_p0_fold |= record.type == poker::TerminalType::P0_Fold;
            saw_p1_fold |= record.type == poker::TerminalType::P1_Fold;
            saw_showdown |= record.type == poker::TerminalType::Showdown;
            saw_extended_board |= record.board_index != 0;
        }
        check((saw_all_in || board_cards == 5) && saw_p0_fold && saw_p1_fold && saw_showdown,
              "Parity fixture must exercise all terminal types.");
        check(board_cards == 5 || saw_extended_board, "Parity fixture must exercise board decoding.");
        for (int id = 0; id < records.num_nodes(); ++id) {
            if (records.node(id).type != poker::PublicNodeType::Terminal) continue;
            for (int pair = 0; pair < records.hand_pairs.pair_count(); ++pair)
                check_near(record_values.utility_p0(records, id, pair),
                           dense_values.utility_p0(dense, id, pair), 1e-4,
                           "Record and dense terminal values must agree on every pair/runout.");
        }
        // Whole-traversal parity verifies the new provider is used by CFR.
        poker::CpuCfrSolver dense_solver(dense, dense_values);
        poker::CpuCfrSolver record_solver(records, record_values);
        dense_solver.run_iterations(3);
        record_solver.run_iterations(3);
        const auto dense_average = dense_solver.average_strategy();
        const auto record_average = record_solver.average_strategy();
        test_support::check_strategy(records, record_average);
        check(dense_average.size() == record_average.size(), "Strategy tensor sizes must match.");
        for (std::size_t i = 0; i < dense_average.size(); ++i)
            check_near(record_average[i], dense_average[i], 1e-6, "Terminal modes must produce the same CPU strategy.");
        check_near(record_solver.stats().last_root_value_p0, dense_solver.stats().last_root_value_p0,
                   1e-4, "Terminal modes must produce the same CPU root EV.");
        std::cout << "[pass] record/dense terminal and CFR parity (" << board_cards << " board cards)\n";
    }
}

void test_record_payoffs_are_analytically_correct() {
    auto evaluate = [](const poker::Board& board, poker::HandId p0, poker::HandId p1,
                       poker::TerminalType type) {
        poker::Game game;
        game.starting_board = board;
        game.set_hand_domains(poker::HandDomain{{p0}}, poker::HandDomain{{p1}},
                              poker::HandPairTable{{0}, {0}});
        poker::PublicNode terminal;
        terminal.type = poker::PublicNodeType::Terminal;
        terminal.player = poker::Player::Terminal;
        game.root = game.add_node(terminal);
        game.terminal_records.push_back(poker::TerminalRecord{type, 0, 1400, 200});
        poker::TerminalValueProvider values;
        return values.utility_p0(game, game.root, 0);
    };
    const auto strong = poker::make_hand(phevaluator::Card("Ah"), phevaluator::Card("Kh"));
    const auto weak = poker::make_hand(phevaluator::Card("Qc"), phevaluator::Card("Qd"));
    const auto river = make_river_board();
    check_near(evaluate(river, strong, weak, poker::TerminalType::Showdown), 1200.0, kEvTol,
               "Winning record utility must subtract P0's contribution.");
    check_near(evaluate(river, weak, strong, poker::TerminalType::Showdown), -200.0, kEvTol,
               "Losing record utility must subtract P0's contribution.");
    check_near(evaluate(river, strong, weak, poker::TerminalType::P0_Fold), -200.0, kEvTol,
               "P0 fold record utility must lose P0's contribution.");
    check_near(evaluate(river, strong, weak, poker::TerminalType::P1_Fold), 1200.0, kEvTol,
               "P1 fold record utility must award the net pot.");
    const poker::Board royal_board{{phevaluator::Card("As"), phevaluator::Card("Ks"),
                                   phevaluator::Card("Qs"), phevaluator::Card("Js"), phevaluator::Card("Ts")}};
    check_near(evaluate(royal_board, strong, weak, poker::TerminalType::Showdown), 500.0, kEvTol,
               "Tie record utility must award half the pot minus P0's contribution.");
    poker::Board royal_draw{{phevaluator::Card("Qh"), phevaluator::Card("Jh"), phevaluator::Card("Th")}};
    check_near(evaluate(royal_draw, strong, weak, poker::TerminalType::AllIn), 1200.0, kEvTol,
               "Flop all-in with a royal flush must have certain-win EV.");
    royal_draw.cards.push_back(phevaluator::Card("2c"));
    check_near(evaluate(royal_draw, strong, weak, poker::TerminalType::AllIn), 1200.0, kEvTol,
               "Turn all-in with a royal flush must have certain-win EV.");
    std::cout << "[pass] analytical record payoffs: win, loss, tie, folds and pre-river all-ins\n";
}

void test_terminal_provider_rejects_invalid_input() {
    auto config = make_base_config();
    config.terminal_mode = poker::TerminalMode::RecordComputed;
    config.betting_abstraction = make_check_only_betting();
    const auto game = poker::holdem::HoldemSubgameBuilder(config).build();
    int terminal = -1;
    for (int id = 0; id < game.num_nodes(); ++id)
        if (game.node(id).type == poker::PublicNodeType::Terminal) terminal = id;
    check(terminal >= 0, "Invalid-input fixture must contain a terminal.");
    auto rejects = [](auto operation) {
        try {
            operation();
        } catch (const std::invalid_argument&) {
            return;
        }
        throw std::runtime_error("Terminal provider must reject invalid input with a clear argument error.");
    };
    poker::TerminalValueProvider values;
    rejects([&] { (void)values.utility_p0(game, -1, 0); });
    rejects([&] { (void)values.utility_p0(game, game.root, 0); });
    rejects([&] { (void)values.utility_p0(game, terminal, game.hand_pairs.pair_count()); });
    auto incomplete = game;
    incomplete.terminal_records.clear();
    rejects([&] { (void)values.utility_p0(incomplete, terminal, 0); });
    // A failed cache initialization must not prevent subsequent valid calls.
    check_near(values.utility_p0(game, terminal, 0), 1000.0, kEvTol,
               "Valid terminal evaluation must work after rejecting malformed metadata.");
    std::cout << "[pass] terminal provider rejects invalid input\n";
}

void test_river_training_has_learning_signal() {
    const auto game = build_pot_bet_game();
    poker::TerminalValueProvider terminal_values;
    poker::CpuCfrSolver solver(game, terminal_values);
    test_support::check_strategy(game, solver.average_strategy());
    const auto initial = solver.current_strategy();
    solver.run_iterations(1000);
    const auto current = solver.current_strategy();
    const auto average = solver.average_strategy();
    test_support::check_strategy(game, current);
    test_support::check_strategy(game, average);
    check(solver.stats().iterations_run == 1000, "Iteration count mismatch.");
    bool learned = false;
    for (std::size_t i = 0; i < current.size(); ++i)
        learned |= std::abs(current[i] - initial[i]) > 0.05f;
    check(learned, "River strategy must change in response to terminal payoffs.");
    std::cout << "[pass] river training learns and produces normalized tensors\n";
}

void test_known_single_pair_river_converges() {
    auto config = test_support::tiny_config();
    config.p0_range.clear();
    config.p1_range.clear();
    config.p0_range.set_weight(poker::make_hand(phevaluator::Card("Kh"), phevaluator::Card("Qh")), 1.0f);
    config.p1_range.set_weight(poker::make_hand(phevaluator::Card("Qc"), phevaluator::Card("Qd")), 1.0f);
    const auto game = poker::holdem::HoldemSubgameBuilder(config).build();
    check(game.hand_pairs.pair_count() == 1, "Convergence fixture must have exactly one pair.");
    poker::TerminalValueProvider terminal_values;
    poker::PublicExploitabilityEvaluator evaluator(game, terminal_values);
    poker::CpuCfrSolver solver(game, terminal_values);
    const auto initial = evaluator.exploitability(solver.average_strategy());
    // The current best-response evaluator maximizes per private pair. With one
    // pair there is no hidden opponent hand, so it is a valid convergence oracle.
    solver.run_iterations(4000);
    const auto average = solver.average_strategy();
    test_support::check_strategy(game, average);
    const auto result = evaluator.exploitability(average);
    check(initial.exploitability > 100.0, "Uniform strategy must be exploitable.");
    check(result.exploitability < 1.0, "Known river game must converge to low exploitability.");
    check_near(result.strategy_value_p0, 1000.0, 1.0,
               "Winning straight must retain the starting pot at equilibrium.");
    for (const auto& state : game.action_states) {
        const int fold = test_support::local_action(game, state, poker::holdem::ActionType::Fold);
        if (state.player == 1 && fold >= 0)
            check(average.at(state.tensor_index(0, fold)) > 0.99f,
                  "Known losing hand must fold to a river bet.");
    }
    std::cout << "[pass] single-pair river convergence: exploitability=" << result.exploitability << '\n';
}

} // namespace

int main() {
    return test_support::run([] {
        test_check_only_river_ev_constant();
        test_default_record_terminal_river_ev();
        test_record_and_precomputed_terminals_agree();
        test_record_payoffs_are_analytically_correct();
        test_terminal_provider_rejects_invalid_input();
        test_river_training_has_learning_signal();
        test_known_single_pair_river_converges();
    });
}
