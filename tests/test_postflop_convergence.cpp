#include "test_support.hpp"
#include "postflop_reference.hpp"
#include "cfr_cpu.hpp"
#include "cfr_gpu.hpp"
#include <phevaluator/phevaluator.h>
#include <array>
#include <fstream>
#include <iomanip>
#include <limits>

using namespace test_support;

namespace {
using Matrix = std::array<std::array<double, 16>, 16>;
struct Benchmark {
    poker::Game game;
    std::array<double, 4> equity{};
    // P0 decisions: opening check/shove, then fold/call after check/shove.
    // P1 decisions: fold/call facing opening shove, then check/shove after check.
    std::array<int, 2> p0_states{}, p1_states{};
};

double independent_equity(const poker::Board& starting, poker::HoleCards p0, poker::HoleCards p1) {
    const auto dead = poker::board_mask(starting) | p0.mask() | p1.mask();
    int outcomes = 0;
    double wins = 0.0;
    for (int a = 0; a < 52; ++a) {
        if (dead & (poker::DeckMask{1} << a)) continue;
        for (int b = 0; b < (starting.size() == 3 ? a : 1); ++b) {
            if (starting.size() == 3 && (dead & (poker::DeckMask{1} << b))) continue;
            auto board = starting.cards;
            board.push_back(phevaluator::Card(a));
            if (board.size() == 4) board.push_back(phevaluator::Card(b));
            const auto r0 = phevaluator::EvaluateCards(p0.a, p0.b, board[0], board[1], board[2], board[3], board[4]);
            const auto r1 = phevaluator::EvaluateCards(p1.a, p1.b, board[0], board[1], board[2], board[3], board[4]);
            wins += r0 > r1 ? 1.0 : (r0 == r1 ? 0.5 : 0.0);
            ++outcomes;
        }
    }
    check(outcomes == (starting.size() == 3 ? 990 : 44), "Independent equity must enumerate legal final boards.");
    return wins / outcomes;
}

Benchmark benchmark(int street) {
    Benchmark b;
    auto config = tiny_config(street);
    b.game.starting_board = config.board;
    b.game.set_hand_domains(
        poker::HandDomain{{poker::make_hand(phevaluator::Card("Kh"), phevaluator::Card("Qh")),
                           poker::make_hand(phevaluator::Card("2d"), phevaluator::Card("3c"))}},
        poker::HandDomain{{poker::make_hand(phevaluator::Card("Qc"), phevaluator::Card("Qd")),
                           poker::make_hand(phevaluator::Card("Th"), phevaluator::Card("9h"))}},
        poker::HandPairTable{{0, 0, 1, 1}, {0, 1, 0, 1}, {3.0, 1.0, 6.0, 2.0}});
    for (int pair = 0; pair < 4; ++pair)
        b.equity[pair] = independent_equity(config.board,
            poker::hand_from_id(b.game.p0_hands.hands[b.game.hand_pairs.p0_index[pair]]),
            poker::hand_from_id(b.game.p1_hands.hands[b.game.hand_pairs.p1_index[pair]]));
    auto decision = [&](poker::Player player) {
        poker::PublicNode node; node.type = poker::PublicNodeType::Action; node.player = player;
        return b.game.add_node(node);
    };
    b.game.root = decision(poker::Player::P0);
    const int after_check = decision(poker::Player::P1);
    const int facing_shove = decision(poker::Player::P1);
    // Connect parents before descendants so Game's depth metadata is correct.
    auto actions = [&](int parent, int a, int c) {
        b.game.add_action_child(parent, a, b.game.add_action(poker::GameAction{0, 0}));
        b.game.add_action_child(parent, c, b.game.add_action(poker::GameAction{1, 1000}));
        b.game.register_action_state(parent, b.game.node(parent).player);
    };
    actions(b.game.root, after_check, facing_shove);
    const int checkdown = b.game.add_node(poker::PublicNode{});
    const int facing_return_shove = decision(poker::Player::P0);
    actions(after_check, checkdown, facing_return_shove);
    const int p1_fold = b.game.add_node(poker::PublicNode{});
    const int p1_call = b.game.add_node(poker::PublicNode{});
    actions(facing_shove, p1_fold, p1_call);
    const int p0_fold = b.game.add_node(poker::PublicNode{});
    const int p0_call = b.game.add_node(poker::PublicNode{});
    actions(facing_return_shove, p0_fold, p0_call);
    // Terminal nodes are ordered by creation, not traversal. Checkdown has no
    // further betting; AllIn records reuse the exact runout-equity evaluator.
    b.game.terminal_records = {
        {poker::TerminalType::AllIn, 0, 1000, 0},
        {poker::TerminalType::P1_Fold, 0, 2000, 1000},
        {poker::TerminalType::AllIn, 0, 3000, 1000},
        {poker::TerminalType::P0_Fold, 0, 2000, 0},
        {poker::TerminalType::AllIn, 0, 3000, 1000}
    };
    b.p0_states = {b.game.node(b.game.root).action_state_index,
                   b.game.node(facing_return_shove).action_state_index};
    b.p1_states = {b.game.node(facing_shove).action_state_index,
                   b.game.node(after_check).action_state_index};
    b.game.validate();
    return b;
}

Matrix independent_matrix(const Benchmark& b) {
    Matrix matrix{};
    double mass = 0.0;
    for (double weight : b.game.hand_pairs.weights) mass += weight;
    // Each player has four binary infoset decisions: two states x two own hands.
    for (int p0 = 0; p0 < 16; ++p0)
        for (int p1 = 0; p1 < 16; ++p1)
            for (int pair = 0; pair < 4; ++pair) {
                const int h0 = b.game.hand_pairs.p0_index[pair], h1 = b.game.hand_pairs.p1_index[pair];
                const bool opening_shove = p0 & (1 << h0);
                double utility;
                if (opening_shove) {
                    const bool call = p1 & (1 << h1);
                    utility = call ? 3000.0 * b.equity[pair] - 1000.0 : 1000.0;
                } else {
                    const bool return_shove = p1 & (1 << (2 + h1));
                    const bool call = p0 & (1 << (2 + h0));
                    utility = !return_shove ? 1000.0 * b.equity[pair]
                        : (call ? 3000.0 * b.equity[pair] - 1000.0 : 0.0);
                }
                matrix[p0][p1] += b.game.hand_pairs.weight(pair) / mass * utility;
            }
    return matrix;
}

std::array<double, 16> normal_form_policy(const Benchmark& b, const std::vector<float>& strategy,
                                         poker::Player player) {
    const auto& states = player == poker::Player::P0 ? b.p0_states : b.p1_states;
    std::array<double, 16> policy{};
    for (int pure = 0; pure < 16; ++pure) {
        policy[pure] = 1.0;
        for (int state = 0; state < 2; ++state)
            for (int hand = 0; hand < 2; ++hand)
                policy[pure] *= strategy[b.game.action_state(states[state]).tensor_index(
                    hand, (pure >> (state * 2 + hand)) & 1)];
    }
    return policy;
}

struct Bounds { double lower, upper, value; };
Bounds bounds(const Matrix& matrix, const std::array<double, 16>& p0, const std::array<double, 16>& p1) {
    Bounds result{std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), 0.0};
    for (int i = 0; i < 16; ++i) {
        double vs_p1 = 0.0, vs_p0 = 0.0;
        for (int j = 0; j < 16; ++j) {
            vs_p1 += matrix[i][j] * p1[j];
            vs_p0 += p0[j] * matrix[j][i];
            result.value += p0[i] * matrix[i][j] * p1[j];
        }
        result.upper = std::max(result.upper, vs_p1);
        result.lower = std::min(result.lower, vs_p0);
    }
    return result;
}

void check_policy(const Benchmark& b, const Matrix& matrix, const std::vector<float>& policy,
                  double reference, const char* label) {
    const auto result = bounds(matrix, normal_form_policy(b, policy, poker::Player::P0),
                               normal_form_policy(b, policy, poker::Player::P1));
    check_near(result.value, reference, 1.0, std::string(label) + " value must match independent minimax reference.");
    check((result.upper - result.lower) / 2.0 < 1.0,
          std::string(label) + " independent exploitability must be below 1 chip per 1000-chip starting pot. actual=" +
              std::to_string((result.upper - result.lower) / 2.0));
    poker::TerminalValueProvider terminals;
    poker::PublicExploitabilityEvaluator evaluator(b.game, terminals);
    check_near(evaluator.expected_value_p0(policy), result.value, 0.001, "Production EV agrees with independent matrix.");
    check_near(evaluator.exploitability(policy).exploitability, (result.upper - result.lower) / 2.0,
               0.001, "Production exploitability agrees with exhaustive independent pure responses.");
    std::cout << "[pass] " << b.game.starting_board.size() << "-card " << label
              << " reference=" << reference << " EV=" << result.value
              << " exploitability=" << (result.upper - result.lower) / 2.0 << '\n';
}

void test_convergence(int street) {
    const auto b = benchmark(street);
    const auto matrix = independent_matrix(b);
    const auto& reference = postflop_references[street == 4 ? 0 : 1];
    check(reference.street == street, "Certificate matches the starting street.");
    for (const auto& policy : {reference.p0, reference.p1}) {
        double sum = 0.0;
        for (double probability : policy) {
            check(probability >= -1e-10 && probability <= 1.0, "Certificate probabilities are valid.");
            sum += probability;
        }
        check_near(sum, 1.0, 1e-9, "Certificate probabilities sum to one.");
    }
    const auto certificate = bounds(matrix, reference.p0, reference.p1);
    check_near(certificate.lower, reference.value, 1e-7, "Reference P0 guarantees the minimax value against every pure response.");
    check_near(certificate.upper, reference.value, 1e-7, "Reference P1 caps the minimax value against every pure response.");
    const auto initial_policy = uniform_strategy(b.game);
    const auto initial = bounds(matrix, normal_form_policy(b, initial_policy, poker::Player::P0),
                                 normal_form_policy(b, initial_policy, poker::Player::P1));
    check((initial.upper - initial.lower) / 2.0 > 100.0, "Uniform starting policy must be meaningfully exploitable.");
    for (bool plus : {false, true}) {
        const int iterations = plus ? 20000 : 50000;
        poker::TerminalValueProvider terminals;
        poker::CfrConfig config;
        config.use_cfr_plus = plus; config.linear_averaging = plus;
        poker::CpuCfrSolver cpu(b.game, terminals, config);
        cpu.run_iterations(iterations);
        check_policy(b, matrix, cpu.average_strategy(), reference.value, plus ? "CPU CFR+" : "CPU CFR");
        poker::GpuCfrConfig gpu_config;
        gpu_config.evaluator_data_dir = UTG_TEST_EVALUATOR_DIR;
        gpu_config.use_cfr_plus = plus; gpu_config.linear_averaging = plus;
        gpu_config.pair_chunk_size = plus ? 3 : 4;
        poker::GpuCfrSolver gpu(b.game, gpu_config);
        gpu.run_iterations(iterations);
        check_policy(b, matrix, gpu.average_strategy(), reference.value, plus ? "GPU CFR+" : "GPU CFR");
    }
}
}

int main(int argc, char** argv) {
    return run([&] {
        if (argc == 3 && std::string(argv[1]) == "--export-matrix") {
            std::ofstream output(argv[2]);
            check(static_cast<bool>(output), "Open matrix export file.");
            output << std::setprecision(17);
            for (int street : {4, 3}) {
                const auto matrix = independent_matrix(benchmark(street));
                output << street << '\n';
                for (const auto& row : matrix) {
                    for (double value : row) output << value << ' ';
                    output << '\n';
                }
            }
            return;
        }
        if (argc == 3 && std::string(argv[1]) == "--street") {
            const int street = std::stoi(argv[2]);
            check(street == 3 || street == 4, "Benchmark street must be flop or turn.");
            test_convergence(street);
            return;
        }
        test_convergence(4); test_convergence(3);
    });
}
