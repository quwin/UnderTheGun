#include "exploitability.hpp"
#include "holdem/terminal_utility.hpp"

#include <unordered_map>
#include <utility>

namespace poker {

struct TerminalValueProvider::Cache {
    const Game* game;
    std::vector<int> terminal_index_by_node;
    std::unordered_map<BoardIndex, Board> boards;
    std::unique_ptr<holdem::AllInEquityCache> all_in_equities;

    explicit Cache(const Game& source) : game(&source), terminal_index_by_node(source.num_nodes(), -1) {
        int count = 0;
        for (int id = 0; id < source.num_nodes(); ++id)
            if (source.node(id).type == PublicNodeType::Terminal)
                terminal_index_by_node[id] = count++;

        const std::size_t expected_values = static_cast<std::size_t>(count) * source.hand_pairs.pair_count();
        if (!source.terminal_value_p0.empty()) {
            if (source.terminal_value_p0.size() != expected_values)
                throw std::invalid_argument("Terminal payoff table size must match terminal count times hand-pair count.");
        } else if (source.terminal_records.size() != static_cast<std::size_t>(count)) {
            throw std::invalid_argument("Terminal record count must match terminal node count.");
        }
    }

    const Board& board_for(BoardIndex index) {
        if (const auto found = boards.find(index); found != boards.end()) return found->second;
        return boards.emplace(index, make_board(game->starting_board, index)).first->second;
    }

    holdem::AllInEquityCache* equity_cache() {
        if (!all_in_equities) {
            Range p0, p1;
            p0.clear();
            p1.clear();
            for (auto hand : game->p0_hands.hands) p0.set_weight(hand, 1.0f);
            for (auto hand : game->p1_hands.hands) p1.set_weight(hand, 1.0f);
            auto initialized = std::make_unique<holdem::AllInEquityCache>();
            initialized->initialize_for_subgame(game->starting_board, p0, p1);
            all_in_equities = std::move(initialized);
        }
        return all_in_equities.get();
    }
};

TerminalValueProvider::TerminalValueProvider() = default;
TerminalValueProvider::~TerminalValueProvider() = default;

double TerminalValueProvider::utility_p0(const Game& game, int terminal_node_id, int hand_pair_id) const {
    if (terminal_node_id < 0 || terminal_node_id >= game.num_nodes())
        throw std::invalid_argument("terminal_node_id out of range.");
    if (game.node(terminal_node_id).type != PublicNodeType::Terminal)
        throw std::invalid_argument("Expected terminal node.");
    if (hand_pair_id < 0 || hand_pair_id >= game.hand_pairs.pair_count())
        throw std::invalid_argument("hand_pair_id out of range.");
    if (!cache_ || cache_->game != &game) cache_ = std::make_unique<Cache>(game);

    const auto terminal = static_cast<std::size_t>(cache_->terminal_index_by_node.at(terminal_node_id));
    // DebugComputed games contain both representations; preserve dense-table
    // semantics when a precomputed table is present.
    if (!game.terminal_value_p0.empty())
        return game.terminal_value_p0.at(terminal * game.hand_pairs.pair_count() + hand_pair_id);

    const auto& record = game.terminal_records.at(terminal);
    const auto& board = cache_->board_for(record.board_index);
    const holdem::PrivateState private_state{
        hand_from_id(game.p0_hands.hands.at(game.hand_pairs.p0_index.at(hand_pair_id))),
        hand_from_id(game.p1_hands.hands.at(game.hand_pairs.p1_index.at(hand_pair_id)))
    };
    // Public trees retain the starting private domain on every runout. Match
    // precomputed terminal semantics for impossible public/private combinations.
    if (private_state.overlaps_mask(board_mask(board))) return 0.0;

    holdem::PublicState state;
    state.board = board;
    state.pot = record.pot;
    state.betting.p0_committed_this_round = record.p0_committed;
    state.player_to_act = Player::Terminal;
    state.terminal_type = record.type;
    auto* equity = record.type == TerminalType::AllIn && !board.is_river()
        ? cache_->equity_cache() : nullptr;
    return holdem::terminal_utility_p0(state, private_state, equity);
}

} // namespace poker
