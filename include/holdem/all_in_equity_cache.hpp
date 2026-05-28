#pragma once

#include "holdem/private_state.hpp"
#include "holdem/street.hpp"

#include "poker/board.hpp"
#include "poker/deck_mask.hpp"
#include "poker/hand.hpp"
#include "poker/range.hpp"
#include "../../external/PokerHandEvaluator/cpp/include/phevaluator/phevaluator.h"
#include "../../external/PokerHandEvaluator/cpp/include/phevaluator/rank.h"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace poker::holdem {

class AllInEquityCache {
public:
    AllInEquityCache() = default;

    void initialize_for_subgame(
        const Board& start_board,
        Street start_street,
        const Range& p0_range,
        const Range& p1_range
    ) {
        start_board.validate();
        validate_street(start_street);

        if (!board_size_matches_street(start_street, start_board.size())) {
            throw std::invalid_argument(
                "AllInEquityCache board size does not match start street."
            );
        }

        start_board_ = start_board;
        start_street_ = start_street;
        initialized_ = true;

        p0_index_.fill(-1);
        p1_index_.fill(-1);

        const DeckMask board_dead = board_mask(start_board_);

        p0_hands_ = p0_range.legal_hands(board_dead);
        p1_hands_ = p1_range.legal_hands(board_dead);

        for (int i = 0; i < static_cast<int>(p0_hands_.size()); ++i) {
            p0_index_[p0_hands_[i]] = i;
        }

        for (int j = 0; j < static_cast<int>(p1_hands_.size()); ++j) {
            p1_index_[p1_hands_[j]] = j;
        }

        pair_count_ =
            static_cast<int>(p0_hands_.size() * p1_hands_.size());

        flop_equity_.assign(pair_count_, 0.0f);
        flop_computed_.assign(pair_count_, 0);

        for (TurnCardCache& cache : turn_cache_) {
            cache = TurnCardCache{};
        }
    }

    float p0_equity(
        const Board& board,
        Street street,
        const PrivateState& private_state
    ) {
        ensure_initialized();

        board.validate();
        private_state.validate();

        if (!board_size_matches_street(street, board.size())) {
            throw std::invalid_argument(
                "AllInEquityCache::p0_equity board size does not match street."
            );
        }

        const HandId p0 = make_hand(private_state.p0_hand);
        const HandId p1 = make_hand(private_state.p1_hand);

        if (street == Street::Flop) {
            return flop_equity(p0, p1, board, private_state);
        }

        if (street == Street::Turn) {
            const phevaluator::Card turn_card = turn_card_from_board(board);
            return turn_equity(p0, p1, board, turn_card, private_state);
        }

        if (street == Street::River) {
            return river_equity(board, private_state);
        }

        throw std::invalid_argument(
            "AllInEquityCache supports flop, turn, and river equities."
        );
    }

private:
    struct TurnCardCache {
        std::vector<float> equity;
        std::vector<std::uint8_t> computed;
        bool allocated = false;
    };

    bool initialized_ = false;

    Board start_board_;
    Street start_street_ = Street::River;

    std::vector<HandId> p0_hands_;
    std::vector<HandId> p1_hands_;

    std::array<int, kNumHands> p0_index_{};
    std::array<int, kNumHands> p1_index_{};

    int pair_count_ = 0;

    std::vector<float> flop_equity_;
    std::vector<std::uint8_t> flop_computed_;

    // Indexed by raw CardId 0..51. Illegal cards are never used.
    std::array<TurnCardCache, kNumCards> turn_cache_{};

    void ensure_initialized() const {
        if (!initialized_) {
            throw std::logic_error("AllInEquityCache was not initialized.");
        }
    }

    int pair_index(HandId p0, HandId p1) const {
        validate_hand_id(p0);
        validate_hand_id(p1);

        const int i = p0_index_[p0];
        const int j = p1_index_[p1];

        if (i < 0 || j < 0) {
            throw std::invalid_argument(
                "Hand pair is not present in AllInEquityCache ranges."
            );
        }

        return i * static_cast<int>(p1_hands_.size()) + j;
    }

    static phevaluator::Card turn_card_from_board(const Board& board) {
        if (!board.is_turn()) {
            throw std::invalid_argument("Expected turn board.");
        }
        return board.cards[3];
    }

    void ensure_turn_allocated(phevaluator::Card turn_card) {
        validate_card(turn_card);

        TurnCardCache& cache = turn_cache_[turn_card];

        if (!cache.allocated) {
            cache.equity.assign(pair_count_, 0.0f);
            cache.computed.assign(pair_count_, 0);
            cache.allocated = true;
        }
    }

    float flop_equity(
        HandId p0,
        HandId p1,
        const Board& board,
        const PrivateState& private_state
    ) {
        if (!board.is_flop()) {
            throw std::invalid_argument("Expected flop board.");
        }

        const int idx = pair_index(p0, p1);

        if (!flop_computed_[idx]) {
            flop_equity_[idx] = compute_exact_equity(board, Street::Flop, private_state);
            flop_computed_[idx] = 1;
        }
        return flop_equity_[idx];
    }

    float turn_equity(
        const HandId p0,
        const HandId p1,
        const Board& board,
        phevaluator::Card turn_card,
        const PrivateState& private_state
    ) {
        if (!board.is_turn()) {
            throw std::invalid_argument("Expected turn board.");
        }
        ensure_turn_allocated(turn_card);
        const int idx = pair_index(p0, p1);
        TurnCardCache& cache = turn_cache_[turn_card];
        if (!cache.computed[idx]) {
            cache.equity[idx] = compute_exact_equity(board, Street::Turn, private_state);
            cache.computed[idx] = 1;
        }
        return cache.equity[idx];
    }

    static float river_equity(
        const Board& board,
        const PrivateState& private_state
    ) {
        if (!board.is_river()) {
            throw std::invalid_argument("Expected river board.");
        }
        const phevaluator::Rank p0_rank = phevaluator::EvaluateCards(
            private_state.p0_hand.a,
            private_state.p0_hand.b,
            board.cards[0],
            board.cards[1],
            board.cards[2],
            board.cards[3],
            board.cards[4]
        );
        const phevaluator::Rank p1_rank = phevaluator::EvaluateCards(
            private_state.p1_hand.a,
            private_state.p1_hand.b,
            board.cards[0],
            board.cards[1],
            board.cards[2],
            board.cards[3],
            board.cards[4]
        );
        // Smaller value is stronger, from 1 to 7462
        if (p0_rank < p1_rank) {
            return 1.0;
        }
        if (p1_rank == p0_rank) {
            return 0.5;
        }
        return 0.0;
    }

    static float compute_exact_equity(
        const Board& board,
        const Street street,
        const PrivateState& private_state
    ) {
        board.validate();
        private_state.validate();

        if (private_state.overlaps_mask(board_mask(board))) {
            throw std::invalid_argument(
                "compute_exact_equity private hands overlap board."
            );
        }
        DeckMask dead =
            board_mask(board) |
            private_state.p0_mask() |
            private_state.p1_mask();

        const std::vector<phevaluator::Card> remaining = cards_from_mask(remaining_cards(dead));
        double score = 0.0;
        int total = 0;

        if (street == Street::Turn) {
            for (phevaluator::Card river : remaining) {
                Board river_board = board.with_added_card(river);
                const phevaluator::Rank p0_rank = phevaluator::EvaluateCards(
                   private_state.p0_hand.a,
                   private_state.p0_hand.b,
                   river_board.cards[0],
                   river_board.cards[1],
                   river_board.cards[2],
                   river_board.cards[3],
                   river_board.cards[4]
                );
                const phevaluator::Rank p1_rank = phevaluator::EvaluateCards(
                    private_state.p1_hand.a,
                    private_state.p1_hand.b,
                    river_board.cards[0],
                    river_board.cards[1],
                    river_board.cards[2],
                    river_board.cards[3],
                    river_board.cards[4]
                );
                // Smaller value is stronger, from 1 to 7462
                if (p0_rank < p1_rank) {
                    score += 1.0;
                } else if (p1_rank == p0_rank) {
                    score += 0.5;
                }
                ++total;
            }
        } else if (street == Street::Flop) {
            for (std::size_t i = 0; i < remaining.size(); ++i) {
                for (std::size_t j = i + 1; j < remaining.size(); ++j) {
                    Board river_board = board;
                    river_board = river_board.with_added_card(remaining[i]);
                    river_board = river_board.with_added_card(remaining[j]);

                    const phevaluator::Rank p0_rank = phevaluator::EvaluateCards(
                       private_state.p0_hand.a,
                       private_state.p0_hand.b,
                       river_board.cards[0],
                       river_board.cards[1],
                       river_board.cards[2],
                       river_board.cards[3],
                       river_board.cards[4]
                    );
                    const phevaluator::Rank p1_rank = phevaluator::EvaluateCards(
                        private_state.p1_hand.a,
                        private_state.p1_hand.b,
                        river_board.cards[0],
                        river_board.cards[1],
                        river_board.cards[2],
                        river_board.cards[3],
                        river_board.cards[4]
                    );
                    // Smaller value is stronger, from 1 to 7462
                    if (p0_rank < p1_rank) {
                        score += 1.0;
                    } else if (p1_rank == p0_rank) {
                        score += 0.5;
                    }
                    ++total;
                }
            }
        } else {
            throw std::invalid_argument(
                "compute_exact_equity supports flop and turn only."
            );
        }
        if (total <= 0) {
            throw std::runtime_error("No legal all-in runouts.");
        }
        return static_cast<float>(score / static_cast<double>(total));
    }
};

} // namespace poker::holdem