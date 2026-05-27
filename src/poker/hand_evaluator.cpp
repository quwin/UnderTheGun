#include "poker/hand_evaluator.hpp"
#include "poker/deck_mask.hpp"
#include "../external/PokerHandEvaluator/cpp/include/phevaluator/phevaluator.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../external/PokerHandEvaluator/cpp/include/phevaluator/rank.h"

namespace phevaluator {
    class Rank;
}

namespace poker {

namespace {
    constexpr std::uint32_t kPheWeakestRank = 7462;

    int to_phe_card_id(CardId card) {
        validate_card(card);

        const int rank_index = static_cast<int>(rank_of(card)) - 2; // 2..A -> 0..12
        const int suit_index = static_cast<int>(suit_of(card));     // C,D,H,S -> 0..3

        return rank_index * 4 + suit_index;
    }

    phevaluator::Card to_phe_card(CardId card) {
        return {to_phe_card_id(card)};
    }

    std::uint32_t phe_score_from_rank(const phevaluator::Rank& rank) {
        // PHEvaluator: 1 is strongest, 7462 is weakest.
        // Your project: higher score means stronger.
        return kPheWeakestRank + 1U - static_cast<std::uint32_t>(rank.value());
    }

    HandCategory category_from_phe(const phevaluator::Rank& rank) {
        switch (rank.category()) {
            case rank_category::STRAIGHT_FLUSH:
                return HandCategory::StraightFlush;
            case FOUR_OF_A_KIND:
                return HandCategory::FourOfAKind;
            case FULL_HOUSE:
                return HandCategory::FullHouse;
            case FLUSH:
                return HandCategory::Flush;
            case STRAIGHT:
                return HandCategory::Straight;
            case THREE_OF_A_KIND:
                return HandCategory::ThreeOfAKind;
            case TWO_PAIR:
                return HandCategory::TwoPair;
            case ONE_PAIR:
                return HandCategory::OnePair;
            case HIGH_CARD:
                return HandCategory::HighCard;
        }

        throw std::runtime_error("Unknown PHEvaluator rank category.");
    }
    std::array<CardId, 7> make_seven_cards(
        const HoleCards& hand,
        const Board& board
    ) {
        if (!board.is_river()) {
            throw std::invalid_argument(
                "evaluate_7 requires a complete five-card river board."
            );
        }

        hand.validate();
        board.validate();

        DeckMask seen = empty_deck_mask();

        auto add_unique = [&](CardId card) {
            validate_card(card);
            if (contains_card(seen, card)) {
                throw std::invalid_argument(
                    "evaluate_7 received duplicate cards."
                );
            }
            seen = add_card(seen, card);
        };
        add_unique(hand.a);
        add_unique(hand.b);
        for (CardId card : board.cards) {
            add_unique(card);
        }
        return std::array<CardId, 7>{
            hand.a,
            hand.b,
            board.cards[0],
            board.cards[1],
            board.cards[2],
            board.cards[3],
            board.cards[4]
        };
    }
    std::array<int, 5> sort_desc(std::array<int, 5> ranks) {
        std::sort(ranks.begin(), ranks.end(), std::greater<int>());
        return ranks;
    }
    std::uint32_t packed_score(
        HandCategory category,
        const std::array<int, 5>& ranks
    ) {
        std::uint32_t score = static_cast<std::uint32_t>(static_cast<int>(category)) << 20;

        score |= static_cast<std::uint32_t>(ranks[0] & 0xF) << 16;
        score |= static_cast<std::uint32_t>(ranks[1] & 0xF) << 12;
        score |= static_cast<std::uint32_t>(ranks[2] & 0xF) << 8;
        score |= static_cast<std::uint32_t>(ranks[3] & 0xF) << 4;
        score |= static_cast<std::uint32_t>(ranks[4] & 0xF);

        return score;
    }

    int straight_high_card(const std::array<int, 15>& rank_count) {
        // Normal straights: A-high down to 6-high.
        for (int high = 14; high >= 6; --high) {
            bool present = true;

            for (int r = high; r >= high - 4; --r) {
                if (rank_count[r] == 0) {
                    present = false;
                    break;
                }
            }

            if (present) {
                return high;
            }
        }
        // Wheel: A-2-3-4-5.
        if (rank_count[14] > 0 &&
            rank_count[5] > 0 &&
            rank_count[4] > 0 &&
            rank_count[3] > 0 &&
            rank_count[2] > 0) {
            return 5;
        }
        return 0;
    }

    HandStrength make_strength(
        HandCategory category,
        std::array<int, 5> ranks
    ) {
        HandStrength strength;
        strength.category = category;
        strength.ranks = ranks;
        strength.score = packed_score(category, ranks);
        return strength;
    }
    HandStrength evaluate_5(const std::array<CardId, 5>& cards) {
    std::array<int, 15> rank_count{};
    std::array<int, 4> suit_count{};
    for (CardId card : cards) {
        const int rank = static_cast<int>(rank_of(card));
        const int suit = static_cast<int>(suit_of(card));

        ++rank_count[rank];
        ++suit_count[suit];
    }
    const bool is_flush =
        suit_count[0] == 5 ||
        suit_count[1] == 5 ||
        suit_count[2] == 5 ||
        suit_count[3] == 5;

    const int straight_high = straight_high_card(rank_count);

    if (is_flush && straight_high > 0) {
        return make_strength(
            HandCategory::StraightFlush,
            {straight_high, 0, 0, 0, 0}
        );
    }

    int quad_rank = 0;
    int trips_rank = 0;
    std::vector<int> pair_ranks;
    std::vector<int> single_ranks;

    for (int rank = 14; rank >= 2; --rank) {
        if (rank_count[rank] == 4) {
            quad_rank = rank;
        } else if (rank_count[rank] == 3) {
            if (trips_rank == 0) {
                trips_rank = rank;
            }
        } else if (rank_count[rank] == 2) {
            pair_ranks.push_back(rank);
        } else if (rank_count[rank] == 1) {
            single_ranks.push_back(rank);
        }
    }

    if (quad_rank > 0) {
        int kicker = 0;

        for (int rank = 14; rank >= 2; --rank) {
            if (rank != quad_rank && rank_count[rank] > 0) {
                kicker = rank;
                break;
            }
        }

        return make_strength(
            HandCategory::FourOfAKind,
            {quad_rank, kicker, 0, 0, 0}
        );
    }

    if (trips_rank > 0 && !pair_ranks.empty()) {
        return make_strength(
            HandCategory::FullHouse,
            {trips_rank, pair_ranks[0], 0, 0, 0}
        );
    }

    if (is_flush) {
        std::array<int, 5> ranks{};

        for (int i = 0; i < 5; ++i) {
            ranks[i] = static_cast<int>(rank_of(cards[i]));
        }

        ranks = sort_desc(ranks);

        return make_strength(
            HandCategory::Flush,
            ranks
        );
    }

    if (straight_high > 0) {
        return make_strength(
            HandCategory::Straight,
            {straight_high, 0, 0, 0, 0}
        );
    }

    if (trips_rank > 0) {
        return make_strength(
            HandCategory::ThreeOfAKind,
            {
                trips_rank,
                single_ranks[0],
                single_ranks[1],
                0,
                0
            }
        );
    }

    if (pair_ranks.size() >= 2) {
        return make_strength(
            HandCategory::TwoPair,
            {
                pair_ranks[0],
                pair_ranks[1],
                single_ranks[0],
                0,
                0
            }
        );
    }

    if (pair_ranks.size() == 1) {
        return make_strength(
            HandCategory::OnePair,
            {
                pair_ranks[0],
                single_ranks[0],
                single_ranks[1],
                single_ranks[2],
                0
            }
        );
    }

    return make_strength(
        HandCategory::HighCard,
        {
            single_ranks[0],
            single_ranks[1],
            single_ranks[2],
            single_ranks[3],
            single_ranks[4]
        }
    );
}

} // namespace
    HandStrength HandEvaluator::evaluate_7(
        const HoleCards& hand,
        const Board& board
    ) {
        const std::array<CardId, 7> cards = make_seven_cards(hand, board);
        const phevaluator::Rank phe_rank = phevaluator::EvaluateCards(
            to_phe_card(cards[0]),
            to_phe_card(cards[1]),
            to_phe_card(cards[2]),
            to_phe_card(cards[3]),
            to_phe_card(cards[4]),
            to_phe_card(cards[5]),
            to_phe_card(cards[6])
        );

        // Keep your existing detailed category/ranks behavior for tests/debugging.
        // This still does 21 choose-5 work, but only once per terminal construction.
        // compare_7 below will use the fast PHEvaluator score.
        bool have_best = false;
        HandStrength best;

        for (int a = 0; a < 7; ++a) {
            for (int b = a + 1; b < 7; ++b) {
                for (int c = b + 1; c < 7; ++c) {
                    for (int d = c + 1; d < 7; ++d) {
                        for (int e = d + 1; e < 7; ++e) {
                            const std::array<CardId, 5> subset{
                                cards[a],
                                cards[b],
                                cards[c],
                                cards[d],
                                cards[e]
                            };

                            const HandStrength current = evaluate_5(subset);

                            if (!have_best || current > best) {
                                best = current;
                                have_best = true;
                            }
                        }
                    }
                }
            }
        }

    best.category = category_from_phe(phe_rank);
    best.score = phe_score_from_rank(phe_rank);

    return best;
}

    // Returns:
    //   +1 if p0 wins
    //    0 if tie
    //   -1 if p1 wins
    int HandEvaluator::compare_7(
        const HoleCards& p0_hand,
        const HoleCards& p1_hand,
        const Board& board
    ) {
        const phevaluator::Rank p0_rank = phevaluator::EvaluateCards(
            to_phe_card(p0_hand.a),
            to_phe_card(p0_hand.b),
            to_phe_card(board.cards[0]),
            to_phe_card(board.cards[3]),
            to_phe_card(board.cards[4]),
            to_phe_card(board.cards[5]),
            to_phe_card(board.cards[6])
        );
        const phevaluator::Rank p1_rank = phevaluator::EvaluateCards(
            to_phe_card(p1_hand.a),
            to_phe_card(p1_hand.b),
            to_phe_card(board.cards[0]),
            to_phe_card(board.cards[3]),
            to_phe_card(board.cards[4]),
            to_phe_card(board.cards[5]),
            to_phe_card(board.cards[6])
        );
        if (p0_rank.value() > p1_rank.value()) {
            return 1;
        }
        if (p1_rank.value() > p0_rank.value()) {
            return -1;
        }
        return 0;
    }

    // Utility helpers.
    std::uint32_t make_hand_score(
        HandCategory category,
        const std::array<int, 5>& ranks
    ) {
        return packed_score(category, ranks);
    }

    std::string to_string(HandCategory category) {
    switch (category) {
        case HandCategory::HighCard:
            return "High Card";

        case HandCategory::OnePair:
            return "One Pair";

        case HandCategory::TwoPair:
            return "Two Pair";

        case HandCategory::ThreeOfAKind:
            return "Three Of A Kind";

        case HandCategory::Straight:
            return "Straight";

        case HandCategory::Flush:
            return "Flush";

        case HandCategory::FullHouse:
            return "Full House";

        case HandCategory::FourOfAKind:
            return "Four Of A Kind";

        case HandCategory::StraightFlush:
            return "Straight Flush";
        }
        return "unknown";
    }

    std::string to_string(const HandStrength& strength) {
        std::string result = to_string(strength.category);
        result += ":";

        for (std::size_t i = 0; i < strength.ranks.size(); ++i) {
            if (i > 0) {
                result += ",";
            }

            result += std::to_string(strength.ranks[i]);
        }

        return result;
    }
}
