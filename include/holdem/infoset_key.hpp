#pragma once

#include "game.hpp"

#include "board_abstraction.hpp"
#include "hand_abstraction.hpp"
#include "private_state.hpp"
#include "public_state.hpp"
#include "street.hpp"

#include "poker/board.hpp"
#include "poker/hand.hpp"

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace poker::holdem {

struct InfoSetKey {
    Player player = Player::Terminal;
    Street street = Street::River;
    Board board;
    BoardBucketId board_bucket = kInvalidBoardBucket;
    HandBucketId private_hand_bucket = kInvalidHandBucket;
    std::vector<Action> public_action_history;
    int pot = 0;
    int p0_stack = 0;
    int p1_stack = 0;
    int p0_committed_this_round = 0;
    int p1_committed_this_round = 0;
    int current_bet_to_call = 0;
    int num_raises_this_street = 0;
    bool operator==(const InfoSetKey& other) const {
        return player == other.player &&
               street == other.street &&
               board == other.board &&
               board_bucket == other.board_bucket &&
               private_hand_bucket == other.private_hand_bucket &&
               public_action_history == other.public_action_history &&
               pot == other.pot &&
               p0_stack == other.p0_stack &&
               p1_stack == other.p1_stack &&
               p0_committed_this_round == other.p0_committed_this_round &&
               p1_committed_this_round == other.p1_committed_this_round &&
               current_bet_to_call == other.current_bet_to_call &&
               num_raises_this_street == other.num_raises_this_street;
    }

    bool operator!=(const InfoSetKey& other) const {
        return !(*this == other);
    }
};

inline std::string action_history_key(
    const std::vector<Action>& actions
) {
    std::string result;
    for (std::size_t i = 0; i < actions.size(); ++i) {
        if (i > 0) {
            result += ",";
        }
        result += action_history_token(actions[i]);
    }
    return result;
}

inline InfoSetKey make_infoset_key(
    Player player,
    const PublicState& public_state,
    const PrivateState& private_state,
    const HandAbstraction& hand_abstraction,
    const BoardAbstraction* board_abstraction = nullptr
) {
    if (player != Player::P0 && player != Player::P1) {
        throw std::invalid_argument(
            "make_infoset_key requires P0 or P1."
        );
    }

    public_state.validate();
    private_state.validate();

    if (public_state.player_to_act != player) {
        throw std::invalid_argument(
            "Infoset key player must match public_state.player_to_act."
        );
    }

    const HoleCards own_hand = private_state.hand_for(player);

    InfoSetKey key;

    key.player = player;
    key.street = public_state.street;
    key.board = public_state.board;

    if (board_abstraction != nullptr && !board_abstraction->is_exact()) {
        key.board_bucket = board_abstraction->bucket_for(
            public_state.street,
            public_state.board
        );
    } else {
        key.board_bucket = kInvalidBoardBucket;
    }

    key.private_hand_bucket = hand_abstraction.bucket_for(
        player,
        own_hand,
        public_state.board,
        public_state.street
    );

    key.public_action_history = public_state.action_history;

    key.pot = public_state.pot;
    key.p0_stack = public_state.p0_stack;
    key.p1_stack = public_state.p1_stack;

    key.p0_committed_this_round =
        public_state.betting.p0_committed_this_round;

    key.p1_committed_this_round =
        public_state.betting.p1_committed_this_round;

    key.current_bet_to_call =
        public_state.betting.current_bet_to_call;

    key.num_raises_this_street =
        public_state.betting.num_raises_this_street;

    return key;
}

inline InfoSetKey make_infoset_key_exact(
    Player player,
    const PublicState& public_state,
    const PrivateState& private_state
) {
    const ExactHandAbstraction exact_hand;
    const ExactBoardAbstraction exact_board;

    return make_infoset_key(
        player,
        public_state,
        private_state,
        exact_hand,
        &exact_board
    );
}

inline std::string to_string(const InfoSetKey& key) {
    std::ostringstream oss;

    oss << "p=" << poker::to_string(key.player)
        << "|st=" << to_string(key.street);

    if (key.board_bucket != kInvalidBoardBucket) {
        oss << "|bbucket=" << key.board_bucket;
    } else {
        oss << "|board=" << poker::to_string(key.board);
    }

    oss << "|hbucket=" << key.private_hand_bucket
        << "|hist=" << action_history_key(key.public_action_history)
        << "|pot=" << key.pot
        << "|p0s=" << key.p0_stack
        << "|p1s=" << key.p1_stack
        << "|p0c=" << key.p0_committed_this_round
        << "|p1c=" << key.p1_committed_this_round
        << "|btc=" << key.current_bet_to_call
        << "|raises=" << key.num_raises_this_street;

    return oss.str();
}

struct InfoSetKeyHash {
    std::size_t operator()(const InfoSetKey& key) const {
        return std::hash<std::string>{}(to_string(key));
    }
};

} // namespace poker::holdem