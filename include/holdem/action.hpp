#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace poker::holdem {

enum class ActionType : int {
    Fold  = 0,
    Check = 1,
    Call  = 2,
    Bet   = 3,
    Raise = 4,
    AllIn = 5
};

struct Action {
    ActionType type = ActionType::Check;

    // Amount means:
    //   Fold:
    //     0
    //   Check:
    //     0
    //   Call:
    //     additional chips put in by the caller
    //   Bet:
    //     total chips put in by bettor on this street
    //   Raise:
    //     total chips committed by raiser on this street,
    //     not the raise increment
    //   AllIn:
    //     total chips committed by actor on this street after action
    int amount = 0;

    Action() = default;
    Action(ActionType action_type, int action_amount): type(action_type), amount(action_amount) {
        validate();
    }

    void validate() const {
        switch (type) {
            case ActionType::Fold:
            case ActionType::Check:
                if (amount != 0) {
                    throw std::invalid_argument(
                        "Fold and Check actions must have amount 0."
                    );
                }
                break;
            case ActionType::Call:
            case ActionType::Bet:
            case ActionType::Raise:
            case ActionType::AllIn:
                if (amount <= 0) {
                    throw std::invalid_argument(
                        "Call, Bet, Raise, and AllIn actions must have positive amount."
                    );
                }
            break;
        }
    }
};

inline Action fold_action() {
    return Action{ActionType::Fold, 0};
}
inline Action check_action() {
    return Action{ActionType::Check, 0};
}
inline Action call_action(int call_amount) {
    return Action{ActionType::Call, call_amount};
}
inline Action bet_action(int bet_amount) {
    return Action{ActionType::Bet, bet_amount};
}
inline Action raise_action(int total_committed_after_raise) {
    return Action{ActionType::Raise, total_committed_after_raise};
}
inline Action all_in_action(int total_committed_after_all_in) {
    return Action{ActionType::AllIn, total_committed_after_all_in};
}

inline bool operator==(const Action& a, const Action& b) {
    return a.type == b.type && a.amount == b.amount;
}
inline bool operator!=(const Action& a, const Action& b) {
    return !(a == b);
}

inline bool is_passive_action(ActionType type) {
    return type == ActionType::Check || type == ActionType::Call;
}
inline bool is_aggressive_action(ActionType type) {
    return type == ActionType::Bet || type == ActionType::Raise || type == ActionType::AllIn;
}

inline bool is_terminal_action_candidate(ActionType type) {
    return type == ActionType::Fold;
}
inline std::string to_string(ActionType type) {
    switch (type) {
        case ActionType::Fold:  return "fold";
        case ActionType::Check: return "check";
        case ActionType::Call:  return "call";
        case ActionType::Bet:   return "bet";
        case ActionType::Raise: return "raise";
        case ActionType::AllIn: return "allin";
    }
    return "unknown";
}

inline std::string to_string(const Action& action) {
    switch (action.type) {
        case ActionType::Fold:
        case ActionType::Check:
            return to_string(action.type);
        case ActionType::Call:
        case ActionType::Bet:
        case ActionType::Raise:
        case ActionType::AllIn:
            return to_string(action.type) + ":" + std::to_string(action.amount);
    }
    return "unknown";
}

inline char action_code(ActionType type) {
    switch (type) {
        case ActionType::Fold:  return 'f';
        case ActionType::Check: return 'x';
        case ActionType::Call:  return 'c';
        case ActionType::Bet:   return 'b';
        case ActionType::Raise: return 'r';
        case ActionType::AllIn: return 'a';
    }

    return '?';
}
inline std::string action_history_token(const Action& action) {
    switch (action.type) {
        case ActionType::Fold:
        case ActionType::Check:
            return std::string(1, action_code(action.type));
        case ActionType::Call:
        case ActionType::Bet:
        case ActionType::Raise:
        case ActionType::AllIn:
            return std::string(1, action_code(action.type)) +
                   std::to_string(action.amount);
    }
    return "?";
}

} // namespace poker::holdem