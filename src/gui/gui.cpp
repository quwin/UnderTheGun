#include <FL/Fl.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Box.H>
#include <FL/fl_ask.H>
#include <FL/x.H>

#ifdef _WIN32
#include <windows.h>
#include <dwmapi.h>
#pragma comment(lib, "dwmapi.lib")
#ifndef DWMWA_CAPTION_COLOR
#define DWMWA_CAPTION_COLOR 35
#endif
#ifndef DWMWA_TEXT_COLOR
#define DWMWA_TEXT_COLOR 36
#endif
#endif

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

// Native solver headers
#include "cfr_cpu.hpp"
#include "game.hpp"
#include "holdem/action.hpp"
#include "holdem/betting_abstraction.hpp"
#include "holdem/subgame_builder.hpp"
#include "holdem/subgame_config.hpp"
#include "poker/board.hpp"
#include "poker/hand.hpp"
#include "poker/range.hpp"
#include "cfr_gpu.hpp"

// GUI components
#include "components/Page1_Settings.hh"
#include "components/Page2_Board.hh"
#include "components/Page3_HeroRange.hh"
#include "components/Page4_VillainRange.hh"
#include "components/Page5_Progress.hh"
#include "components/Page6_Strategy.hh"
#include "components/ComboStrategyDisplay.hh"
#include "utils/RangeData.hh"
#include "utils/MemoryUtil.hh"

namespace {

poker::Board make_board_from_labels(const std::vector<std::string>& labels) {
  std::vector<phevaluator::Card> cards;
  cards.reserve(labels.size());
  for (const std::string& label : labels) {
    cards.emplace_back(label);
  }
  return poker::Board{cards};
}


std::vector<phevaluator::Card> cards_of_suitless_rank(const char rank_char) {
  return {
    phevaluator::Card(std::string(1, rank_char).append("c")),
    phevaluator::Card(std::string(1, rank_char).append("d")),
    phevaluator::Card(std::string(1, rank_char).append("h")),
    phevaluator::Card(std::string(1, rank_char).append("s")),
  };
}

void add_combo_if_legal(poker::Range& range, phevaluator::Card a, phevaluator::Card b, poker::DeckMask dead) {
  if (a == b) return;
  const poker::HoleCards hand{a, b};
  if (poker::masks_overlap(hand.mask(), dead)) return;
  range.set_weight(poker::make_hand(hand), 1.0f);
}

poker::Range make_range_from_hand_labels(
    const std::vector<std::string>& labels,
    const poker::Board& board) {
  poker::Range range;
  range.clear();
  const poker::DeckMask dead = poker::board_mask(board);

  for (const std::string& raw : labels) {
    std::string h = raw;
    std::erase_if(h, ::isspace);
    if (h.empty()) continue;

    // Exact combo: AhKh, AsAd, etc.
    if (h.size() == 4) {
      add_combo_if_legal(range, phevaluator::Card(h.substr(0, 2)), phevaluator::Card(h.substr(2, 2)), dead);
      continue;
    }

    // Grid hand: AA, AKs, AKo, AK.
    if (h.size() < 2 || h.size() > 3) {
      throw std::invalid_argument("Unsupported range token: " + raw);
    }

    const char r1 = h[0];
    const char r2 = h[1];
    const bool pair = (std::toupper(r1) == std::toupper(r2));
    const char suffix = (h.size() == 3 ? static_cast<char>(std::tolower(h[2])) : '\0');
    const std::vector<phevaluator::Card> c1 = cards_of_suitless_rank(r1);
    const std::vector<phevaluator::Card> c2 = cards_of_suitless_rank(r2);

    for (phevaluator::Card a : c1) {
      for (phevaluator::Card b : c2) {
        if (a == b) continue;
        const bool suited = a.describeSuit() == b.describeSuit();
        if (pair && static_cast<int>(a) >= static_cast<int>(b)) continue;
        if (!pair && suffix == 's' && !suited) continue;
        if (!pair && suffix == 'o' && suited) continue;
        add_combo_if_legal(range, a, b, dead);
      }
    }
  }

  if (range.empty()) {
    throw std::invalid_argument("Range contains no legal combos after board blockers.");
  }

  return range;
}

std::string hand_type_from_hole_cards(const poker::HoleCards& hand) {
  const char r1 = hand.a.describeRank();
  const char r2 = hand.b.describeRank();
  const int v1 = hand.a;
  const int v2 = hand.b;

  char hi = r1;
  char lo = r2;
  if (v2 > v1) {
    hi = r2;
    lo = r1;
  }

  if (hi == lo) {
    return std::string{hi, lo};
  }

  const bool suited = hand.a.describeSuit() == hand.b.describeSuit();
  return std::string{hi} + std::string{lo} + (suited ? "s" : "o");
}

std::string combo_label_from_hole_cards(const poker::HoleCards& hand) {
  return hand.a.describeCard() + hand.b.describeCard();
}

std::string action_label(const poker::GameAction& action) {
  using poker::holdem::ActionType;
  switch (static_cast<ActionType>(action.action_type)) {
    case ActionType::Fold: return "Fold";
    case ActionType::Check: return "Check";
    case ActionType::Call: return "Call " + std::to_string(action.amount);
    case ActionType::Bet: return "Bet " + std::to_string(action.amount);
    case ActionType::Raise: return "Raise to " + std::to_string(action.amount);
    case ActionType::AllIn: return "All-in " + std::to_string(action.amount);
  }
  return poker::to_string(action);
}

Fl_Color color_for_action(const poker::GameAction& action, int& bet_index) {
  using poker::holdem::ActionType;
  switch (static_cast<ActionType>(action.action_type)) {
    case ActionType::Fold: return fl_rgb_color(91, 141, 238);
    case ActionType::Check:
    case ActionType::Call: return fl_rgb_color(94, 186, 125);
    case ActionType::Bet:
    case ActionType::Raise:
    case ActionType::AllIn: {
      switch (bet_index++) {
        case 0: return fl_rgb_color(245, 166, 35);
        case 1: return fl_rgb_color(224, 124, 84);
        case 2: return fl_rgb_color(214, 100, 95);
        default: return fl_rgb_color(196, 69, 105);
      }
    }
  }
  return FL_GRAY;
}

const poker::HandDomain& acting_hand_domain(
      const poker::Game& game,
      poker::Player player
  ) {
  return game.hand_domain(player);
}

poker::HandId hand_id_for_bucket(
      const poker::Game& game,
      poker::Player player,
      int bucket
) {
  const poker::HandDomain& domain = acting_hand_domain(game, player);

  if (bucket < 0 || bucket >= domain.hand_count()) {
    throw std::out_of_range("hand bucket out of range");
  }

  return domain.hands[static_cast<std::size_t>(bucket)];
}

} // namespace

class Wizard : public Fl_Double_Window {
  struct UserInputs {
    int stackSize{}, startingPot{}, minBet{}, iterations{};
    float min_exploitability{};
    std::string potType, yourPos, theirPos;
    std::vector<std::string> board;
    std::vector<std::string> heroRange;
    std::vector<std::string> villainRange;
    bool autoImportRanges{true};
    bool forceDonkCheck{true};
  } m_data;

  // Native solver state. Hero is P0; Villain is P1.
  std::unique_ptr<poker::Game> m_game;
  std::vector<float> m_average_strategy;
  std::vector<int> m_current_nodes;

  int m_current_pot{};
  int m_p1_stack{};
  int m_p2_stack{};
  int m_p1_wager{};
  int m_p2_wager{};

  Page1_Settings *m_pg1{};
  Page2_Board *m_pg2{};
  Page3_HeroRange *m_pg3{};
  Page4_VillainRange *m_pg4{};
  Page5_Progress *m_pg5{};
  Page6_Strategy *m_pg6{};

  struct GameState {
    std::vector<int> nodes;
    int p1_stack;
    int p2_stack;
    int current_pot;
    int p1_wager;
    int p2_wager;
    std::vector<std::string> board;
  };
  std::vector<GameState> m_history;

  std::map<std::string, std::vector<ComboStrategyDisplay::ComboStrategy>> m_overallStrategyCache;

  struct SolveParams {
    int stackSize{0}, startingPot{0}, minBet{0}, iterations{0};
    float minExploitability{0};
    std::string potType, yourPos, theirPos;
    std::vector<std::string> board;
    std::vector<std::string> heroRange;
    std::vector<std::string> villainRange;

    bool operator==(const SolveParams& other) const {
      return stackSize == other.stackSize &&
             startingPot == other.startingPot &&
             minBet == other.minBet &&
             iterations == other.iterations &&
             minExploitability == other.minExploitability &&
             potType == other.potType &&
             yourPos == other.yourPos &&
             theirPos == other.theirPos &&
             board == other.board &&
             heroRange == other.heroRange &&
             villainRange == other.villainRange;
    }
  };
  SolveParams m_lastSolveParams;
  bool m_hasCachedSolve{false};

  static void cb1Next(Fl_Widget *, void *d) { static_cast<Wizard *>(d)->do1Next(); }
  static void cb2Back(Fl_Widget *, void *d) { static_cast<Wizard *>(d)->doBack2(); }
  static void cb2Next(Fl_Widget *, void *d) { static_cast<Wizard *>(d)->do2Next(); }
  static void cb3Back(Fl_Widget *, void *d) { static_cast<Wizard *>(d)->doBack3(); }
  static void cb3Next(Fl_Widget *, void *d) { static_cast<Wizard *>(d)->do3Next(); }
  static void cb4Back(Fl_Widget *, void *d) { static_cast<Wizard *>(d)->doBack4(); }
  static void cb4Next(Fl_Widget *, void *d) { static_cast<Wizard *>(d)->do4Next(); }

  void do1Next() {
    std::string error = m_pg1->validateInputs();
    if (!error.empty()) {
      fl_message("%s", error.c_str());
      return;
    }

    std::string yourPos = m_pg1->getYourPosition();
    std::string theirPos = m_pg1->getTheirPosition();
    if (yourPos == theirPos) {
      fl_message("Your Position and Their Position must be different.");
      return;
    }

    m_data.stackSize = m_pg1->getStackSize();
    m_data.startingPot = m_pg1->getStartingPot();
    m_data.minBet = m_pg1->getMinBet();
    m_data.potType = m_pg1->getPotType();
    m_data.iterations = m_pg1->getIterations();
    m_data.yourPos = yourPos;
    m_data.theirPos = theirPos;
    m_data.autoImportRanges = m_pg1->getAutoImport();
    m_data.forceDonkCheck = m_pg1->getForceDonkCheck();

    m_pg1->stopAnimation();
    m_pg1->hide();
    m_pg2->show();
  }

  void doBack2() {
    m_pg3->clearSelection();
    m_pg4->clearSelection();
    m_pg2->hide();
    m_pg1->show();
  }

  void do2Next() {
    auto selectedCards = m_pg2->getSelectedCards();
    if (selectedCards.size() < 3 || selectedCards.size() > 5) {
      fl_message("Select 3-5 cards.");
      return;
    }

    m_data.board = selectedCards;
    m_pg3->clearSelection();
    m_pg2->hide();
    m_pg3->show();

    if (m_data.autoImportRanges) {
      auto range = RangeData::getRangeForPosition(m_data.yourPos, m_data.potType, true);
      m_pg3->setSelectedRange(range);
    }
  }

  void doBack3() {
    m_pg3->hide();
    m_pg2->show();
  }

  void do3Next() {
    m_data.heroRange = m_pg3->getSelectedRange();
    if (m_data.heroRange.empty()) {
      fl_message("Please select at least one hand for your range.");
      return;
    }

    m_pg3->hide();
    m_pg4->show();

    if (m_data.autoImportRanges) {
      auto range = RangeData::getRangeForPosition(m_data.theirPos, m_data.potType, false);
      m_pg4->setSelectedRange(range);
    }
  }

  void doBack4() {
    m_pg4->hide();
    m_pg3->show();
    if (m_data.autoImportRanges) {
      auto range = RangeData::getRangeForPosition(m_data.yourPos, m_data.potType, true);
      m_pg3->setSelectedRange(range);
    }
  }

  void do4Next() {
    m_data.villainRange = m_pg4->getSelectedRange();
    if (m_data.villainRange.empty()) {
      fl_message("Please select at least one hand for the villain's range.");
      return;
    }

    SolveParams currentParams;
    currentParams.stackSize = m_data.stackSize;
    currentParams.startingPot = m_data.startingPot;
    currentParams.minBet = m_data.minBet;
    currentParams.iterations = m_data.iterations;
    currentParams.minExploitability = m_data.min_exploitability;
    currentParams.potType = m_data.potType;
    currentParams.yourPos = m_data.yourPos;
    currentParams.theirPos = m_data.theirPos;
    currentParams.board = m_data.board;
    currentParams.heroRange = m_data.heroRange;
    currentParams.villainRange = m_data.villainRange;

    if (m_hasCachedSolve && currentParams == m_lastSolveParams && m_game) {
      reset_navigation_to_root_public_state();
      m_history.clear();
      m_overallStrategyCache.clear();
      m_pg4->hide();
      m_pg6->show();
      updateStrategyDisplay();
      Fl::check();
      return;
    }

    m_pg4->hide();
    m_pg5->show();
    Fl::check();

    try {
      runTraining();
      m_lastSolveParams = currentParams;
      m_hasCachedSolve = true;
    } catch (const std::exception& e) {
      fl_alert("Solve failed:\n\n%s", e.what());
      m_pg5->hide();
      m_pg4->show();
    }
  }

  [[nodiscard]] poker::holdem::BettingAbstraction make_gui_betting_abstraction() const {
    poker::holdem::BettingAbstraction abstraction = poker::holdem::make_standard_abstraction();
    if (const float min_bet_ratio = static_cast<float>(m_data.minBet)/static_cast<float>(m_data.startingPot); abstraction.first_bet_sizes.front().value < min_bet_ratio) {
      abstraction.first_bet_sizes[0] = poker::holdem::BetSize::pot_fraction(min_bet_ratio);
    }
    abstraction.validate();
    return abstraction;
  }

  [[nodiscard]] poker::Player initial_player_to_act() const {
    const int heroPos = RangeData::getPositionIndex(m_data.yourPos);
    const int villainPos = RangeData::getPositionIndex(m_data.theirPos);
    const bool hero_is_ip = heroPos > villainPos;
    return hero_is_ip ? poker::Player::P1 : poker::Player::P0;
  }

  void runTraining() {
    m_pg5->setStatus("Parsing board and ranges...");
    Fl::check();

    const poker::Board board = make_board_from_labels(m_data.board);
    const poker::Range hero_range = make_range_from_hand_labels(m_data.heroRange, board);
    const poker::Range villain_range = make_range_from_hand_labels(m_data.villainRange, board);

    poker::holdem::HoldemSubgameConfig config;
    config.board = board;
    config.pot_size = m_data.startingPot;
    config.effective_stack = m_data.stackSize;
    config.player_to_act = initial_player_to_act();
    config.p0_range = hero_range;
    config.p1_range = villain_range;
    config.betting_abstraction = make_gui_betting_abstraction();
    config.collapse_all_in_runouts_to_ev = true;
    config.board_abstraction = poker::holdem::make_isomorphic_board_abstraction(config.p0_range, config.p1_range);

    const std::size_t availableMemory = MemoryUtil::getAvailableMemory();
    m_pg5->setMemoryEstimate( sizeof(poker::Game) + config.memoryEstimate(), availableMemory);
    Fl::check();

    if (!m_pg5->isMemoryOk()) {
      throw std::runtime_error("Not enough memory for this solve. Try reducing range sizes or solving from a later street.");
    }

    m_pg5->setStatus("Building native Hold'em subgame tree...");
    Fl::check();

    poker::Game built = poker::holdem::HoldemSubgameBuilder(config).build();

    m_pg5->setStatus("Training native CFR solver...");
    m_pg5->reset();
    Fl::check();

    m_game = std::make_unique<poker::Game>(std::move(built));

    const int total = std::max<int>(0, m_data.iterations);
    const int chunk = std::max<int>(1, total / 100);

    if (std::string(m_pg1->getCFRRenderer()) == "GPU") {
        poker::GpuCfrConfig cfr_config;
        cfr_config.num_players = 2;
        cfr_config.synchronize_each_iteration = false;
        cfr_config.threads_per_block = 512;
        cfr_config.use_cfr_plus = false;
        cfr_config.linear_averaging = false;
        cfr_config.terminal_mode = poker::GpuTerminalMode::HostPrecomputed;

        poker::GpuCfrSolver solver(*m_game, cfr_config);
        for (int done = 0; done < total;) {
            const int step = std::min<int>(chunk, total - done);

            solver.run_iterations(step);

            done += step;
            m_pg5->setIteration(done, total);
            m_pg5->setProgress(done, total);
            Fl::check();
        }

        m_average_strategy = solver.average_strategy();
    } else {
        poker::CfrConfig cfr_config;
        cfr_config.num_players = 2;
        cfr_config.use_cfr_plus = false;
        cfr_config.linear_averaging = false;
        cfr_config.simultaneous_updates = true;

        poker::TerminalValueProvider terminal_values;

        poker::CpuCfrSolver solver(
            *m_game,
            terminal_values,
            cfr_config
        );

        for (int done = 0; done < total;) {
            const int step = std::min<int>(chunk, total - done);

            solver.run_iterations(step);

            done += step;
            m_pg5->setIteration(done, total);
            m_pg5->setProgress(done, total);
            Fl::check();
        }

        m_average_strategy = solver.average_strategy();
    }

    {
        std::ofstream log("under_the_gun_gui_debug.log", std::ios::app);
        log << "=== GUI native solve complete ===\n";
        log << "Nodes: " << m_game->num_nodes() << "\n";
        log << "Edges: " << m_game->num_edges() << "\n";
        log << "Actions: " << m_game->num_actions() << "\n";
        log << "Action states: " << m_game->num_action_states() << "\n";
        log << "CFR tensor entries: " << m_game->cfr_tensor_entries() << "\n";
        log << "State-bucket entries: " << m_game->state_bucket_entries() << "\n";
        log << "Hand pairs: " << m_game->hand_pairs.pair_count() << "\n";
        log << "Iterations: " << total << "\n";
        log << "Board: ";
        for (const auto& card : m_data.board) {
            log << card << ' ';
        }
        log << "\n\n";
    }

    reset_navigation_to_root_public_state();
    m_history.clear();
    m_overallStrategyCache.clear();

    m_pg5->hide();
    m_pg6->show();
    updateStrategyDisplay();
    Fl::check();
  }

  void reset_navigation_to_root_public_state() {
    m_current_pot = m_data.startingPot;
    m_p1_stack = m_p2_stack = m_data.stackSize;
    m_p1_wager = m_p2_wager = 0;
    m_current_nodes.clear();
    if (!m_game) {
      return;
    }
    m_current_nodes.push_back(m_game->root);
  }

  std::vector<int> dedupe_nodes(const std::vector<int>& nodes) const {
    std::vector<int> result;
    std::unordered_set<int> seen;
    for (int id : nodes) {
      if (seen.insert(id).second) result.push_back(id);
    }
    return result;
  }

  std::vector<int> current_action_nodes() const {
    std::vector<int> result;
    if (!m_game) {
      return result;
    }
    for (int node_id : m_current_nodes) {
      const poker::PublicNode& node = m_game->node(node_id);
      if (node.type == poker::PublicNodeType::Action &&
          (node.player == poker::Player::P0 ||
          node.player == poker::Player::P1) &&
          node.action_state_index >= 0) {
        result.push_back(node_id);
      }
    }
    return result;
  }

  std::vector<int> current_chance_nodes() const {
    std::vector<int> result;

    if (!m_game) {
      return result;
    }

    for (int node_id : m_current_nodes) {
      const poker::PublicNode& node = m_game->node(node_id);

      if (node.type == poker::PublicNodeType::Chance &&
          node.player == poker::Player::Chance) {
        result.push_back(node_id);
      }
    }

    return result;
  }

  bool current_is_terminal_only() const {
    if (!m_game || m_current_nodes.empty()) {
      return false;
    }

    for (int node_id : m_current_nodes) {
      if (m_game->node(node_id).type != poker::PublicNodeType::Terminal) {
        return false;
      }
    }

    return true;
  }
  std::vector<poker::GameAction> representative_actions() const {
    const auto nodes = current_action_nodes();
    if (nodes.empty() || !m_game) {
      return {};
    }
    const poker::PublicNode& node = m_game->node(nodes.front());
    if (node.action_state_index < 0) {
      return {};
    }
    const poker::ActionState& state = m_game->action_state(node.action_state_index);
    std::vector<poker::GameAction> result;
    result.reserve(static_cast<std::size_t>(state.action_count));
    for (int a = 0; a < state.action_count; ++a) {
      result.push_back(m_game->action(state.first_action + a));
    }
    return result;
  }

  void updatePotAndStacks(const poker::GameAction &action, poker::Player player) {
    using poker::holdem::ActionType;
    int &current_wager = (player == poker::Player::P0) ? m_p1_wager : m_p2_wager;
    const int &other_wager = (player == poker::Player::P0) ? m_p2_wager : m_p1_wager;
    int &current_stack = (player == poker::Player::P0) ? m_p1_stack : m_p2_stack;
    int &other_stack = (player == poker::Player::P0) ? m_p2_stack : m_p1_stack;

    switch (const auto type = static_cast<ActionType>(action.action_type)) {
      case ActionType::Fold:
        other_stack += m_current_pot + current_wager + other_wager;
        m_current_pot = 0;
        m_p1_wager = m_p2_wager = 0;
        break;
      case ActionType::Check:
        if (current_wager == other_wager && current_wager > 0) {
          m_current_pot += current_wager + other_wager;
          m_p1_wager = m_p2_wager = 0;
        }
        break;
      case ActionType::Call: {
        const int call_amount = std::max<int>(0, other_wager - current_wager);
        current_stack -= call_amount;
        current_wager = other_wager;
        m_current_pot += current_wager + other_wager;
        m_p1_wager = m_p2_wager = 0;
        break;
      }
      case ActionType::Bet:
      case ActionType::AllIn:
        current_stack -= std::max<int>(0, action.amount - current_wager);
        current_wager = action.amount;
        break;
      case ActionType::Raise:
        current_stack -= std::max<int>(0, action.amount - current_wager);
        current_wager = action.amount;
        break;
    }
  }

  void set_common_strategy_header() const {
    std::string board = "Board: ";
    for (const auto &card : m_data.board) board += card + " ";
    m_pg6->setBoardInfo(board);

    const int effectivePot = m_current_pot + m_p1_wager + m_p2_wager;
    std::string info = "Hero: " + std::to_string(m_p1_stack) +
                       " | Villain: " + std::to_string(m_p2_stack) +
                       " | Pot: " + std::to_string(effectivePot);
    m_pg6->setPotInfo(info);
  }

  void updateStrategyDisplay() const {
    if (!m_game) return;

    if (current_is_terminal_only()) {
      m_pg6->setTitle("Terminal Node - Hand Complete");
      m_pg6->showStrategyGrid(false);
      m_pg6->showAnalysisPanel(false);
      m_pg6->showCardSelection(false);
      m_pg6->setActions({});
      set_common_strategy_header();
      return;
    }

    const auto chance_nodes = current_chance_nodes();
    if (!chance_nodes.empty() && current_action_nodes().empty()) {
      std::string prompt = "Select ";
      prompt += (m_data.board.size() == 3 ? "Turn" : "River");
      prompt += " Card";
      m_pg6->setTitle(prompt);
      m_pg6->showStrategyGrid(false);
      m_pg6->showAnalysisPanel(false);
      m_pg6->setActions({});

      std::set<std::string> available;

      for (int node_id : chance_nodes) {
        const poker::PublicNode& chance = m_game->node(node_id);
        for (int e = 0; e < chance.edge_count; ++e) {
          if (const poker::NodeEdge& edge = m_game->edge(chance.first_edge + e); edge.public_card >= 0) {
            const phevaluator::Card card(edge.public_card);
            available.insert(card.describeCard());
          }
        }
      }
      m_pg6->populateCardChoices(std::vector<std::string>(available.begin(), available.end()));
      m_pg6->showCardSelection(true);
      set_common_strategy_header();
      return;
    }

    const auto action_nodes = current_action_nodes();
    if (action_nodes.empty()) {
        return;
    }
    const poker::Player player = m_game->node(action_nodes.front()).player;
    m_pg6->setTitle(
        player == poker::Player::P0
            ? "Hero's Turn"
            : "Villain's Turn"
    );

    set_common_strategy_header();
    const auto actions = representative_actions();
    std::map<std::string, std::vector<float>> handTypeStrategies;
    std::map<std::string, int> handTypeCounts;
    for (int node_id : action_nodes) {
        const poker::PublicNode& node = m_game->node(node_id);
        if (node.action_state_index < 0) {
            continue;
        }
        const poker::ActionState& state = m_game->action_state(node.action_state_index);
        for (int bucket = 0; bucket < state.bucket_count; ++bucket) {
            const poker::HandId hand_id = hand_id_for_bucket(*m_game, player, bucket);
            const poker::HoleCards hand = poker::hand_from_id(hand_id);

            const std::string hand_type = hand_type_from_hole_cards(hand);
            if (!handTypeStrategies.contains(hand_type)) {
                handTypeStrategies[hand_type] = std::vector<float>(actions.size(), 0.0f);
                handTypeCounts[hand_type] = 0;
            }

            for (int a = 0; a < state.action_count; ++a) {
                const std::size_t idx = state.tensor_index(bucket, a);

                if (idx < m_average_strategy.size()) {
                    handTypeStrategies[hand_type][static_cast<std::size_t>(a)] += m_average_strategy[idx];
                }
            }
            ++handTypeCounts[hand_type];
        }
    }

    std::map<std::string, std::map<std::string, float>> strategyMap;
    for (const auto& [hand, stratVec] : handTypeStrategies) {
        const int count = std::max<int>(1, handTypeCounts[hand]);
        std::map<std::string, float> actionProbs;
        for (std::size_t i = 0; i < actions.size(); ++i) {
            const float prob = stratVec[i] / static_cast<float>(count);
            if (prob > 0.001f) {
                actionProbs[action_label(actions[i])] = prob;
            }
        }
        strategyMap[hand] = actionProbs;
    }

    m_pg6->updateStrategyGrid(strategyMap);
    m_pg6->showStrategyGrid(true);
    m_pg6->showAnalysisPanel(true);
    m_pg6->showCardSelection(false);
    m_pg6->deselectHand();

    std::vector<std::string> actionLabels;

    for (const auto& action : actions) {
        actionLabels.push_back(action_label(action));
    }
    m_pg6->setActions(actionLabels);
  }

  void doAction(const std::string& actionStr) {
    if (!m_game) {
      return;
    }

    const auto action_nodes = current_action_nodes();

    if (action_nodes.empty()) {
      return;
    }

    const auto actions = representative_actions();

    int action_idx = -1;

    for (std::size_t i = 0; i < actions.size(); ++i) {
      if (action_label(actions[i]) == actionStr) {
        action_idx = static_cast<int>(i);
        break;
      }
    }

    if (action_idx < 0) {
      return;
    }

    GameState state{
      m_current_nodes,
      m_p1_stack,
      m_p2_stack,
      m_current_pot,
      m_p1_wager,
      m_p2_wager,
      m_data.board
  };

    m_history.push_back(state);

    updatePotAndStacks(
        actions[static_cast<std::size_t>(action_idx)],
        m_game->node(action_nodes.front()).player
    );

    std::vector<int> next;

    for (int node_id : action_nodes) {
      const poker::PublicNode& node = m_game->node(node_id);

      for (int e = 0; e < node.edge_count; ++e) {
        const poker::NodeEdge& edge =
            m_game->edge(node.first_edge + e);

        if (edge.local_action == action_idx) {
          next.push_back(edge.child);
        }
      }
    }

    m_current_nodes = dedupe_nodes(next);
    updateStrategyDisplay();
  }

  void doBack6() {
    m_current_nodes.clear();
    m_p1_stack = m_data.stackSize;
    m_p2_stack = m_data.stackSize;
    m_current_pot = m_data.startingPot;
    m_p1_wager = m_p2_wager = 0;
    m_history.clear();
    m_pg6->hide();
    m_pg4->show();
  }

  void doCardSelected(const std::string& cardStr) {
    if (!m_game) {
      return;
    }

    const auto chance_nodes = current_chance_nodes();

    if (chance_nodes.empty()) {
      return;
    }

    const phevaluator::Card selected_card(cardStr);
    const int selected_id = static_cast<int>(selected_card);

    GameState state{
      m_current_nodes,
      m_p1_stack,
      m_p2_stack,
      m_current_pot,
      m_p1_wager,
      m_p2_wager,
      m_data.board
  };

    m_history.push_back(state);

    std::vector<int> next;

    for (int node_id : chance_nodes) {
      const poker::PublicNode& chance = m_game->node(node_id);

      for (int e = 0; e < chance.edge_count; ++e) {
        const poker::NodeEdge& edge =
            m_game->edge(chance.first_edge + e);

        if (edge.public_card == selected_id) {
          next.push_back(edge.child);
        }
      }
    }

    if (next.empty()) {
      m_history.pop_back();
      return;
    }

    m_data.board.push_back(cardStr);
    m_current_nodes = dedupe_nodes(next);

    updateStrategyDisplay();
  }

  void doUndo() {
    if (m_history.empty()) return;
    auto state = m_history.back();
    m_history.pop_back();

    m_current_nodes = state.nodes;
    m_p1_stack = state.p1_stack;
    m_p2_stack = state.p2_stack;
    m_current_pot = state.current_pot;
    m_p1_wager = state.p1_wager;
    m_p2_wager = state.p2_wager;
    m_data.board = state.board;
    updateStrategyDisplay();
  }

void handleHandSelect(const std::string& hand) {
    if (!m_game) {
        return;
    }

    const auto action_nodes = current_action_nodes();

    if (action_nodes.empty()) {
        return;
    }

    const auto actions = representative_actions();

    if (actions.empty()) {
        showOverallStrategyWithHeader("No actions available");
        return;
    }

    const poker::Player player =
        m_game->node(action_nodes.front()).player;

    std::vector<Fl_Color> actionColors;
    actionColors.reserve(actions.size());

    int betIndex = 0;

    for (const auto& action : actions) {
        actionColors.push_back(color_for_action(action, betIndex));
    }

    std::map<std::string, std::vector<double>> combo_action_sum;
    std::map<std::string, int> combo_count;

    for (int node_id : action_nodes) {
        const poker::PublicNode& node = m_game->node(node_id);

        if (node.action_state_index < 0) {
            continue;
        }

        const poker::ActionState& state =
            m_game->action_state(node.action_state_index);

        for (int bucket = 0; bucket < state.bucket_count; ++bucket) {
            const poker::HandId hand_id =
                hand_id_for_bucket(*m_game, player, bucket);

            const poker::HoleCards hole =
                poker::hand_from_id(hand_id);

            if (hand_type_from_hole_cards(hole) != hand) {
                continue;
            }

            const std::string combo_label =
                combo_label_from_hole_cards(hole);

            auto& sums = combo_action_sum[combo_label];

            if (sums.empty()) {
                sums.assign(actions.size(), 0.0);
            }

            for (int a = 0; a < state.action_count; ++a) {
                const std::size_t idx =
                    state.tensor_index(bucket, a);

                const float prob =
                    idx < m_average_strategy.size()
                        ? m_average_strategy[idx]
                        : 0.0f;

                sums[static_cast<std::size_t>(a)] +=
                    static_cast<double>(prob);
            }

            ++combo_count[combo_label];
        }
    }

    std::vector<ComboStrategyDisplay::ComboStrategy> combos;
    combos.reserve(combo_action_sum.size());

    for (const auto& entry : combo_action_sum) {
        const std::string& combo_label = entry.first;
        const std::vector<double>& sums = entry.second;

        const auto count_it = combo_count.find(combo_label);

        if (count_it == combo_count.end() || count_it->second <= 0) {
            continue;
        }

        const int count = count_it->second;

        ComboStrategyDisplay::ComboStrategy comboStrat;
        comboStrat.combo = combo_label;

        for (std::size_t a = 0;
             a < actions.size() && a < sums.size();
             ++a) {
            const float prob =
                static_cast<float>(
                    sums[a] / static_cast<double>(count)
                );

            if (prob <= 0.001f) {
                continue;
            }

            ComboStrategyDisplay::ActionProb ap;
            ap.name = action_label(actions[a]);
            ap.prob = prob;
            ap.color = actionColors[a];

            comboStrat.actions.push_back(ap);
        }

        if (!comboStrat.actions.empty()) {
            combos.push_back(comboStrat);
        }
    }

    if (combos.empty()) {
        showOverallStrategyWithHeader("Hand not in range");
        return;
    }

    m_pg6->setComboStrategies(hand, combos);
}
  std::string generateRangeString() {
    if (!m_game) {
      return "";
    }

    const auto action_nodes = current_action_nodes();

    if (action_nodes.empty()) {
      return "";
    }

    const poker::Player player =
        m_game->node(action_nodes.front()).player;

    std::set<std::string> handTypes;

    for (int node_id : action_nodes) {
      const poker::PublicNode& node = m_game->node(node_id);

      if (node.action_state_index < 0) {
        continue;
      }

      const poker::ActionState& state =
          m_game->action_state(node.action_state_index);

      for (int bucket = 0; bucket < state.bucket_count; ++bucket) {
        const poker::HandId hand_id =
            hand_id_for_bucket(*m_game, player, bucket);

        handTypes.insert(
            hand_type_from_hole_cards(
                poker::hand_from_id(hand_id)
            )
        );
      }
    }

    std::stringstream result;
    bool first = true;

    for (const auto& hand : handTypes) {
      if (!first) {
        result << ",";
      }

      first = false;
      result << hand;
    }

    return result.str();
  }

  void showOverallStrategyWithHeader(const std::string& header) {
    const std::string key = navigation_cache_key();
    auto cacheIt = m_overallStrategyCache.find(key);
    if (cacheIt == m_overallStrategyCache.end()) {
      computeAndCacheOverallStrategy(key);
      cacheIt = m_overallStrategyCache.find(key);
    }
    if (cacheIt != m_overallStrategyCache.end()) {
      m_pg6->setComboStrategies(header, cacheIt->second);
    }
  }

  std::string navigation_cache_key() const {
    std::stringstream ss;
    for (int id : m_current_nodes) ss << id << ',';
    return ss.str();
  }

  void showOverallStrategy() {
    const std::string key = navigation_cache_key();
    auto cacheIt = m_overallStrategyCache.find(key);
    if (cacheIt == m_overallStrategyCache.end()) {
      computeAndCacheOverallStrategy(key);
      cacheIt = m_overallStrategyCache.find(key);
    }
    if (cacheIt != m_overallStrategyCache.end()) {
      m_pg6->setOverallStrategy(cacheIt->second);
    }
  }

  void computeAndCacheOverallStrategy(const std::string& key) {
    if (!m_game) {
        return;
    }

    const auto action_nodes = current_action_nodes();
    const auto actions = representative_actions();

    if (action_nodes.empty() || actions.empty()) {
        return;
    }

    std::vector<float> overallProbs(actions.size(), 0.0f);
    int validBuckets = 0;

    for (int node_id : action_nodes) {
        const poker::PublicNode& node = m_game->node(node_id);

        if (node.action_state_index < 0) {
            continue;
        }

        const poker::ActionState& state =
            m_game->action_state(node.action_state_index);

        for (int bucket = 0; bucket < state.bucket_count; ++bucket) {
            for (int a = 0; a < state.action_count; ++a) {
                const std::size_t idx =
                    state.tensor_index(bucket, a);

                if (idx < m_average_strategy.size()) {
                    overallProbs[static_cast<std::size_t>(a)] +=
                        m_average_strategy[idx];
                }
            }

            ++validBuckets;
        }
    }

    if (validBuckets > 0) {
        for (float& p : overallProbs) {
            p /= static_cast<float>(validBuckets);
        }
    }

    std::vector<Fl_Color> actionColors;
    int betIndex = 0;

    for (const auto& action : actions) {
        actionColors.push_back(color_for_action(action, betIndex));
    }

    std::vector<ComboStrategyDisplay::ComboStrategy> overall;

    ComboStrategyDisplay::ComboStrategy overallStrat;
    overallStrat.combo = "Range Average";

    for (std::size_t a = 0; a < actions.size(); ++a) {
        if (overallProbs[a] > 0.001f) {
            ComboStrategyDisplay::ActionProb ap;
            ap.name = action_label(actions[a]);
            ap.prob = overallProbs[a];
            ap.color = actionColors[a];

            overallStrat.actions.push_back(ap);
        }
    }

    if (!overallStrat.actions.empty()) {
        overall.push_back(overallStrat);
    }

    m_overallStrategyCache[key] = overall;
}
public:
  Wizard(const char *L = 0) : Fl_Double_Window(100, 100, L) { init(); }

private:
  void init() {
    int sx, sy, sw, sh;
    Fl::screen_work_area(sx, sy, sw, sh, 0);
    int new_w = static_cast<int>(sw * 0.8);
    int new_h = static_cast<int>(sh * 0.8);
    size(new_w, new_h);
    position((sw - new_w) / 2 + sx, (sh - new_h) / 2 + sy);
    size_range(650, 550);
    border(1);

#ifdef _WIN32
    HICON hIcon = LoadIcon(GetModuleHandle(NULL), MAKEINTRESOURCE(1));
    if (hIcon) icon(hIcon);
#endif

    m_pg1 = new Page1_Settings(0, 0, new_w, new_h);
    m_pg1->setNextCallback(cb1Next, this);

    m_pg2 = new Page2_Board(0, 0, new_w, new_h);
    m_pg2->setBackCallback(cb2Back, this);
    m_pg2->setNextCallback(cb2Next, this);
    m_pg2->hide();

    m_pg3 = new Page3_HeroRange(0, 0, new_w, new_h);
    m_pg3->setBackCallback(cb3Back, this);
    m_pg3->setNextCallback(cb3Next, this);
    m_pg3->hide();

    m_pg4 = new Page4_VillainRange(0, 0, new_w, new_h);
    m_pg4->setBackCallback(cb4Back, this);
    m_pg4->setNextCallback(cb4Next, this);
    m_pg4->hide();

    m_pg5 = new Page5_Progress(0, 0, new_w, new_h);
    m_pg5->hide();

    m_pg6 = new Page6_Strategy(0, 0, new_w, new_h);
    m_pg6->setActionCallback([this](const std::string &action) { doAction(action); });
    m_pg6->setBackCallback([this]() { doBack6(); });
    m_pg6->setUndoCallback([this]() { doUndo(); });
    m_pg6->setHandSelectCallback([this](const std::string &hand) { handleHandSelect(hand); });
    m_pg6->setCardSelectedCallback([this](const std::string &card) { doCardSelected(card); });
    m_pg6->setCopyRangeCallback([this]() { return generateRangeString(); });
    m_pg6->setShowOverallStrategyCallback([this]() { showOverallStrategy(); });
    m_pg6->hide();

    resizable(this);
    end();
  }
};

int main(int argc, char **argv) {
#ifdef _WIN32
  Fl::set_font(FL_HELVETICA, "Segoe UI");
  Fl::set_font(FL_HELVETICA_BOLD, "Segoe UI Bold");
#elif defined(__APPLE__)
  Fl::set_font(FL_HELVETICA, "Helvetica Neue");
  Fl::set_font(FL_HELVETICA_BOLD, "Helvetica Neue Bold");
#else
  Fl::set_font(FL_HELVETICA, "DejaVu Sans");
  Fl::set_font(FL_HELVETICA_BOLD, "DejaVu Sans Bold");
#endif

  Fl::background(20, 40, 80);
  Fl::foreground(255, 255, 255);
  Fl::background2(25, 50, 90);

  fl_message_font(FL_HELVETICA, FL_NORMAL_SIZE * 2);
  fl_message_hotspot(1);
  Wizard wiz("Under The Gun: PostFlop Poker Solver");
  wiz.show(argc, argv);

#ifdef _WIN32
  HWND hwnd = fl_xid(&wiz);
  if (hwnd) {
    COLORREF captionColor = RGB(10, 25, 50);
    DwmSetWindowAttribute(hwnd, DWMWA_CAPTION_COLOR, &captionColor, sizeof(captionColor));
    COLORREF textColor = RGB(255, 255, 255);
    DwmSetWindowAttribute(hwnd, DWMWA_TEXT_COLOR, &textColor, sizeof(textColor));
  }
#endif

  return Fl::run();
}
