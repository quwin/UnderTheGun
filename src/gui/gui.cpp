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
#include <cmath>
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
#include "holdem/street.hpp"
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

static const std::vector<std::string> RANKS = {
    "A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"};
static const std::vector<char> SUITS = {'h', 'd', 'c', 's'};

namespace {

poker::Board make_board_from_labels(const std::vector<std::string>& labels) {
  std::vector<phevaluator::Card> cards;
  cards.reserve(labels.size());
  for (const std::string& label : labels) {
    cards.emplace_back(label);
  }
  return poker::Board{cards};
}

poker::holdem::Street street_from_board_size(std::size_t board_size) {
  switch (board_size) {
    case 3: return poker::holdem::Street::Flop;
    case 4: return poker::holdem::Street::Turn;
    case 5: return poker::holdem::Street::River;
    default: throw std::invalid_argument("Board must contain 3, 4, or 5 cards.");
  }
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
    h.erase(std::remove_if(h.begin(), h.end(), ::isspace), h.end());
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
  const char r1 = hand.a.describeSuit();
  const char r2 = hand.b.describeSuit();
  const int v1 = hand.a.describeRank();
  const int v2 = hand.b.describeRank();

  char hi = r1;
  char lo = r2;
  if (v2 > v1) {
    hi = r2;
    lo = r1;
  }

  if (hi == lo) {
    return std::string{hi, lo};
  }

  const bool suited = r1 == r2;
  return std::string{hi} + std::string{lo} + (suited ? "s" : "o");
}

std::string combo_label_from_hole_cards(const poker::HoleCards& hand) {
  return poker::to_string(hand.a) + poker::to_string(hand.b);
}

std::optional<int> extract_int_field(const std::string& key, const std::string& name) {
  const std::string needle = "|" + name + "=";
  std::size_t pos = key.find(needle);
  if (pos == std::string::npos) {
    if (key.rfind(name + "=", 0) == 0) {
      pos = 0;
    } else {
      return std::nullopt;
    }
  } else {
    pos += 1;
  }
  pos += name.size() + 1;
  const std::size_t end = key.find('|', pos);
  return std::stoi(key.substr(pos, end == std::string::npos ? std::string::npos : end - pos));
}

std::optional<poker::HoleCards> hand_from_infoset_key(const std::string& key) {
  const auto id = extract_int_field(key, "hbucket");
  if (!id || *id < 0 || *id >= poker::kNumHands) {
    return std::nullopt;
  }
  return poker::hand_from_id(static_cast<poker::HandId>(*id));
}

std::string action_label(const poker::GameAction& action) {
  using poker::holdem::ActionType;
  const ActionType type = static_cast<ActionType>(action.action_type);
  switch (type) {
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
  const ActionType type = static_cast<ActionType>(action.action_type);
  switch (type) {
    case ActionType::Fold: return fl_rgb_color(91, 141, 238);
    case ActionType::Check:
    case ActionType::Call: return fl_rgb_color(94, 186, 125);
    case ActionType::Bet:
    case ActionType::Raise:
    case ActionType::AllIn: {
      switch (bet_index++) {
        case 0: return fl_rgb_color(245, 166, 35);
        case 1: return fl_rgb_color(224, 124, 84);
        default: return fl_rgb_color(196, 69, 105);
      }
    }
  }
  return FL_GRAY;
}

bool node_is_public_chance(const poker::Game& game, int node_id) {
  const poker::Node& node = game.node(node_id);
  return node.player == poker::Player::Chance && node.id != game.root;
}

std::optional<std::string> dealt_board_card_from_action(const poker::GameAction& action) {
  const std::string prefix = "deal_board:";
  if (action.label.rfind(prefix, 0) != 0) return std::nullopt;
  const std::string board = action.label.substr(prefix.size());
  if (board.size() < 2) return std::nullopt;
  return board.substr(board.size() - 2);
}

} // namespace

class Wizard : public Fl_Double_Window {
  struct UserInputs {
    int stackSize{}, startingPot{}, minBet{}, iterations{}, threadCount{};
    float allInThreshold{};
    std::string potType, yourPos, theirPos;
    std::vector<std::string> board;
    std::vector<std::string> heroRange;
    std::vector<std::string> villainRange;
    float min_exploitability{};
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
    std::string action_taken;
  };
  std::vector<GameState> m_history;

  std::map<std::string, std::vector<ComboStrategyDisplay::ComboStrategy>> m_overallStrategyCache;

  struct SolveParams {
    int stackSize{0}, startingPot{0}, minBet{0}, iterations{0};
    float allInThreshold{0}, minExploitability{0};
    std::string potType, yourPos, theirPos;
    std::vector<std::string> board;
    std::vector<std::string> heroRange;
    std::vector<std::string> villainRange;

    bool operator==(const SolveParams& other) const {
      return stackSize == other.stackSize &&
             startingPot == other.startingPot &&
             minBet == other.minBet &&
             iterations == other.iterations &&
             allInThreshold == other.allInThreshold &&
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

  static void cb1Next(Fl_Widget *, void *d) { ((Wizard *)d)->do1Next(); }
  static void cb2Back(Fl_Widget *, void *d) { ((Wizard *)d)->doBack2(); }
  static void cb2Next(Fl_Widget *, void *d) { ((Wizard *)d)->do2Next(); }
  static void cb3Back(Fl_Widget *, void *d) { ((Wizard *)d)->doBack3(); }
  static void cb3Next(Fl_Widget *, void *d) { ((Wizard *)d)->do3Next(); }
  static void cb4Back(Fl_Widget *, void *d) { ((Wizard *)d)->doBack4(); }
  static void cb4Next(Fl_Widget *, void *d) { ((Wizard *)d)->do4Next(); }

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
    m_data.allInThreshold = m_pg1->getAllInThreshold();
    m_data.iterations = m_pg1->getIterations();
    m_data.threadCount = m_pg1->getThreadCount();
    m_data.min_exploitability = m_pg1->getMinExploitability();
    m_data.potType = m_pg1->getPotType();
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
    currentParams.allInThreshold = m_data.allInThreshold;
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

  poker::holdem::BettingAbstraction make_gui_betting_abstraction() const {
    poker::holdem::BettingAbstraction abstraction = poker::holdem::make_standard_abstraction();
    if (m_data.minBet > 0) {
      abstraction.first_bet_sizes.push_back(poker::holdem::BetSize::fixed_amount(m_data.minBet));
    }
    abstraction.validate();
    return abstraction;
  }

  poker::Player initial_player_to_act() const {
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
    config.start_street = street_from_board_size(m_data.board.size());
    config.board = board;
    config.pot_size = m_data.startingPot;
    config.effective_stack = m_data.stackSize;
    config.player_to_act = initial_player_to_act();
    config.p0_range = hero_range;
    config.p1_range = villain_range;
    config.betting_abstraction = make_gui_betting_abstraction();
    config.hand_abstraction = poker::holdem::make_exact_hand_abstraction();
    config.board_abstraction = poker::holdem::make_exact_board_abstraction();
    config.expand_all_in_runouts = false;
    config.collapse_all_in_runouts_to_ev = true;
    config.validate_tree_during_build = false;
    config.reject_preflop = true;

    m_pg5->setStatus("Building native Hold'em subgame tree...");
    Fl::check();

    poker::Game built = poker::holdem::HoldemSubgameBuilder(config).build();
    const std::size_t estimatedMemory = estimate_game_memory(built);
    const std::size_t availableMemory = MemoryUtil::getAvailableMemory();
    m_pg5->setMemoryEstimate(estimatedMemory, availableMemory);
    Fl::check();

    if (!m_pg5->isMemoryOk()) {
      throw std::runtime_error(
          "Not enough memory for this solve. Try reducing range sizes or solving from a later street.");
    }

    m_pg5->setStatus("Training native CFR solver...");
    m_pg5->reset();
    Fl::check();

    const bool useGpu = std::string(m_pg1->getCFRRenderer()) == "GPU";
    if (useGpu) {
      poker::GpuCfrConfig cfr_config;
      cfr_config.num_players = 2;
      cfr_config.synchronize_each_iteration = false;
      cfr_config.threads_per_block = 256;
      cfr_config.use_cfr_plus = false;
      cfr_config.linear_averaging = true;

      m_game = std::make_unique<poker::Game>(std::move(built));
      poker::GpuCfrSolver solver(*m_game, cfr_config);

      const int total = std::max<int>(0, m_data.iterations);
      const int chunk = std::max<int>(1, total / 100);
      for (int done = 0; done < total;) {
        const int step = std::min<int>(chunk, total - done);
        solver.run_iterations(step);
        done += step;
        m_pg5->setIteration(done, total);
        m_pg5->setProgress(done, total);
        Fl::check();
      }

      m_average_strategy = solver.average_strategy();

      {
        std::ofstream log("under_the_gun_gui_debug.log", std::ios::app);
        log << "=== GUI native solve complete ===\n";
        log << "Nodes: " << m_game->num_nodes() << "\n";
        log << "Infosets: " << m_game->num_infosets() << "\n";
        log << "Q entries: " << m_game->num_q() << "\n";
        log << "Iterations: " << total << "\n";
        log << "Board: ";
        for (const auto& card : m_data.board) log << card << ' ';
        log << "\n\n";
      }
    } else {
      poker::CfrConfig cfr_config;
      cfr_config.num_players = 2;
      cfr_config.use_cfr_plus = false;
      cfr_config.linear_averaging = true;
      cfr_config.simultaneous_updates = true;

      m_game = std::make_unique<poker::Game>(std::move(built));
      poker::CpuCfrSolver solver(*m_game, cfr_config);

      const int total = std::max<int>(0, m_data.iterations);
      const int chunk = std::max<int>(1, total / 100);
      for (int done = 0; done < total;) {
        const int step = std::min<int>(chunk, total - done);
        solver.run_iterations(step);
        done += step;
        m_pg5->setIteration(done, total);
        m_pg5->setProgress(done, total);
        Fl::check();
      }

      m_average_strategy = solver.average_strategy();

      {
        std::ofstream log("under_the_gun_gui_debug.log", std::ios::app);
        log << "=== GUI native solve complete ===\n";
        log << "Nodes: " << m_game->num_nodes() << "\n";
        log << "Infosets: " << m_game->num_infosets() << "\n";
        log << "Q entries: " << m_game->num_q() << "\n";
        log << "Iterations: " << total << "\n";
        log << "Board: ";
        for (const auto& card : m_data.board) log << card << ' ';
        log << "\n\n";
      }
    }

    reset_navigation_to_root_public_state();
    m_history.clear();
    m_overallStrategyCache.clear();

    m_pg5->hide();
    m_pg6->show();
    updateStrategyDisplay();
    Fl::check();
  }

  std::size_t estimate_game_memory(const poker::Game& game) const {
    std::size_t bytes = sizeof(poker::Game);
    bytes += game.nodes.size() * sizeof(poker::Node);
    bytes += game.infosets.size() * sizeof(poker::InfoSet);
    bytes += game.q_entries.size() * sizeof(poker::InfoSetAction);
    bytes += game.num_q() * sizeof(float) * 4;
    for (const poker::Node& node : game.nodes) bytes += node.children.size() * sizeof(int);
    for (const poker::InfoSet& infoset : game.infosets) {
      bytes += infoset.actions.size() * sizeof(poker::GameAction);
      bytes += infoset.q_indices.size() * sizeof(int);
      bytes += infoset.key.size();
    }
    return bytes;
  }

  void reset_navigation_to_root_public_state() {
    m_current_pot = m_data.startingPot;
    m_p1_stack = m_p2_stack = m_data.stackSize;
    m_p1_wager = m_p2_wager = 0;
    m_current_nodes.clear();

    if (!m_game) return;

    const poker::Node& root = m_game->node(m_game->root);
    for (int child_id : root.children) {
      m_current_nodes.push_back(child_id);
    }
    normalize_current_nodes();
  }

  void normalize_current_nodes() {
    if (!m_game) return;

    bool changed = true;
    while (changed) {
      changed = false;
      std::vector<int> next;
      for (int node_id : m_current_nodes) {
        const poker::Node& node = m_game->node(node_id);
        // Private-deal chance nodes are only the root. Other chance nodes are public-card choices.
        if (node.player == poker::Player::Chance && node.id == m_game->root) {
          next.insert(next.end(), node.children.begin(), node.children.end());
          changed = true;
        } else {
          next.push_back(node_id);
        }
      }
      m_current_nodes = dedupe_nodes(next);
    }
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
    if (!m_game) return result;
    for (int node_id : m_current_nodes) {
      const poker::Node& node = m_game->node(node_id);
      if (node.player == poker::Player::P0 || node.player == poker::Player::P1) {
        if (node.infoset >= 0 && !node.terminal) result.push_back(node_id);
      }
    }
    return result;
  }

  std::vector<int> current_chance_nodes() const {
    std::vector<int> result;
    if (!m_game) return result;
    for (int node_id : m_current_nodes) {
      if (node_is_public_chance(*m_game, node_id)) result.push_back(node_id);
    }
    return result;
  }

  bool current_is_terminal_only() const {
    if (!m_game || m_current_nodes.empty()) return false;
    for (int node_id : m_current_nodes) {
      if (!m_game->node(node_id).terminal) return false;
    }
    return true;
  }

  std::optional<poker::Player> current_player() const {
    const auto nodes = current_action_nodes();
    if (nodes.empty()) return std::nullopt;
    return m_game->node(nodes.front()).player;
  }

  std::vector<poker::GameAction> representative_actions() const {
    const auto nodes = current_action_nodes();
    if (nodes.empty()) return {};
    const poker::Node& node = m_game->node(nodes.front());
    return m_game->infoset(node.infoset).actions;
  }

  void updatePotAndStacks(const poker::GameAction &action, poker::Player player) {
    using poker::holdem::ActionType;
    int &current_wager = (player == poker::Player::P0) ? m_p1_wager : m_p2_wager;
    int &other_wager = (player == poker::Player::P0) ? m_p2_wager : m_p1_wager;
    int &current_stack = (player == poker::Player::P0) ? m_p1_stack : m_p2_stack;
    int &other_stack = (player == poker::Player::P0) ? m_p2_stack : m_p1_stack;

    const ActionType type = static_cast<ActionType>(action.action_type);
    switch (type) {
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

  void set_common_strategy_header() {
    std::string board = "Board: ";
    for (const auto &card : m_data.board) board += card + " ";
    m_pg6->setBoardInfo(board);

    const int effectivePot = m_current_pot + m_p1_wager + m_p2_wager;
    std::string info = "Hero: " + std::to_string(m_p1_stack) +
                       " | Villain: " + std::to_string(m_p2_stack) +
                       " | Pot: " + std::to_string(effectivePot);
    m_pg6->setPotInfo(info);
  }

  void updateStrategyDisplay() {
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
        const poker::Node& chance = m_game->node(node_id);
        for (int child_id : chance.children) {
          const auto label = dealt_board_card_from_action(m_game->node(child_id).incoming_action);
          if (label) available.insert(*label);
        }
      }
      m_pg6->populateCardChoices(std::vector<std::string>(available.begin(), available.end()));
      m_pg6->showCardSelection(true);
      set_common_strategy_header();
      return;
    }

    const auto action_nodes = current_action_nodes();
    if (action_nodes.empty()) return;

    const poker::Player player = m_game->node(action_nodes.front()).player;
    m_pg6->setTitle(player == poker::Player::P0 ? "Hero's Turn" : "Villain's Turn");
    set_common_strategy_header();

    const auto actions = representative_actions();
    std::map<std::string, std::vector<float>> handTypeStrategies;
    std::map<std::string, int> handTypeCounts;

    for (int node_id : action_nodes) {
      const poker::Node& node = m_game->node(node_id);
      const poker::InfoSet& infoset = m_game->infoset(node.infoset);
      const auto maybe_hand = hand_from_infoset_key(infoset.key);
      if (!maybe_hand) continue;
      const std::string hand_type = hand_type_from_hole_cards(*maybe_hand);

      if (!handTypeStrategies.count(hand_type)) {
        handTypeStrategies[hand_type] = std::vector<float>(actions.size(), 0.0f);
        handTypeCounts[hand_type] = 0;
      }

      for (std::size_t a = 0; a < actions.size() && a < infoset.q_indices.size(); ++a) {
        const int q = infoset.q_indices[a];
        if (q >= 0 && q < static_cast<int>(m_average_strategy.size())) {
          handTypeStrategies[hand_type][a] += m_average_strategy[q];
        }
      }
      handTypeCounts[hand_type]++;
    }

    std::map<std::string, std::map<std::string, float>> strategyMap;
    for (const auto &[hand, stratVec] : handTypeStrategies) {
      const int count = std::max<int>(1, handTypeCounts[hand]);
      std::map<std::string, float> actionProbs;
      for (std::size_t i = 0; i < actions.size(); ++i) {
        const float prob = stratVec[i] / static_cast<float>(count);
        if (prob > 0.001f) actionProbs[action_label(actions[i])] = prob;
      }
      strategyMap[hand] = actionProbs;
    }

    m_pg6->updateStrategyGrid(strategyMap);
    m_pg6->showStrategyGrid(true);
    m_pg6->showAnalysisPanel(true);
    m_pg6->showCardSelection(false);
    m_pg6->deselectHand();

    std::vector<std::string> actionLabels;
    for (const auto& action : actions) actionLabels.push_back(action_label(action));
    m_pg6->setActions(actionLabels);
  }

  void doAction(const std::string &actionStr) {
    if (!m_game) return;
    const auto action_nodes = current_action_nodes();
    if (action_nodes.empty()) return;

    const auto actions = representative_actions();
    int action_idx = -1;
    for (std::size_t i = 0; i < actions.size(); ++i) {
      if (action_label(actions[i]) == actionStr) {
        action_idx = static_cast<int>(i);
        break;
      }
    }
    if (action_idx < 0) return;

    GameState state{m_current_nodes, m_p1_stack, m_p2_stack, m_current_pot,
                    m_p1_wager, m_p2_wager, m_data.board, actionStr};
    m_history.push_back(state);

    updatePotAndStacks(actions[action_idx], m_game->node(action_nodes.front()).player);

    std::vector<int> next;
    for (int node_id : action_nodes) {
      const poker::Node& node = m_game->node(node_id);
      if (action_idx < static_cast<int>(node.children.size())) {
        next.push_back(node.children[action_idx]);
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

  void doCardSelected(const std::string &cardStr) {
    if (!m_game) return;
    const auto chance_nodes = current_chance_nodes();
    if (chance_nodes.empty()) return;

    GameState state{m_current_nodes, m_p1_stack, m_p2_stack, m_current_pot,
                    m_p1_wager, m_p2_wager, m_data.board, "card:" + cardStr};
    m_history.push_back(state);

    std::vector<int> next;
    for (int node_id : chance_nodes) {
      const poker::Node& chance = m_game->node(node_id);
      for (int child_id : chance.children) {
        const auto dealt = dealt_board_card_from_action(m_game->node(child_id).incoming_action);
        if (dealt && *dealt == cardStr) next.push_back(child_id);
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
    std::vector<Fl_Color> actionColors;
    actionColors.reserve(actions.size());
    int betIndex = 0;
    for (const auto& action : actions) {
        actionColors.push_back(color_for_action(action, betIndex));
    }
    std::map<std::string, std::vector<double>> combo_action_sum;
    std::map<std::string, int> combo_count;
    for (int node_id : action_nodes) {
        const poker::Node& node = m_game->node(node_id);

        if (node.infoset < 0) {
            continue;
        }

        const poker::InfoSet& infoset = m_game->infoset(node.infoset);

        const auto maybe_hand = hand_from_infoset_key(infoset.key);
        if (!maybe_hand) {
            continue;
        }

        if (hand_type_from_hole_cards(*maybe_hand) != hand) {
            continue;
        }

        const std::string combo_label =
            combo_label_from_hole_cards(*maybe_hand);

        auto& sums = combo_action_sum[combo_label];
        if (sums.empty()) {
            sums.assign(actions.size(), 0.0);
        }

        for (std::size_t a = 0;
             a < actions.size() && a < infoset.q_indices.size();
             ++a) {
            const int q = infoset.q_indices[a];

            const float prob =
                (q >= 0 && q < static_cast<int>(m_average_strategy.size()))
                    ? m_average_strategy[q]
                    : 0.0f;

            sums[a] += static_cast<double>(prob);
        }

        combo_count[combo_label] += 1;
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

        for (std::size_t a = 0; a < actions.size() && a < sums.size(); ++a) {
            const float prob =
                static_cast<float>(sums[a] / static_cast<double>(count));

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
    if (!m_game) return "";
    const auto action_nodes = current_action_nodes();
    if (action_nodes.empty()) return "";

    std::set<std::string> handTypes;
    for (int node_id : action_nodes) {
      const poker::InfoSet& infoset = m_game->infoset(m_game->node(node_id).infoset);
      const auto maybe_hand = hand_from_infoset_key(infoset.key);
      if (maybe_hand) handTypes.insert(hand_type_from_hole_cards(*maybe_hand));
    }

    std::stringstream result;
    bool first = true;
    for (const auto& hand : handTypes) {
      if (!first) result << ",";
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
    if (!m_game) return;
    const auto action_nodes = current_action_nodes();
    const auto actions = representative_actions();
    if (action_nodes.empty() || actions.empty()) return;

    std::vector<float> overallProbs(actions.size(), 0.0f);
    int validInfosets = 0;

    for (int node_id : action_nodes) {
      const poker::InfoSet& infoset = m_game->infoset(m_game->node(node_id).infoset);
      for (std::size_t a = 0; a < actions.size() && a < infoset.q_indices.size(); ++a) {
        const int q = infoset.q_indices[a];
        if (q >= 0 && q < static_cast<int>(m_average_strategy.size())) overallProbs[a] += m_average_strategy[q];
      }
      ++validInfosets;
    }

    if (validInfosets > 0) {
      for (float& p : overallProbs) p /= static_cast<float>(validInfosets);
    }

    std::vector<Fl_Color> actionColors;
    int betIndex = 0;
    for (const auto &action : actions) actionColors.push_back(color_for_action(action, betIndex));

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

    if (!overallStrat.actions.empty()) overall.push_back(overallStrat);
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
