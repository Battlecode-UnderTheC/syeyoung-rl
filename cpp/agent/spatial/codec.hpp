// The conditional command codec in C++ (spec section 6), mirroring
// snake_rl/env/actions.py: the private prefix simulator that conditions each
// MOVE token, its masks, and the split candidate table.
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "memory.hpp"

namespace dragons::spatial {

inline constexpr int kPrefixFeatures = 49;
inline constexpr int kTokens = 5;           // N E S W STOP
inline constexpr int kStop = 4;

enum Prefix : int {
  P_STEP = 0, P_POSITION_KNOWN = 1, P_ALIVE_KNOWN = 2, P_FATAL_KNOWN = 3, P_LENGTH_KNOWN = 4,
  P_LENGTH = 5, P_CAN_PAY = 6, P_RESOURCE_UNCERTAIN = 7, P_PREV = 8, P_AHEAD_EMPTY = 12,
  P_AHEAD_KELP = 16, P_AHEAD_PORTAL = 20, P_AHEAD_UNKNOWN = 24, P_OCC_SELF = 28,
  P_OCC_OTHER = 32, P_OCC_HEAD = 36, P_OCC_UNKNOWN = 40, P_PEARL = 44, P_PEARLS_EATEN = 48,
};

class PrefixSimulator {
 public:
  // Reads the lifetime's current facts (after Prepare for this turn).
  PrefixSimulator(const LifetimeState& state);

  void Features(float* out) const;                 // [kPrefixFeatures]
  void Mask(std::array<bool, kTokens>& out) const; // allowed tokens before the next one
  void Advance(int token);

 private:
  enum class Occ { Self, Other, Head, Empty, Unknown };
  Occ Occupancy(Cell c) const;
  void Look(int d, FactualMemory::StepKind& kind, bool& has_landing, Cell& landing,
            Occ& occ, bool& pearl) const;
  bool OldPresent(int index) const { return index < length_ - static_cast<int>(trail_.size()); }

  const LifetimeState& state_;
  const FactualMemory& facts_;
  int K_;
  double area_;
  bool pos_known_ = true;
  Cell pos_;
  int old_length_;
  int length_;
  bool length_known_ = true;
  bool fatal_known_ = false;
  int eaten_ = 0;
  int steps_ = 0;
  int last_ = -1;
  std::vector<Cell> trail_;
  std::map<Cell, int> body_index_;
};

// Legal detached sizes and their generic features (n/L, (L-n)/L, log terms).
struct SplitTable {
  std::vector<int64_t> sizes;
  std::vector<std::array<float, 4>> features;
};
SplitTable MakeSplitTable(const LifetimeState& state);

}  // namespace dragons::spatial
