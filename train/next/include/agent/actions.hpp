// The model's action space, its translation into engine commands, and the
// mask of what the dragon may choose. The only definition anywhere: the JAX
// side receives the mask as an input and never recomputes it.
#pragma once

#include <array>
#include <bitset>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "agent/observation.hpp"

namespace agent {

inline constexpr int kActions = 24;

// 0..3 one step N E S W; 4..19 two steps (first, second); 20..23 split fractions.
class ActionIndex {
 public:
  static constexpr ActionIndex Step(Dir d);
  static constexpr ActionIndex Sprint(Dir first, Dir second);
  static constexpr ActionIndex Split(int fraction_index);      // 0..3
  static constexpr std::optional<ActionIndex> FromInt(int i);  // nullopt outside 0..23
  constexpr int value() const { return value_; }

 private:
  constexpr explicit ActionIndex(int v) : value_(v) {}
  int value_;
};

inline constexpr std::array<double, 4> kSplitFractions = {0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0};

// What the engine is told.
struct Command {
  struct Move { std::vector<Dir> steps; };
  struct Split { int32_t child_segments; };
  struct None {};
  std::variant<Move, Split, None> what;
  std::optional<uint32_t> sonar;
};

// The command for an action of a dragon of `length` segments.
Command ToCommand(ActionIndex a, int32_t length);
// The judge's reply text for a command.
std::string ReplyText(const Command& c);

// true = allowed. Removes what is certain death from what the dragon can see,
// plus the two conservative cases documented in MODEL.md; never empty (falls
// back to the four single steps).
using ActionMask = std::bitset<kActions>;
ActionMask ComputeMask(const Observation& obs);

}  // namespace agent
