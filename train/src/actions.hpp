// The model's 24 actions as engine commands; mirrors action_command in
// dragons_model/features.py (the model decides, this only translates).
//
//   0..3    MOVE one step N E S W
//   4..19   MOVE two steps, first (a - 4) / 4, then (a - 4) % 4
//   20..23  SPLIT, child = clamp(round(f * length), 2, length - 2), f = 1/4, 1/3, 1/2, 2/3
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "dragons/core.hpp"

namespace train {

struct Command {
  uint8_t kind;
  uint8_t steps[2];
  int32_t nsteps;
  int32_t split;
};

inline Command ActionCommand(int action, int length) {
  static constexpr double kFractions[4] = {0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0};
  Command c{dragons::kMove, {0, 0}, 1, 0};
  if (action < 4) {
    c.steps[0] = static_cast<uint8_t>(action);
  } else if (action < 20) {
    c.steps[0] = static_cast<uint8_t>((action - 4) / 4);
    c.steps[1] = static_cast<uint8_t>((action - 4) % 4);
    c.nsteps = 2;
  } else {
    c.kind = dragons::kSplit;
    // Python's round() is half-to-even; std::nearbyint in the default rounding mode matches it
    const int child = static_cast<int>(std::nearbyint(kFractions[action - 20] * length));
    c.split = std::max(2, std::min(child, length - 2));
  }
  return c;
}

}  // namespace train
