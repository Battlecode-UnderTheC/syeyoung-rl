// The critic's view of a turn: the whole game state, summarised. Training only
// (a bot never sees it); built from the engine's state on the agent side, so
// dragons-env carries no critic features.
#pragma once

#include <cstdint>
#include <span>
#include <string>

#include "dragons/core.hpp"

namespace train {

class CriticLayout {
 public:
  CriticLayout();
  int64_t bytes() const;
  std::string Describe() const;   // exported and checked like the feature layout
};

// For the dragon about to act in `game`: the board in 16x16 blocks (pearls,
// beds, kelp, bodies and heads of its team and the other, the actor) and both
// teams' standing. Writes exactly layout.bytes() into `out`.
void EncodeCritic(const dragons::Game& game, const CriticLayout& layout, std::span<uint8_t> out);

}  // namespace train
