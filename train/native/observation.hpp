// The legal input of one own turn: the spawn block plus one round block,
// nothing more (spec section 2). Everything the actor knows comes through
// this type; the privileged critic reads the game directly (critic.hpp).
//
// Built from a native game (collection) or parsed from the judge's text
// (deployment); tests hold the two equal.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "dragons/core.hpp"

namespace rl {

inline constexpr int kWindow = dragons::kVision;                // 7
inline constexpr int kWindowTiles = kWindow * kWindow;          // 49
inline constexpr int kMinSide = 7, kMaxSide = 64;               // supported boards (spec section 1)

struct Spawn {
  int32_t id = 0, team = 0, width = 0, height = 0, unit_limit = 0;
};

struct Segment {
  int32_t id = 0;
  uint8_t team = 0;
  int32_t x = 0, y = 0;
  uint8_t facing = 0;
  bool head = false;
};

struct Observation {
  Spawn spawn;
  int32_t round = 0, facing = 0, length = 0, unit_count = 0;
  std::vector<uint64_t> inbox;
  bool echo_present = false;
  std::array<std::string, 5> echo;      // raw field text after ECHOES
  // window tile k = row * 7 + col: its coordinates, pearl and countdown (-1 off beds)
  std::array<int32_t, kWindowTiles> tile_x{}, tile_y{}, countdown{};
  std::array<uint8_t, kWindowTiles> pearl{};
  std::vector<Segment> segments;        // DRAGON_BODIES, in order
  // edge codes (-1 empty, -2 kelp, else portal id): north sides of rows 0..7
  // (row 7 is the south side of row 6), west sides of columns 0..7
  std::array<int32_t, (kWindow + 1) * kWindow> hedge{};
  std::array<int32_t, kWindow * (kWindow + 1)> vedge{};

  int32_t head_x() const { return tile_x[kWindowTiles / 2]; }
  int32_t head_y() const { return tile_y[kWindowTiles / 2]; }
  bool operator==(const Observation&) const = default;
};

// The current actor's turn in `game`.
Observation Observe(const dragons::Game& game);
Spawn SpawnOf(const dragons::Game& game, int32_t dragon);

// The judge's text. Throws std::runtime_error on malformed input or a board
// outside 7..64 a side.
Spawn ParseSpawn(const std::string& init_block);
Observation ParseRound(const Spawn& spawn, const std::string& round_block);

}  // namespace rl
