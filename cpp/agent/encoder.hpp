// The agent's side of the native sim: the engine's exact protocol text and
// full-state dumps. It lives apart from cpp/dragons/ on purpose: that
// directory holds the game rules, pinned to the real engine by differential
// tests, and nothing about how a policy looks at the board should require
// editing it.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "dragons/game.hpp"

namespace dragons {

inline constexpr int kVision = 7;
inline constexpr int kTiles = kVision * kVision;
inline constexpr int kVisionRadius = 3;

}  // namespace dragons
// ---- dragons-spatial-v1: exact protocol text and full state (spec 7, 14) ----
namespace dragons {
// The judge's spawn and turn blocks for one dragon, byte for byte what the wasm
// engine prints, so the Python factual memory can be driven from the native
// sim. `portal_ids` has 2*W*H entries: [y*W+x] is the north edge of (x, y) and
// [W*H + y*W+x] its west edge, -1 where the edge is not a portal. The sim keeps
// only partner links, so the ids come from the map text.
std::string SpawnBlock(const Game& game, int dragon_index);
std::string TurnBlock(const Game& game, int dragon_index, const std::vector<int>& portal_ids);
// Full-board dumps for the training-only critic: pearl flags, countdowns (-1
// where a tile never spawns) and edge kinds (0 empty, 1 kelp, 2 portal) on the
// north and west side of every tile.
void DumpTiles(const Game& game, uint8_t* pearl, int32_t* countdown);
void DumpEdges(const Game& game, uint8_t* north_kind, uint8_t* west_kind);
}  // namespace dragons
