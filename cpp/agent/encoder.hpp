// Observation encoding, legal-action masking, and batched stepping.
//
// This is the *agent's* side of the boundary and it lives apart from
// cpp/dragons/ on purpose: that directory holds the game rules, pinned to the
// real engine by the differential tests, and nothing about how a policy
// chooses to look at the board should ever require editing it.
//
// The channel layout, scalar layout, action ordering and masking rules mirror
// snake_rl/env/observation.py and actions.py exactly, so a policy trained
// through one can be run through the other. tests/test_cpp_obs.py compares them
// tensor for tensor against the real engine's observations.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "dragons/game.hpp"

namespace dragons {

inline constexpr int kVision = 7;
inline constexpr int kTiles = kVision * kVision;
inline constexpr int kVisionRadius = 3;
inline constexpr int kChannels = 21;
// Spec section 3's scalar vector. Heading is a 4-way one-hot in slots 8..11:
// it was missing entirely, which left the policy unable to tell which way it
// faced whenever the window was not rotated into its frame.
inline constexpr int kScalars = 12;

// channel offsets, matching observation.py
inline constexpr int kPearl = 0, kPearlSoon = 1, kPearlNever = 2;
inline constexpr int kOwnBody = 3, kOwnHead = 4;
inline constexpr int kAllyHead = 5, kAllyBody = 6;
inline constexpr int kEnemyHead = 7, kEnemyBody = 8;
inline constexpr int kFacing = 9, kKelp = 13, kPortal = 17;

// Relative steps. Reverse is left out: it is always fatal.
enum class Rel : uint8_t { F = 0, L = 1, R = 2 };

class ActionSpace {
 public:
  explicit ActionSpace(int max_sprint = 3,
                       std::vector<int> split_sizes = {2, 3, 4, 5, 6, 7, 8, 9},
                       bool allow_portal_steps = true);

  int size() const { return n_; }
  int n_moves() const { return static_cast<int>(paths_.size()); }
  bool is_split(int index) const { return index >= n_moves(); }
  int split_size(int index) const { return split_sizes_[index - n_moves()]; }
  const std::vector<Rel>& path(int index) const { return paths_[index]; }
  bool allow_portal_steps() const { return allow_portal_steps_; }

  Action ToAction(int index, Dir facing, bool sonar, uint32_t payload) const;
  void Mask(const Game& game, const Dragon& dragon, bool* out) const;

 private:
  std::vector<std::vector<Rel>> paths_;
  std::vector<int> split_sizes_;
  bool allow_portal_steps_;
  int n_ = 0;
};

Dir Turn(Dir facing, Rel rel);

void EncodeObservation(const Game& game, const Dragon& dragon, bool ego_rotate,
                       float* grid, float* scalars);

// Many independent arenas stepped together. Within a round, dragons still act
// one at a time in id order -- what is batched is the same slot across arenas.
class VecGame {
 public:
  VecGame(const std::vector<std::string>& maps, uint64_t seed,
          ActionSpace space, bool ego_rotate = true);

  int arenas() const { return static_cast<int>(games_.size()); }
  const ActionSpace& space() const { return space_; }
  Game& game(int i) { return games_[i]; }

  void BeginRound();
  void EndRound();

  // Fills `grid`, `scalars` and `mask` for every arena that still has a dragon
  // at `slot`, and records which arena each row came from. Returns the row
  // count; zero means the round is done.
  int PrepareSlot(int slot, float* grid, float* scalars, bool* mask,
                  int32_t* arena_out, int32_t* dragon_out);
  void SubmitSlot(const int32_t* actions, const uint8_t* sonar,
                  const uint32_t* payloads);
  // Same selection, without writing observations: isolates encoder cost.
  int PrepareSlotNoEncode(int slot, bool* mask, int32_t* arena_out,
                          int32_t* dragon_out);

  // Resets arenas whose match has ended, and reports how many were reset.
  int ResetFinished(const std::vector<std::string>& maps, uint64_t seed);

 private:
  std::vector<Game> games_;
  ActionSpace space_;
  bool ego_rotate_;
  std::vector<int32_t> pending_arena_;
  std::vector<int32_t> pending_index_;
};

}  // namespace dragons

namespace dragons {
// Plays whole rounds entirely in C++, picking the first legal action for every
// dragon. Exists to measure the sim + encoder + mask without a language
// boundary in the loop, which is what the native trainer will see.
struct BenchResult {
  int64_t dragon_turns = 0;
  int64_t rounds = 0;
};
BenchResult BenchmarkRounds(VecGame& vec, int rounds, bool encode);
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
