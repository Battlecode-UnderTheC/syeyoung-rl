#include "encoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace dragons {
namespace {

constexpr int kMinSize = Game::kMinSize;

int WrapTo(int value, int modulus) {
  const int r = value % modulus;
  return r < 0 ? r + modulus : r;
}

// The engine reports a facing per segment, not per dragon: the head points
// where it is heading and every body segment points at the segment in front of
// it, found by trying each direction (its DirectionOfStepBetween). Portals make
// this more than a coordinate subtraction.
Dir SegmentFacing(const Game& game, const Dragon& dragon, size_t i) {
  if (i == 0) return dragon.facing;
  const Point from = dragon.segments[i];
  const Point toward = dragon.segments[i - 1];
  for (int d = 0; d < 4; ++d) {   // N, E, S, W, the order the engine tries
    const std::optional<Point> step = game.TileAfterStep(from, static_cast<Dir>(d));
    if (step && *step == toward) return static_cast<Dir>(d);
  }
  return dragon.facing;
}

// Where a step would land if there were no portal: used to tell a portal jump
// apart from an ordinary step.
Point PlainStep(const Game& game, Point from, Dir dir) {
  const int d = static_cast<int>(dir);
  return Point{static_cast<int16_t>(WrapTo(from.x + kDx[d], game.width())),
               static_cast<int16_t>(WrapTo(from.y + kDy[d], game.height()))};
}

// Destination of source cell (row, col) under numpy's rot90 by k.
// np.rot90(A)[i][j] == A[j][n-1-i], so source (r, c) lands at (n-1-c, r).
int RotatedIndex(int row, int col, int k) {
  switch (k & 3) {
    case 0: return row * kVision + col;
    case 1: return (kVision - 1 - col) * kVision + row;
    case 2: return (kVision - 1 - row) * kVision + (kVision - 1 - col);
    default: return col * kVision + (kVision - 1 - row);
  }
}

}  // namespace

Dir Turn(Dir facing, Rel rel) {
  const int f = static_cast<int>(facing);
  if (rel == Rel::F) return facing;
  return static_cast<Dir>(rel == Rel::L ? (f + 3) % 4 : (f + 1) % 4);
}

ActionSpace::ActionSpace(int max_sprint, std::vector<int> split_sizes,
                         bool allow_portal_steps)
    : split_sizes_(std::move(split_sizes)),
      allow_portal_steps_(allow_portal_steps) {
  // Enumerated exactly like itertools.product(("F","L","R"), repeat=k) for
  // k = 1..max_sprint, so action indices agree with the Python space.
  for (int k = 1; k <= max_sprint; ++k) {
    const int total = static_cast<int>(std::pow(3, k));
    for (int code = 0; code < total; ++code) {
      std::vector<Rel> path(k);
      int rest = code;
      for (int position = k - 1; position >= 0; --position) {
        path[position] = static_cast<Rel>(rest % 3);
        rest /= 3;
      }
      paths_.push_back(std::move(path));
    }
  }
  n_ = n_moves() + static_cast<int>(split_sizes_.size());
}

Action ActionSpace::ToAction(int index, Dir facing, bool sonar,
                             uint32_t payload) const {
  Action action;
  action.sonar = sonar;
  action.payload = payload;
  if (is_split(index)) {
    action.kind = Action::Kind::Split;
    action.split_size = split_size(index);
    return action;
  }
  action.kind = Action::Kind::Move;
  Dir at = facing;
  for (Rel rel : path(index)) {
    at = Turn(at, rel);
    action.dirs.push_back(at);
  }
  return action;
}

void ActionSpace::Mask(const Game& game, const Dragon& dragon, bool* out) const {
  const int length = dragon.length();
  for (int a = 0; a < n_moves(); ++a) {
    const std::vector<Rel>& steps = path(a);
    const int count = static_cast<int>(steps.size());
    if (length - (count - 1) < kMinSize) { out[a] = false; continue; }
    Point at = dragon.head();
    Dir facing = dragon.facing;
    bool ok = true;
    for (int s = 0; s < count; ++s) {
      facing = Turn(facing, steps[s]);
      const std::optional<Point> next = game.TileAfterStep(at, facing);
      if (!next) { ok = false; break; }                     // kelp
      if (!(*next == PlainStep(game, at, facing))) {
        // A portal's far side cannot be seen, so it is only offered as a lone
        // first step and its landing tile goes unchecked.
        ok = (count == 1) && allow_portal_steps_;
        break;
      }
      at = *next;
      // Reading the board rather than the dragon's window is equivalent here:
      // a path is at most max_sprint (<= 3) steps, so every tile it can reach
      // is within the 7x7 view. tests/test_cpp_obs.py pins that.
      if (game.AliveDragonOccupying(at) != nullptr) { ok = false; break; }
    }
    out[a] = ok;
  }
  const bool room = game.AliveUnitCount(dragon.team) < game.unit_limit();
  for (size_t j = 0; j < split_sizes_.size(); ++j) {
    const int size = split_sizes_[j];
    out[n_moves() + j] = room && size >= kMinSize && length - size >= kMinSize;
  }
}

void EncodeObservation(const Game& game, const Dragon& dragon, bool ego_rotate,
                       float* grid, float* scalars) {
  std::memset(grid, 0, sizeof(float) * kChannels * kTiles);
  const Point head = dragon.head();
  const int k = ego_rotate ? static_cast<int>(dragon.facing) : 0;

  for (int row = 0; row < kVision; ++row) {
    for (int col = 0; col < kVision; ++col) {
      const int x = WrapTo(head.x - kVisionRadius + col, game.width());
      const int y = WrapTo(head.y - kVisionRadius + row, game.height());
      const Point p{static_cast<int16_t>(x), static_cast<int16_t>(y)};
      const Tile& tile = game.TileAt(p);
      const int cell = RotatedIndex(row, col, k);
      auto at = [&](int channel) -> float& { return grid[channel * kTiles + cell]; };

      const int32_t timer = tile.spawns ? tile.countdown : -1;
      at(kPearl) = tile.pearl ? 1.0f : 0.0f;
      if (timer < 0) {
        at(kPearlNever) = 1.0f;
      } else {
        at(kPearlSoon) = 1.0f / (1.0f + static_cast<float>(timer));
      }

      const int32_t occupant = game.OccupantIndex(p);
      if (occupant >= 0) {
        const Dragon& other = game.dragons()[occupant];
        if (other.alive) {
          const bool is_head = other.head() == p;
          if (other.id == dragon.id) {
            at(is_head ? kOwnHead : kOwnBody) = 1.0f;
          } else if (other.team == dragon.team) {
            at(is_head ? kAllyHead : kAllyBody) = 1.0f;
          } else {
            at(is_head ? kEnemyHead : kEnemyBody) = 1.0f;
          }
          // directional groups are rolled so index 0 is "forward" in ego frame
          const int facing = (static_cast<int>(game.TileFacing(p)) - k + 4) % 4;
          at(kFacing + facing) = 1.0f;
        }
      }
      for (int d = 0; d < 4; ++d) {
        const Dir side = static_cast<Dir>(d);
        const std::optional<Point> step = game.TileAfterStep(p, side);
        const int slot = (d - k + 4) % 4;
        if (!step) {
          at(kKelp + slot) = 1.0f;
        } else if (!(*step == PlainStep(game, p, side))) {
          at(kPortal + slot) = 1.0f;
        }
      }
    }
  }

  const int length = dragon.length();
  const int units = game.AliveUnitCount(dragon.team);
  const int max_sprint = std::min(3, std::max(length - kMinSize + 1, 1));
  scalars[0] = length / 20.0f;
  scalars[1] = std::log1p(static_cast<float>(length)) / 4.0f;
  scalars[2] = units / 8.0f;
  scalars[3] = game.round() / 500.0f;
  scalars[4] = max_sprint / 3.0f;
  scalars[5] = (length - kMinSize >= kMinSize && units < game.unit_limit()) ? 1.0f : 0.0f;
  scalars[6] = static_cast<float>(dragon.inbox.size()) / 4.0f;
  scalars[7] = units >= game.unit_limit() ? 1.0f : 0.0f;
  for (int d = 0; d < 4; ++d) {
    scalars[8 + d] = static_cast<int>(dragon.facing) == d ? 1.0f : 0.0f;
  }
}

VecGame::VecGame(const std::vector<std::string>& maps, uint64_t seed,
                 ActionSpace space, bool ego_rotate)
    : space_(std::move(space)), ego_rotate_(ego_rotate) {
  games_.reserve(maps.size());
  for (size_t i = 0; i < maps.size(); ++i) {
    games_.push_back(Game::FromMapText(maps[i], seed + i));
  }
}

void VecGame::BeginRound() {
  for (Game& game : games_) {
    if (!game.Finished()) game.BeginRound();
  }
}

void VecGame::EndRound() {
  for (Game& game : games_) {
    if (!game.Finished()) game.EndRound();
  }
}

int VecGame::PrepareSlot(int slot, float* grid, float* scalars, bool* mask,
                         int32_t* arena_out, int32_t* dragon_out) {
  pending_arena_.clear();
  pending_index_.clear();
  int rows = 0;
  for (int a = 0; a < arenas(); ++a) {
    Game& game = games_[a];
    if (game.Finished()) continue;
    if (slot >= static_cast<int>(game.dragons().size())) continue;
    const Dragon& dragon = game.dragons()[slot];
    if (!dragon.alive) continue;
    EncodeObservation(game, dragon, ego_rotate_,
                      grid + static_cast<size_t>(rows) * kChannels * kTiles,
                      scalars + static_cast<size_t>(rows) * kScalars);
    space_.Mask(game, dragon, mask + static_cast<size_t>(rows) * space_.size());
    arena_out[rows] = a;
    dragon_out[rows] = dragon.id;
    pending_arena_.push_back(a);
    pending_index_.push_back(slot);
    ++rows;
  }
  return rows;
}

int VecGame::PrepareSlotNoEncode(int slot, bool* mask, int32_t* arena_out,
                                 int32_t* dragon_out) {
  pending_arena_.clear();
  pending_index_.clear();
  int rows = 0;
  for (int a = 0; a < arenas(); ++a) {
    Game& game = games_[a];
    if (game.Finished()) continue;
    if (slot >= static_cast<int>(game.dragons().size())) continue;
    const Dragon& dragon = game.dragons()[slot];
    if (!dragon.alive) continue;
    space_.Mask(game, dragon, mask + static_cast<size_t>(rows) * space_.size());
    arena_out[rows] = a;
    dragon_out[rows] = dragon.id;
    pending_arena_.push_back(a);
    pending_index_.push_back(slot);
    ++rows;
  }
  return rows;
}

void VecGame::SubmitSlot(const int32_t* actions, const uint8_t* sonar,
                         const uint32_t* payloads) {
  for (size_t row = 0; row < pending_arena_.size(); ++row) {
    Game& game = games_[pending_arena_[row]];
    const int index = pending_index_[row];
    const Dragon& dragon = game.dragons()[index];
    const Action action = space_.ToAction(actions[row], dragon.facing,
                                          sonar != nullptr && sonar[row] != 0,
                                          payloads != nullptr ? payloads[row] : 0);
    game.TakeTurn(index, action);
  }
  pending_arena_.clear();
  pending_index_.clear();
}

BenchResult BenchmarkRounds(VecGame& vec, int rounds, bool encode) {
  const int cap = vec.arenas();
  const int actions = vec.space().size();
  std::vector<float> grid(static_cast<size_t>(cap) * kChannels * kTiles);
  std::vector<float> scalars(static_cast<size_t>(cap) * kScalars);
  std::vector<bool> mask_bits;
  std::vector<uint8_t> mask(static_cast<size_t>(cap) * actions);
  std::vector<int32_t> arena(cap), dragon(cap), chosen(cap);
  BenchResult out;
  for (int r = 0; r < rounds; ++r) {
    vec.BeginRound();
    for (int slot = 0;; ++slot) {
      const int rows = encode
          ? vec.PrepareSlot(slot, grid.data(), scalars.data(),
                            reinterpret_cast<bool*>(mask.data()), arena.data(),
                            dragon.data())
          : vec.PrepareSlotNoEncode(slot, reinterpret_cast<bool*>(mask.data()),
                                    arena.data(), dragon.data());
      if (rows == 0) break;
      for (int row = 0; row < rows; ++row) {
        const uint8_t* row_mask = mask.data() + static_cast<size_t>(row) * actions;
        int pick = 0;
        for (int a = 0; a < actions; ++a) {
          if (row_mask[a]) { pick = a; break; }
        }
        chosen[row] = pick;
      }
      vec.SubmitSlot(chosen.data(), nullptr, nullptr);
      out.dragon_turns += rows;
    }
    vec.EndRound();
    ++out.rounds;
  }
  return out;
}

int VecGame::ResetFinished(const std::vector<std::string>& maps, uint64_t seed) {
  int reset = 0;
  for (size_t a = 0; a < games_.size(); ++a) {
    if (!games_[a].Finished()) continue;
    games_[a] = Game::FromMapText(maps[reset % maps.size()], seed + a);
    ++reset;
  }
  return reset;
}

}  // namespace dragons
