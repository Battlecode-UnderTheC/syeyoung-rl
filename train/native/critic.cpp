#include "critic.hpp"

#include <algorithm>
#include <stdexcept>

namespace rl {
namespace {

int32_t EdgeType(int32_t code) {
  return code == dragons::kEdgeEmpty ? 1 : code == dragons::kEdgeKelp ? 2 : 3;
}

}  // namespace

void EncodeBoard(const dragons::Board& b, int32_t n_pad, int32_t* tile, int32_t* dest) {
  const int W = b.width, H = b.height, n = W * H;
  if (n_pad < n) throw std::invalid_argument("n_pad is smaller than the board");
  for (int t = 0; t < n_pad; ++t) {
    if (t >= n) {
      tile[2 * t] = tile[2 * t + 1] = 0;
      for (int d = 0; d < 4; ++d) dest[4 * t + d] = -1;
      continue;
    }
    const int x = t % W, y = t / W;
    const int32_t sides[4] = {b.north_code[t], b.west_code[y * W + (x + 1) % W], b.north_code[((y + 1) % H) * W + x],
                              b.west_code[t]};
    int32_t packed = 0;
    for (int d = 0; d < 4; ++d) {
      packed |= EdgeType(sides[d]) << (2 * d);
      dest[4 * t + d] = b.step_to[4 * t + d];
    }
    tile[2 * t] = packed;
    tile[2 * t + 1] = 1;
  }
}

int32_t EncodeEvent(const dragons::Game& g, int32_t n_pad, int32_t* grid, int32_t d_pad, int32_t* rows, int32_t* q) {
  const dragons::Board& b = g.board();
  const int n = b.width * b.height;
  if (g.actor() < 0) throw std::logic_error("no event: the game is over");
  for (int t = 0; t < n_pad; ++t) {
    if (t >= n) {
      grid[t] = 0;
      continue;
    }
    int32_t occ = 0;
    const int32_t who = g.occupant(t);
    if (who >= 0) {
      const dragons::Dragon& d = g.dragons()[who];
      occ = 1 + 2 * d.team + (d.Head() != t);
    }
    const bool bed = b.spawns[t] != 0;
    grid[t] = static_cast<int32_t>(g.pearl(t)) | occ << 1 | static_cast<int32_t>(bed) << 4 |
              (bed ? g.next_pearl(t) + kCountdownBias : 0) << 8;
  }
  const auto& dragons = g.dragons();
  const int32_t next = g.actor();
  int32_t alive = 0, due = 0, next_rank = 0;
  int32_t longest[2] = {0, 0}, total[2] = {0, 0}, count[2] = {0, 0};
  for (const dragons::Dragon& d : dragons) {
    if (!d.alive) continue;
    if (d.id == next) next_rank = alive;
    if (alive < d_pad) {
      int32_t* r = rows + static_cast<int64_t>(alive) * kCriticDragonFields;
      r[kDTeam] = d.team;
      r[kDLength] = d.length;
      r[kDHeadX] = d.Head() % b.width;
      r[kDHeadY] = d.Head() / b.width;
      r[kDFacing] = d.facing;
      r[kDRank] = alive;
      r[kDActed] = d.id < next;
      r[kDId] = d.id;
      r[kDValid] = 1;
    }
    ++alive;
    due += d.id >= next;
    longest[d.team] = std::max(longest[d.team], d.length);
    total[d.team] += d.length;
    ++count[d.team];
  }
  for (int i = alive; i < d_pad; ++i)
    for (int f = 0; f < kCriticDragonFields; ++f) rows[static_cast<int64_t>(i) * kCriticDragonFields + f] = 0;
  q[kQRound] = g.round();
  q[kQWidth] = b.width;
  q[kQHeight] = b.height;
  q[kQUnitLimit] = b.unit_limit;
  q[kQLongestA] = longest[0];
  q[kQLongestB] = longest[1];
  q[kQTotalA] = total[0];
  q[kQTotalB] = total[1];
  q[kQCountA] = count[0];
  q[kQCountB] = count[1];
  q[kQNextTeam] = dragons[next].team;
  q[kQNextRank] = next_rank;
  q[kQAlive] = alive;
  q[kQDue] = due;
  q[kQNextLength] = dragons[next].length;
  return alive;
}

}  // namespace rl
