#include "critic.hpp"

#include <algorithm>
#include <stdexcept>

namespace rl {
namespace {

int32_t EdgeType(int32_t code) {
  return code == dragons::kEdgeEmpty ? 1 : code == dragons::kEdgeKelp ? 2 : 3;
}

}  // namespace

void EncodeCriticGrid(const dragons::Game& g, int32_t* cells) {
  const dragons::Board& b = g.board();
  const int W = b.width, H = b.height;
  std::fill(cells, cells + kCriticCells * kCriticCellFields, 0);
  for (int t = 0; t < W * H; ++t) {
    const int x = t % W, y = t / W;
    int32_t* c = cells + ((y * kCriticGrid / H) * kCriticGrid + x * kCriticGrid / W) * kCriticCellFields;
    ++c[kCTiles];
    c[kCPearls] += g.pearl(t) ? 1 : 0;
    if (b.spawns[t] != 0) {
      ++c[kCBeds];
      c[kCTimerSum] += g.next_pearl(t);
    }
    const int32_t who = g.occupant(t);
    if (who >= 0) {
      const dragons::Dragon& d = g.dragons()[who];
      const bool head = d.Head() == t;
      ++c[d.team == 0 ? (head ? kCHeadA : kCBodyA) : (head ? kCHeadB : kCBodyB)];
    }
    const int32_t sides[4] = {b.north_code[t], b.west_code[y * W + (x + 1) % W], b.north_code[((y + 1) % H) * W + x],
                              b.west_code[t]};
    for (int d = 0; d < 4; ++d) {
      const int32_t e = EdgeType(sides[d]);
      c[kCKelp] += e == 2;
      c[kCPortal] += e == 3;
    }
  }
}

int32_t EncodeEvent(const dragons::Game& g, int32_t* cells, int32_t d_pad, int32_t* rows, int32_t* q) {
  const dragons::Board& b = g.board();
  if (g.actor() < 0) throw std::logic_error("no event: the game is over");
  EncodeCriticGrid(g, cells);
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
