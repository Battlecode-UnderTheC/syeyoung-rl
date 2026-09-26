#include "facts.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace rl {

Lifetime::Lifetime(const Spawn& spawn)
    : spawn_(spawn), n_(spawn.width * spawn.height), tiles_(n_), edges_(2 * static_cast<size_t>(n_)) {
  if (spawn.width < kMinSide || spawn.width > kMaxSide || spawn.height < kMinSide || spawn.height > kMaxSide)
    throw std::invalid_argument("boards are 7..64 a side");
  body_index_.assign(n_, -1);
  in_view_.assign(n_, 0);
}

int32_t Lifetime::Wrap(int x, int y) const {
  const int W = spawn_.width, H = spawn_.height;
  x %= W; if (x < 0) x += W;
  y %= H; if (y < 0) y += H;
  return y * W + x;
}

int32_t Lifetime::Neighbour(int32_t t, int dir) const {
  return Wrap(t % spawn_.width + dragons::kDx[dir], t / spawn_.width + dragons::kDy[dir]);
}

int32_t Lifetime::SideEdge(int32_t t, int dir) const {
  switch (dir) {
    case 0: return 2 * t;
    case 2: return 2 * Neighbour(t, 2);
    case 3: return 2 * t + 1;
    default: return 2 * Neighbour(t, 1) + 1;
  }
}

int32_t Lifetime::Dest(int32_t t, int dir) const {
  const int32_t e = SideEdge(t, dir);
  const EdgeFact& f = edges_[e];
  if (f.type == kEdgeOpen) return Neighbour(t, dir);
  if (f.type != kEdgePortal) return kUnknown;
  const auto it = portals_.find(f.portal);
  if (it == portals_.end() || it->second.size() != 2) return kUnknown;
  const int32_t other = it->second[0] == e ? it->second[1] : it->second[0];
  // crossing the partner edge in the same direction: a north side (even) is
  // crossed going N or S, a west side (odd) going W or E
  const int32_t beyond = other / 2;
  if (other % 2 == 0) return dir == 2 ? beyond : Neighbour(beyond, 0);
  return dir == 1 ? beyond : Neighbour(beyond, 3);
}

void Lifetime::Visit(int32_t t, int32_t round) {
  ++tiles_[t].visits;
  tiles_[t].visit_round = round;
}

bool Lifetime::Observe(const Observation& o) {
  if (o.round == last_key_) return false;
  if (o.spawn.id != spawn_.id || o.spawn.width != spawn_.width || o.spawn.height != spawn_.height)
    throw std::invalid_argument("observation of another dragon or board");
  for (const int32_t t : body_) if (t != kUnknown) {       // the old body's index records
    body_index_[t] = -1;
  }
  Reconcile(o);
  Record(o);
  LinkBody(o);
  for (int i = static_cast<int>(body_.size()); i-- > 0;)   // the new body's, lowest index winning
    if (body_[i] != kUnknown) {
      body_index_[body_[i]] = i;
    }
  last_key_ = o.round;
  return true;
}

// Step 1: what the last command did, from what is privately known.
void Lifetime::Reconcile(const Observation& o) {
  const int32_t head = Wrap(o.head_x(), o.head_y());
  const int32_t L = o.length;
  std::vector<int32_t> body;
  if (!prev_valid_) {
    body.assign(L, kUnknown);
    body[0] = head;
    Visit(head, o.round);                                // spawn: the head, once
  } else if (prev_.kind == 0) {
    // intermediate heads along known transitions, newest first
    const int k = prev_.nsteps;
    std::vector<int32_t> path(k, kUnknown);
    int32_t at = cmd_head_;
    for (int i = 0; i < k; ++i) {
      at = at == kUnknown ? kUnknown : Dest(at, prev_.dirs[i]);
      path[i] = at;
    }
    if (path[k - 1] != kUnknown && path[k - 1] != head)   // the facts disagree with the report
      std::fill(path.begin(), path.end(), kUnknown);
    path[k - 1] = head;
    for (int i = 0; i < k; ++i)
      if (path[i] != kUnknown) Visit(path[i], cmd_round_);   // once per passage
    body.assign(path.rbegin(), path.rend());
    body.insert(body.end(), body_.begin(), body_.end());
  } else {
    const int keep = std::max(0, cmd_length_ - prev_.split);
    body.assign(body_.begin(), body_.begin() + std::min<size_t>(keep, body_.size()));
    if (body.empty() || body[0] != head) {                 // contradiction: only the report survives
      body.assign(1, head);
    }
  }
  body.resize(L, kUnknown);
  body_ = std::move(body);
}

// Step 2: promote and overwrite what is in view; record the edges seen.
void Lifetime::Record(const Observation& o) {
  const int W = spawn_.width;
  for (const int32_t t : window_) {             // the last window is no longer in view
    in_view_[t] = 0;
  }
  window_.clear();
  for (int k = 0; k < kWindowTiles; ++k) {
    const int32_t t = o.tile_y[k] * W + o.tile_x[k];
    in_view_[t] = 1;
    window_.push_back(t);
  }
  for (int k = 0; k < kWindowTiles; ++k) {
    TileFact& f = tiles_[o.tile_y[k] * W + o.tile_x[k]];
    if (f.seen_round != kUnknown) {
      f.prev_round = f.seen_round;
      f.prev_pearl = f.pearl;
      f.prev_countdown = f.countdown;
    } else {
      ++seen_tiles_;
    }
    f.seen_round = o.round;
    f.pearl = o.pearl[k];
    f.countdown = o.countdown[k];
    f.occ = kOccEmpty;
    f.occ_id = -1;
    f.occ_facing = -1;
  }
  for (const Segment& s : o.segments) {
    TileFact& f = tiles_[s.y * W + s.x];
    const bool self = s.id == spawn_.id, ally = s.team == spawn_.team;
    f.occ = self ? (s.head ? kOccSelfHead : kOccSelfBody)
                 : ally ? (s.head ? kOccAllyHead : kOccAllyBody) : (s.head ? kOccEnemyHead : kOccEnemyBody);
    f.occ_id = s.id;
    f.occ_facing = static_cast<int8_t>(s.facing);
  }
  auto set_edge = [&](int32_t e, int32_t code) {
    EdgeFact& f = edges_[e];
    if (f.type == kEdgeUnknown) ++known_edges_;
    if (code == dragons::kEdgeEmpty) {
      f.type = kEdgeOpen;
    } else if (code == dragons::kEdgeKelp) {
      f.type = kEdgeKelp;
    } else {
      f.type = kEdgePortal;
      if (code > INT16_MAX) throw std::out_of_range("portal id beyond int16");
      f.portal = static_cast<int16_t>(code);
      auto& ends = portals_[code];
      if (std::find(ends.begin(), ends.end(), e) == ends.end()) {
        ends.push_back(e);
      }
    }
  };
  const int32_t x7 = (o.tile_x[kWindow - 1] + 1) % W;
  const int32_t y7 = (o.tile_y[kWindowTiles - 1] + 1) % spawn_.height;
  for (int r = 0; r <= kWindow; ++r)
    for (int c = 0; c < kWindow; ++c) {
      const int32_t y = r < kWindow ? o.tile_y[r * kWindow] : y7;
      set_edge(2 * (y * W + o.tile_x[c]), o.hedge[r * kWindow + c]);
    }
  for (int r = 0; r < kWindow; ++r)
    for (int c = 0; c <= kWindow; ++c) {
      const int32_t x = c < kWindow ? o.tile_x[c] : x7;
      set_edge(2 * (o.tile_y[r * kWindow] * W + x) + 1, o.vedge[r * (kWindow + 1) + c]);
    }
}

// Step 3: provable links from visible own segments; contradictions become unknown.
void Lifetime::LinkBody(const Observation& o) {
  const int W = spawn_.width;
  const int L = static_cast<int>(body_.size());
  std::vector<std::pair<int32_t, int>> own;          // visible non-head own segments: tile, facing
  for (const Segment& s : o.segments)
    if (s.id == spawn_.id && !s.head) own.emplace_back(s.y * W + s.x, s.facing);
  auto visible = [&](int32_t t) {
    for (int k = 0; k < kWindowTiles; ++k)
      if (o.tile_y[k] * W + o.tile_x[k] == t) return true;
    return false;
  };
  // an inferred segment on a tile in view that shows no body of ours is wrong
  for (int i = 1; i < L; ++i)
    if (body_[i] != kUnknown && visible(body_[i]) && tiles_[body_[i]].occ != kOccSelfBody) body_[i] = kUnknown;
  std::vector<uint8_t> conflicted(L, 0);
  for (bool changed = true; changed;) {
    changed = false;
    for (int i = 0; i + 1 < L; ++i) {
      if (body_[i] == kUnknown || conflicted[i + 1]) continue;
      int32_t cand = kUnknown;
      int count = 0;
      for (const auto& [t, facing] : own)
        if (Dest(t, facing) == body_[i]) { cand = t; ++count; }
      if (count != 1 || body_[i + 1] == cand) continue;
      const bool elsewhere = std::find(body_.begin(), body_.end(), cand) != body_.end();
      if (body_[i + 1] == kUnknown && !elsewhere) {
        body_[i + 1] = cand;
      } else {
        body_[i + 1] = kUnknown;
        conflicted[i + 1] = 1;
      }
      changed = true;
    }
  }
  // a tile claimed twice is not known for either index
  for (int i = 1; i < L; ++i) {
    if (body_[i] == kUnknown) continue;
    bool dup = body_[i] == body_[0];
    for (int j = i + 1; j < L; ++j)
      if (body_[j] == body_[i]) { body_[j] = kUnknown; dup = true; }
    if (dup) body_[i] = kUnknown;
  }
  last_head_ = body_[0];
  last_length_ = L;
  last_round_ = o.round;
}

void Lifetime::Commit(const Command& c) {
  if (c.kind == 0 && (c.nsteps < 1 || c.nsteps > kMaxMoveTokens)) throw std::invalid_argument("MOVE needs 1..8 steps");
  prev_ = c;
  prev_valid_ = true;
  cmd_head_ = last_head_;
  cmd_length_ = last_length_;
  cmd_round_ = last_round_;
  ++turns_;
}

void Lifetime::EncodeTile(int32_t t, int32_t* r, int32_t* dest) const {
  const int W = spawn_.width, H = spawn_.height;
  const TileFact& f = tiles_[t];
  r[kTSeenRound] = f.seen_round;
  r[kTPearl] = f.pearl;
  r[kTCountdown] = f.countdown;
  r[kTVisible] = in_view_[t];
  r[kTVisits] = f.visits;
  r[kTVisitRound] = f.visit_round;
  r[kTBodyIndex] = body_index_[t];
  r[kTPrevRound] = f.prev_round;
  r[kTPrevPearl] = f.prev_pearl;
  r[kTPrevCountdown] = f.prev_countdown;
  r[kTOcc] = f.occ;
  r[kTOccOrder] = f.occ_id < 0 || f.occ_id == spawn_.id ? 0 : f.occ_id < spawn_.id ? 1 : 2;
  r[kTOccFacing] = f.occ_facing;
  int32_t edges = 0, flags = 0;
  const int x = t % W, y = t / W;
  for (int d = 0; d < 4; ++d) {
    const EdgeFact& e = edges_[SideEdge(t, d)];
    const int32_t j = Dest(t, d);
    edges |= e.type << (2 * d);
    const bool seam = (d == 0 && y == 0) || (d == 2 && y == H - 1) || (d == 3 && x == 0) || (d == 1 && x == W - 1);
    if (e.type == kEdgeOpen && seam) flags |= 1 << d;
    if (e.type == kEdgePortal && j != kUnknown) flags |= 1 << (4 + d);
    dest[d] = j;
  }
  r[kTEdges] = edges;
  r[kTFlags] = flags;
  r[kTValid] = 1;
}

void Lifetime::EncodeWindow(WindowRows& rows) const {
  const int W = spawn_.width, H = spawn_.height;
  const int hx = last_head_ == kUnknown ? 0 : last_head_ % W, hy = last_head_ == kUnknown ? 0 : last_head_ / W;
  // shortest toroidal offset of a coordinate from the head's
  // (in [-n/2, n/2): with the grid's (off + n/2) * kGrid / n, every tile lands in cells 0..kGrid-1)
  auto off = [](int v, int h, int n) { int d = ((v - h) % n + n) % n; return d >= (n + 1) / 2 ? d - n : d; };
  auto cell = [](int dx, int dy) { return (dy + kWinRadius) * kWin + (dx + kWinRadius); };
  for (int dy = -kWinRadius; dy <= kWinRadius; ++dy)
    for (int dx = -kWinRadius; dx <= kWinRadius; ++dx) {
      const int32_t t = Wrap(hx + dx, hy + dy);
      const int c = cell(dx, dy);
      int32_t jt[4];
      EncodeTile(t, rows.win + static_cast<int64_t>(c) * kTileFields, jt);
      for (int d = 0; d < 4; ++d) {
        const int32_t j = jt[d];
        int32_t w = kUnknown;
        if (j != kUnknown) {
          if (j == Neighbour(t, d)) {                     // a plain step: the adjacent cell
            const int nx = dx + dragons::kDx[d], ny = dy + dragons::kDy[d];
            w = (std::abs(nx) <= kWinRadius && std::abs(ny) <= kWinRadius) ? cell(nx, ny) : kWinOutside;
          } else {                                        // a portal: its far end's nearest copy
            const int ox = off(j % W, hx, W), oy = off(j / W, hy, H);
            w = (std::abs(ox) <= kWinRadius && std::abs(oy) <= kWinRadius) ? cell(ox, oy) : kWinOutside;
          }
        }
        rows.wdest[c * 4 + d] = w;
      }
    }
  std::fill(rows.grid, rows.grid + kGridCells * kGridFields, 0);
  // cell of each column and row, head-relative (one divide per column / row, not per tile)
  std::vector<int> gxs(W), gys(H);
  for (int x = 0; x < W; ++x) gxs[x] = (off(x, hx, W) + W / 2) * kGrid / W;
  for (int y = 0; y < H; ++y) gys[y] = (off(y, hy, H) + H / 2) * kGrid / H;
  for (int y = 0; y < H; ++y) {
    int32_t* grow = rows.grid + gys[y] * kGrid * kGridFields;
    for (int x = 0; x < W; ++x) {
      const int32_t t = y * W + x;
      int32_t* g = grow + gxs[x] * kGridFields;
      const TileFact& f = tiles_[t];                // the same facts EncodeTile reads
      const bool seen = f.seen_round >= 0;
      ++g[kGTiles];
      if (seen) { ++g[kGSeen]; g[kGSeenRoundSum] += f.seen_round; }
      g[kGVisible] += in_view_[t];
      if (seen && f.pearl) ++g[kGPearl];
      if (seen && f.countdown != -1) ++g[kGBed];
      if (f.visits > 0) { ++g[kGVisited]; g[kGVisitRoundSum] += f.visit_round; }
      if (body_index_[t] >= 0) ++g[kGSelfBody];
      const int occ = f.occ;
      if (occ == kOccAllyHead || occ == kOccAllyBody) ++g[kGAlly];
      if (occ == kOccEnemyHead || occ == kOccEnemyBody) ++g[kGEnemy];
      if (occ == kOccEmpty) ++g[kGEmpty];
      for (int e = 2 * t; e < 2 * t + 2; ++e) {     // its canonical edges (north, west), each counted once
        const int type = edges_[e].type;
        g[kGKelp] += type == kEdgeKelp;
        g[kGPortal] += type == kEdgePortal;
      }
    }
  }
}

void Lifetime::Encode(const Observation& o, DecisionRows& rows) const {
  if (rows.n_pad < n_) throw std::invalid_argument("n_pad is smaller than the board");
  for (int32_t t = 0; t < n_; ++t)
    EncodeTile(t, rows.tile + static_cast<int64_t>(t) * kTileFields, rows.dest + static_cast<int64_t>(t) * 4);
  std::fill(rows.tile + static_cast<int64_t>(n_) * kTileFields, rows.tile + static_cast<int64_t>(rows.n_pad) * kTileFields, 0);
  std::fill(rows.dest + static_cast<int64_t>(n_) * 4, rows.dest + static_cast<int64_t>(rows.n_pad) * 4, kUnknown);
  EncodeRest(o, rows);
}

Lifetime Lifetime::Rebuilt(const Observation& o) const {
  const int W = spawn_.width;
  std::vector<int32_t> index(n_, -1);
  for (int i = static_cast<int>(body_.size()); i-- > 0;)
    if (body_[i] != kUnknown) index[body_[i]] = i;
  std::vector<uint8_t> in_view(n_, 0);
  for (int k = 0; k < kWindowTiles; ++k) in_view[o.tile_y[k] * W + o.tile_x[k]] = 1;
  // the from-scratch record of every tile: the same function over freshly derived body index and view
  Lifetime copy = *this;
  copy.body_index_ = index;
  copy.in_view_ = in_view;
  return copy;
}

void Lifetime::EncodeReference(const Observation& o, DecisionRows& rows) const {
  Rebuilt(o).Encode(o, rows);
}

void Lifetime::EncodeRest(const Observation& o, DecisionRows& rows) const {
  const int W = spawn_.width, H = spawn_.height;
  // entities by id; segments point at their entity row
  std::vector<int32_t> ids;
  for (const Segment& s : o.segments) ids.push_back(s.id);
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  for (size_t e = 0; e < ids.size(); ++e) {
    int32_t* r = rows.entity + e * kEntityFields;
    r[kEValid] = 1;
    r[kEHeadFacing] = -1;
    r[kEHeadX] = r[kEHeadY] = 0;
  }
  for (size_t k = 0; k < o.segments.size(); ++k) {
    const Segment& s = o.segments[k];
    const int e = static_cast<int>(std::lower_bound(ids.begin(), ids.end(), s.id) - ids.begin());
    int32_t* r = rows.entity + e * kEntityFields;
    r[kERelation] = s.id == spawn_.id ? 0 : s.team == spawn_.team ? 1 : 2;
    r[kEOrder] = s.id < spawn_.id ? 0 : s.id == spawn_.id ? 1 : 2;
    ++r[kECount];
    if (s.head) {
      r[kEHeadVisible] = 1;
      r[kEHeadX] = s.x;
      r[kEHeadY] = s.y;
      r[kEHeadFacing] = s.facing;
    }
    int32_t* g = rows.segment + k * kSegmentFields;
    g[0] = e;
    g[1] = s.x;
    g[2] = s.y;
    g[3] = 1;
  }

  const int32_t head = Wrap(o.head_x(), o.head_y());
  int32_t known = 0;
  for (int32_t b : body_) known += b != kUnknown;
  int32_t* s = rows.status;
  s[kSLength] = o.length;
  s[kSRound] = o.round;
  s[kSWidth] = W;
  s[kSHeight] = H;
  s[kSUnitCount] = o.unit_count;
  s[kSUnitLimit] = spawn_.unit_limit;
  s[kSKnownBody] = known;
  s[kSTailKnown] = body_.back() != kUnknown;
  s[kSTurnCount] = turns_;
  s[kSFacing] = o.facing;
  s[kSHeadX] = o.head_x();
  s[kSHeadY] = o.head_y();
  s[kSPrevValid] = prev_valid_;
  s[kSPrevKind] = prev_valid_ ? prev_.kind : 0;
  s[kSPrevSteps] = prev_valid_ && prev_.kind == 0 ? prev_.nsteps : 0;
  s[kSPrevSplit] = prev_valid_ && prev_.kind == 1 ? prev_.split : 0;
  s[kSPrevLength] = prev_valid_ ? cmd_length_ : 0;
  s[kSSeenTiles] = seen_tiles_;
  s[kSKnownEdges] = known_edges_;
  s[kSInboxCount] = static_cast<int32_t>(o.inbox.size());
  s[kSEchoPresent] = o.echo_present;
  s[kSHeadBed] = o.countdown[kWindowTiles / 2] != -1;
  s[kSHeadPearl] = o.pearl[kWindowTiles / 2];
  s[kSHeadTile] = head;
  s[kSPrevSonarMask] = prev_valid_ ? prev_.sonar_mask : 0;
  for (int i = 0; i < kMaxMoveTokens; ++i)
    rows.prev_dirs[i] = prev_valid_ && prev_.kind == 0 && i < prev_.nsteps ? prev_.dirs[i] : -1;
  for (int d = 0; d < 4; ++d) {
    const uint64_t w = prev_valid_ && (prev_.sonar_mask >> d & 1) ? prev_.words[d] : 0;
    for (int b = 0; b < 8; ++b) rows.prev_words[d * 8 + b] = static_cast<uint8_t>(w >> (8 * b));
  }
}

}  // namespace rl
