#include "memory.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace dragons::spatial {

namespace {

constexpr std::array<int, 4> kDx{0, 1, 0, -1};
constexpr std::array<int, 4> kDy{-1, 0, 1, 0};
const double kLogVisits = std::log(501.0);
constexpr double kPi = 3.14159265358979323846;

int Wrap(int v, int m) {
  const int r = v % m;
  return r < 0 ? r + m : r;
}

float Clip01(double v) { return static_cast<float>(std::min(std::max(v, 0.0), 1.0)); }

}  // namespace

// ---------------------------------------------------------------------------
// window

Window ReadWindow(const Game& game, int index, const std::vector<int>& portal_ids) {
  const Dragon& dragon = game.dragons()[index];
  const int w = game.width(), h = game.height();
  Window out;
  out.round = game.round();
  out.facing = static_cast<int>(dragon.facing);
  out.length = dragon.length();
  out.unit_count = game.AliveUnitCount(dragon.team);
  const Point head = dragon.head();
  out.head_x = head.x;
  out.head_y = head.y;
  for (int row = 0; row < kVision; ++row) {
    for (int col = 0; col < kVision; ++col) {
      const int i = row * kVision + col;
      const Point p{static_cast<int16_t>(Wrap(head.x - kRadius + col, w)),
                    static_cast<int16_t>(Wrap(head.y - kRadius + row, h))};
      out.xs[i] = p.x;
      out.ys[i] = p.y;
      const Tile& tile = game.TileAt(p);
      out.pearl[i] = tile.pearl ? 1 : 0;
      out.countdown[i] = tile.spawns ? tile.countdown : -1;
      for (int d = 0; d < 4; ++d) {
        const Dir dir = static_cast<Dir>(d);
        const std::optional<Point> step = game.TileAfterStep(p, dir);
        if (!step) {
          out.edge_kind[i][d] = EDGE_KELP;
          out.edge_portal[i][d] = -1;
          continue;
        }
        const Point plain{static_cast<int16_t>(Wrap(p.x + kDx[d], w)),
                          static_cast<int16_t>(Wrap(p.y + kDy[d], h))};
        if (*step == plain) {
          out.edge_kind[i][d] = EDGE_EMPTY;
          out.edge_portal[i][d] = -1;
        } else {
          out.edge_kind[i][d] = EDGE_PORTAL;
          // canonical owner of the crossed edge, as the map text indexes it
          int ox = p.x, oy = p.y, axis = 0;
          if (d == 2) oy = Wrap(p.y + 1, h);
          if (d == 3) axis = 1;
          if (d == 1) { axis = 1; ox = Wrap(p.x + 1, w); }
          out.edge_portal[i][d] = portal_ids[(axis ? w * h : 0) + oy * w + ox];
        }
      }
    }
  }
  for (const Dragon& other : game.dragons()) {
    if (!other.alive) continue;
    for (size_t s = 0; s < other.segments.size(); ++s) {
      const Point p = other.segments[s];
      const int col = Wrap(p.x - head.x + kRadius, w);
      const int row = Wrap(p.y - head.y + kRadius, h);
      if (col >= kVision || row >= kVision) continue;
      out.parts.push_back(Part{other.team, other.id, p.x, p.y,
                               static_cast<int>(game.TileFacing(p)), s == 0});
    }
  }
  out.inbox = dragon.inbox;
  return out;
}

// ---------------------------------------------------------------------------
// factual memory

FactualMemory::FactualMemory(int width, int height) : width_(width), height_(height) {
  const size_t n = static_cast<size_t>(width) * height;
  seen.assign(n, 0);
  last_seen_round.assign(n, -1);
  pearl.assign(n, 0);
  countdown.assign(n, 0);
  prev_pearl.assign(n, 0);
  prev_countdown.assign(n, 0);
  prev_seen_round.assign(n, -1);
  occupant_kind.assign(n, 0);
  occupant_id.assign(n, -1);
  occupant_facing.assign(n, 0);
  visit_count.assign(n, 0);
  last_visit_round.assign(n, -1);
  edge_known.assign(2 * n, 0);
  edge_type.assign(2 * n, 0);
  edge_portal.assign(2 * n, -1);
  visible.assign(n, 0);
}

CanonicalEdge FactualMemory::Canonical(int x, int y, int d) const {
  switch (d) {
    case 0: return {0, x, y};
    case 2: return {0, x, Wrap(y + 1, height_)};
    case 3: return {1, x, y};
    default: return {1, Wrap(x + 1, width_), y};
  }
}

void FactualMemory::Edge(int x, int y, int d, bool& known, uint8_t& kind, int32_t& portal) const {
  const CanonicalEdge e = Canonical(x, y, d);
  const size_t i = EdgeIndex(e.axis, e.x, e.y);
  known = edge_known[i];
  kind = edge_type[i];
  portal = edge_portal[i];
}

void FactualMemory::SetEdge(int x, int y, int d, uint8_t kind, int32_t portal) {
  const CanonicalEdge e = Canonical(x, y, d);
  const size_t i = EdgeIndex(e.axis, e.x, e.y);
  edge_known[i] = 1;
  edge_type[i] = kind;
  edge_portal[i] = portal;
  if (kind == EDGE_PORTAL) portals[portal].insert(e);
}

std::optional<CanonicalEdge> FactualMemory::PortalPartner(int x, int y, int d) const {
  const CanonicalEdge e = Canonical(x, y, d);
  const size_t i = EdgeIndex(e.axis, e.x, e.y);
  if (!edge_known[i] || edge_type[i] != EDGE_PORTAL) return std::nullopt;
  const auto it = portals.find(edge_portal[i]);
  if (it == portals.end()) return std::nullopt;
  std::vector<CanonicalEdge> others;
  for (const CanonicalEdge& o : it->second) {
    if (!(o == e)) others.push_back(o);
  }
  if (others.size() != 1) return std::nullopt;
  return others[0];
}

FactualMemory::StepKind FactualMemory::Step(int x, int y, int d, Cell& landing) const {
  bool known;
  uint8_t kind;
  int32_t portal;
  Edge(x, y, d, known, kind, portal);
  if (!known) return StepKind::Unknown;
  if (kind == EDGE_KELP) return StepKind::Kelp;
  if (kind == EDGE_EMPTY) {
    landing = Cell{static_cast<int16_t>(Wrap(x + kDx[d], width_)),
                   static_cast<int16_t>(Wrap(y + kDy[d], height_))};
    return StepKind::Empty;
  }
  const std::optional<CanonicalEdge> partner = PortalPartner(x, y, d);
  if (!partner) return StepKind::PortalUnknown;
  // The partner is the north (axis 0) or west (axis 1) edge of (px, py):
  // entering from the positive side lands on that tile, from the negative
  // side on the tile before it -- the engine's TileAfterStep.
  if (partner->axis == 1) {
    landing = Cell{static_cast<int16_t>(d == 1 ? partner->x : Wrap(partner->x - 1, width_)),
                   static_cast<int16_t>(partner->y)};
  } else {
    landing = Cell{static_cast<int16_t>(partner->x),
                   static_cast<int16_t>(d == 2 ? partner->y : Wrap(partner->y - 1, height_))};
  }
  return StepKind::Portal;
}

std::vector<std::pair<CanonicalEdge, CanonicalEdge>> FactualMemory::ConfirmedPortalPairs() const {
  std::vector<std::pair<CanonicalEdge, CanonicalEdge>> out;
  for (const auto& [pid, ends] : portals) {      // std::map: sorted by id
    if (ends.size() != 2) continue;
    auto it = ends.begin();
    const CanonicalEdge a = *it++;
    const CanonicalEdge b = *it;
    out.emplace_back(a, b);                        // std::set: a < b
  }
  return out;
}

std::vector<CanonicalEdge> FactualMemory::UnpairedPortals() const {
  std::vector<CanonicalEdge> out;
  for (const auto& [pid, ends] : portals) {
    if (ends.size() == 2) continue;
    for (const CanonicalEdge& e : ends) out.push_back(e);
  }
  std::sort(out.begin(), out.end());
  return out;
}

void FactualMemory::Observe(const Window& w, int dragon_id, int team) {
  const int rnd = w.round;
  for (int i = 0; i < kTiles; ++i) {
    const size_t k = Index(w.xs[i], w.ys[i]);
    const bool fresh = last_seen_round[k] != rnd;
    if (seen[k] && fresh) {
      prev_pearl[k] = pearl[k];
      prev_countdown[k] = countdown[k];
      prev_seen_round[k] = last_seen_round[k];
    }
  }
  for (int i = 0; i < kTiles; ++i) {
    const size_t k = Index(w.xs[i], w.ys[i]);
    seen[k] = 1;
    last_seen_round[k] = static_cast<int16_t>(rnd);
    pearl[k] = w.pearl[i] != 0;
    countdown[k] = w.countdown[i];
    occupant_kind[k] = OCC_EMPTY;
    occupant_id[k] = -1;
    occupant_facing[k] = 0;
  }
  for (const Part& p : w.parts) {
    uint8_t kind;
    if (p.dragon_id == dragon_id) kind = p.is_head ? OCC_SELF_HEAD : OCC_SELF_BODY;
    else if (p.team == team) kind = p.is_head ? OCC_ALLY_HEAD : OCC_ALLY_BODY;
    else kind = p.is_head ? OCC_ENEMY_HEAD : OCC_ENEMY_BODY;
    const size_t k = Index(p.x, p.y);
    occupant_kind[k] = kind;
    occupant_id[k] = p.dragon_id;
    occupant_facing[k] = static_cast<uint8_t>(p.facing);
  }
  for (int i = 0; i < kTiles; ++i) {
    for (int d = 0; d < 4; ++d) {
      SetEdge(w.xs[i], w.ys[i], d, w.edge_kind[i][d],
              w.edge_kind[i][d] == EDGE_PORTAL ? w.edge_portal[i][d] : -1);
    }
  }
  visible_round = rnd;
  std::fill(visible.begin(), visible.end(), 0);
  for (int i = 0; i < kTiles; ++i) visible[Index(w.xs[i], w.ys[i])] = 1;
}

void FactualMemory::MarkVisited(Cell c, int round) {
  const size_t k = Index(c.x, c.y);
  if (last_visit_round[k] == round) return;
  visit_count[k] = static_cast<uint16_t>(std::min<int>(visit_count[k] + 1, 65535));
  last_visit_round[k] = static_cast<int16_t>(round);
}

// ---------------------------------------------------------------------------
// body

BodyHistory::BodyHistory(int length, Cell head) {
  cells.assign(length, std::nullopt);
  cells[0] = head;
}

float BodyHistory::KnownFraction() const {
  int known = 0;
  for (const auto& c : cells) known += c.has_value();
  return static_cast<float>(known) / static_cast<float>(std::max<int>(cells.size(), 1));
}

std::map<Cell, int> BodyHistory::IndexOf() const {
  std::map<Cell, int> out;
  for (size_t i = 0; i < cells.size(); ++i) {
    if (cells[i]) out.emplace(*cells[i], static_cast<int>(i));
  }
  return out;
}

void BodyHistory::ApplyMove(const std::vector<std::optional<Cell>>& path, int new_length) {
  std::vector<std::optional<Cell>> next(path.rbegin(), path.rend());
  next.insert(next.end(), cells.begin(), cells.end());
  next.resize(new_length, std::nullopt);
  cells = std::move(next);
}

void BodyHistory::ApplySplit(int detached) {
  cells.resize(std::max<int>(0, static_cast<int>(cells.size()) - detached));
}

void BodyHistory::Reconcile(const Window& w, int dragon_id) {
  cells[0] = Cell{static_cast<int16_t>(w.head_x), static_cast<int16_t>(w.head_y)};
  std::set<Cell> own, visible, seen_cells;
  for (const Part& p : w.parts) {
    if (p.dragon_id == dragon_id) own.insert(Cell{static_cast<int16_t>(p.x), static_cast<int16_t>(p.y)});
  }
  for (int i = 0; i < kTiles; ++i) visible.insert(Cell{w.xs[i], w.ys[i]});
  for (size_t i = 1; i < cells.size(); ++i) {
    if (!cells[i]) continue;
    const Cell c = *cells[i];
    if ((visible.count(c) && !own.count(c)) || seen_cells.count(c) || c == *cells[0]) {
      cells[i] = std::nullopt;
    } else {
      seen_cells.insert(c);
    }
  }
  if (static_cast<int>(cells.size()) != w.length) cells.resize(w.length, std::nullopt);
}

// ---------------------------------------------------------------------------
// lifetime

LifetimeState::LifetimeState(int dragon_id, int team, int width, int height, int unit_limit,
                             int max_steps)
    : facts(width, height), unit_limit(unit_limit), max_steps(max_steps),
      dragon_id_(dragon_id), team_(team) {}

std::vector<std::optional<Cell>> LifetimeState::TracePath(Cell start, const std::string& dirs,
                                                          bool& known) const {
  std::vector<std::optional<Cell>> path;
  std::optional<Cell> at = start;
  known = true;
  for (char ch : dirs) {
    if (!at) { path.push_back(std::nullopt); continue; }
    const int d = static_cast<int>(std::string("NESW").find(ch));
    Cell landing;
    const auto kind = facts.Step(at->x, at->y, d, landing);
    if (kind == FactualMemory::StepKind::Empty || kind == FactualMemory::StepKind::Portal) {
      at = landing;
      path.push_back(landing);
    } else {
      at = std::nullopt;
      known = false;
      path.push_back(std::nullopt);
    }
  }
  return path;
}

void LifetimeState::CompletePending(const Window& w) {
  if (!pending) return;
  const PendingCommand p = *pending;
  pending.reset();
  const Cell now{static_cast<int16_t>(w.head_x), static_cast<int16_t>(w.head_y)};
  CommandRecord rec;
  rec.round = p.round;
  rec.old_head = p.head;
  rec.new_head = now;
  rec.old_heading = p.heading;
  rec.new_heading = w.facing;
  rec.old_length = p.length;
  rec.new_length = w.length;
  if (!p.is_move) {
    rec.is_move = false;
    rec.split_n = p.split_n;
    rec.intermediates_known = true;
  } else {
    bool known;
    std::vector<std::optional<Cell>> path = TracePath(p.head, p.dirs, known);
    path.back() = now;                       // the observed final head is always known
    known = true;
    for (const auto& c : path) known = known && c.has_value();
    body->ApplyMove(path, w.length);
    for (const auto& c : path) {
      if (c) facts.MarkVisited(*c, p.round);
    }
    rec.is_move = true;
    rec.dirs = p.dirs;
    rec.intermediates_known = known;
  }
  history.push_back(rec);
  while (static_cast<int>(history.size()) > kHistoryTurns) history.pop_front();
}

void LifetimeState::CommitMove(const std::string& dirs) {
  pending = PendingCommand{true, dirs, 0, round, head, heading, length};
  ++own_turns;
}

void LifetimeState::CommitSplit(int n) {
  pending = PendingCommand{false, "", n, round, head, heading, length};
  ++own_turns;
  body->ApplySplit(n);
}

void LifetimeState::Materialize(int round_num, float* out) const {
  const int W = facts.width(), H = facts.height();
  const size_t n = static_cast<size_t>(W) * H;
  std::fill(out, out + kFactChannels * n, 0.0f);
  auto ch = [&](int c) { return out + static_cast<size_t>(c) * n; };
  const int own = dragon_id_;
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const size_t k = facts.Index(x, y);
      const bool seen = facts.seen[k];
      ch(F_SEEN)[k] = seen;
      ch(F_VISIBLE)[k] = facts.visible_round == round_num && facts.visible[k];
      if (seen) {
        const int age = std::min(std::max(round_num - facts.last_seen_round[k], 0), kMaxRounds);
        ch(F_AGE_SEEN)[k] = static_cast<float>(age) / kMaxRounds;
        ch(F_PEARL)[k] = facts.pearl[k];
        if (facts.countdown[k] >= 0) {
          ch(F_COUNTDOWN)[k] = 1.0f / (1.0f + static_cast<float>(facts.countdown[k]));
        } else {
          ch(F_NEVER)[k] = 1.0f;
        }
        ch(F_OCC + facts.occupant_kind[k])[k] = 1.0f;
        const bool occupied = facts.occupant_kind[k] != OCC_EMPTY;
        if (occupied) {
          ch(F_FACING + facts.occupant_facing[k])[k] = 1.0f;
          const int id = facts.occupant_id[k];
          if (id != -1 && id != own) {
            ch(F_ID_EARLIER)[k] = id < own;
            ch(F_ID_LATER)[k] = id > own;
          }
        }
      }
      const int dirs_x[4] = {x, Wrap(x + 1, W), x, x};
      const int dirs_y[4] = {y, y, Wrap(y + 1, H), y};
      const int axis[4] = {0, 1, 0, 1};
      for (int d = 0; d < 4; ++d) {
        const size_t e = facts.EdgeIndex(axis[d], dirs_x[d], dirs_y[d]);
        if (!facts.edge_known[e]) continue;
        ch(F_EDGE_KNOWN + d)[k] = 1.0f;
        ch(F_KELP + d)[k] = facts.edge_type[e] == EDGE_KELP;
        ch(F_PORTAL + d)[k] = facts.edge_type[e] == EDGE_PORTAL;
      }
      if (facts.visit_count[k] > 0) {
        ch(F_VISIT_LOG)[k] = Clip01(std::log1p(static_cast<double>(facts.visit_count[k])) / kLogVisits);
        const int vage = std::min(std::max(round_num - facts.last_visit_round[k], 0), kMaxRounds);
        ch(F_AGE_VISIT)[k] = static_cast<float>(vage) / kMaxRounds;
        ch(F_VISITED)[k] = 1.0f;
      }
      if (facts.prev_seen_round[k] >= 0) {
        ch(F_PREV_EXISTS)[k] = 1.0f;
        ch(F_PREV_PEARL)[k] = facts.prev_pearl[k];
        if (facts.prev_countdown[k] >= 0) {
          ch(F_PREV_COUNTDOWN)[k] = 1.0f / (1.0f + static_cast<float>(facts.prev_countdown[k]));
        }
        const int interval = std::min(std::max(facts.last_seen_round[k] - facts.prev_seen_round[k], 0), kMaxRounds);
        ch(F_INTERVAL)[k] = static_cast<float>(interval) / kMaxRounds;
      }
    }
  }
  if (body) {
    const int L = body->size();
    for (int i = 0; i < L; ++i) {
      if (!body->cells[i]) continue;
      const size_t k = facts.Index(body->cells[i]->x, body->cells[i]->y);
      ch(F_BODY_KNOWN)[k] = 1.0f;
      ch(F_BODY_INDEX)[k] = static_cast<float>(i) / static_cast<float>(std::max(L - 1, 1));
    }
    if (body->TailKnown() && L > 1) {
      ch(F_TAIL_KNOWN)[facts.Index(body->cells.back()->x, body->cells.back()->y)] = 1.0f;
    }
    ch(F_HEAD)[facts.Index(body->cells[0]->x, body->cells[0]->y)] = 1.0f;
  }
}

void LifetimeState::Status(const Window& w, float* out) const {
  std::fill(out, out + kStatus, 0.0f);
  const int W = facts.width(), H = facts.height();
  const double area = static_cast<double>(W) * H;
  const int L = w.length;
  out[0] = static_cast<float>(L / area);
  out[1] = static_cast<float>(std::log1p(static_cast<double>(L)) / std::log1p(area));
  out[2] = static_cast<float>(static_cast<double>(w.round) / kLastRound);
  out[3] = static_cast<float>(static_cast<double>(std::max(kLastRound - w.round, 0)) / kMaxRounds);
  out[4] = static_cast<float>(W / static_cast<double>(kMaxBoard));
  out[5] = static_cast<float>(H / static_cast<double>(kMaxBoard));
  out[6] = static_cast<float>(static_cast<double>(w.unit_count) / std::max(unit_limit, 1));
  out[7 + w.facing] = 1.0f;
  out[11] = split_legal;
  out[12] = own_turns == 0;
  out[13] = static_cast<float>(static_cast<double>(own_turns) / kMaxRounds);
  out[14] = body->KnownFraction();
  out[15] = body->TailKnown();
  const double ax = 2 * kPi * w.head_x / W, ay = 2 * kPi * w.head_y / H;
  out[16] = static_cast<float>(std::sin(ax));
  out[17] = static_cast<float>(std::cos(ax));
  out[18] = static_cast<float>(std::sin(ay));
  out[19] = static_cast<float>(std::cos(ay));
  if (history.empty()) {
    out[20] = 1.0f;
  } else {
    const CommandRecord& prev = history.back();
    if (!prev.is_move) {
      out[22] = 1.0f;
      out[23] = static_cast<float>(static_cast<double>(prev.split_n) / std::max(prev.old_length, 1));
      out[24] = static_cast<float>(static_cast<double>(prev.old_length - prev.split_n) / std::max(prev.old_length, 1));
      out[34] = 1.0f;
    } else {
      out[21] = 1.0f;
      out[25] = static_cast<float>(static_cast<double>(prev.dirs.size()) / max_steps);
      for (char c : prev.dirs) out[26 + std::string("NESW").find(c)] += static_cast<float>(1.0 / max_steps);
      if (!prev.dirs.empty()) out[30 + std::string("NESW").find(prev.dirs.back())] = 1.0f;
      out[34] = prev.intermediates_known;
    }
  }
  out[35] = static_cast<float>(std::min<size_t>(w.inbox.size(), 8) / 8.0);
}

void LifetimeState::HistoryFeatures(int round_num, float* out, uint8_t* mask) const {
  std::fill(out, out + kHistoryTurns * kHistoryFeatures, 0.0f);
  std::fill(mask, mask + kHistoryTurns, 0);
  const int W = facts.width(), H = facts.height();
  const double area = static_cast<double>(W) * H;
  const int start = kHistoryTurns - static_cast<int>(history.size());
  int j = 0;
  for (const CommandRecord& rec : history) {
    float* row = out + static_cast<size_t>(start + j) * kHistoryFeatures;
    mask[start + j] = 1;
    ++j;
    row[0] = rec.is_move;
    row[1] = !rec.is_move;
    const Cell heads[2] = {rec.old_head, rec.new_head};
    for (int b = 0; b < 2; ++b) {
      const double ax = 2 * kPi * heads[b].x / W, ay = 2 * kPi * heads[b].y / H;
      row[2 + 4 * b] = static_cast<float>(std::sin(ax));
      row[3 + 4 * b] = static_cast<float>(std::cos(ax));
      row[4 + 4 * b] = static_cast<float>(std::sin(ay));
      row[5 + 4 * b] = static_cast<float>(std::cos(ay));
    }
    row[10 + rec.old_heading] = 1.0f;
    row[14 + rec.new_heading] = 1.0f;
    row[18] = static_cast<float>(static_cast<double>(rec.round) / kMaxRounds);
    row[19] = static_cast<float>(static_cast<double>(std::max(round_num - rec.round, 0)) / kMaxRounds);
    row[20] = static_cast<float>(rec.old_length / area);
    row[21] = static_cast<float>(rec.new_length / area);
    row[22] = static_cast<float>(static_cast<double>(rec.new_length - rec.old_length) / max_steps);
    row[23] = static_cast<float>(static_cast<double>(rec.dirs.size()) / max_steps);
    for (char c : rec.dirs) row[24 + std::string("NESW").find(c)] += static_cast<float>(1.0 / max_steps);
    row[28] = static_cast<float>(static_cast<double>(rec.split_n) / std::max(rec.old_length, 1));
    row[29] = rec.intermediates_known;
  }
}

void LifetimeState::Entities(const Window& w, float* out, uint8_t* mask) const {
  std::fill(out, out + kMaxEntities * kEntityFeatures, 0.0f);
  std::fill(mask, mask + kMaxEntities, 0);
  std::map<int, std::vector<const Part*>> groups;
  for (const Part& p : w.parts) {
    if (p.dragon_id != dragon_id_) groups[p.dragon_id].push_back(&p);
  }
  const int W = facts.width(), H = facts.height();
  int slot = 0;
  for (const auto& [other_id, parts] : groups) {
    if (slot >= kMaxEntities) break;
    float* row = out + static_cast<size_t>(slot) * kEntityFeatures;
    mask[slot] = 1;
    ++slot;
    const int n = static_cast<int>(parts.size());
    row[0] = static_cast<float>(static_cast<double>(n) / kTiles);
    row[1] = static_cast<float>(std::log1p(static_cast<double>(n)) / std::log(50.0));
    double sum_dx = 0, sum_dy = 0;
    int count = 0;
    double min_dist = 1e9;
    for (const Part* p : parts) {
      const int col = Wrap(p->x - w.head_x + kRadius, W);
      const int rowi = Wrap(p->y - w.head_y + kRadius, H);
      if (col >= kVision || rowi >= kVision) continue;
      const int dx = col - kRadius, dy = rowi - kRadius;
      sum_dx += dx;
      sum_dy += dy;
      ++count;
      min_dist = std::min(min_dist, static_cast<double>(std::max(std::abs(dx), std::abs(dy))));
      if (p->is_head) {
        row[2] = 1.0f;
        row[3 + p->facing] = 1.0f;
        row[7] = static_cast<float>(dx / static_cast<double>(kRadius));
        row[8] = static_cast<float>(dy / static_cast<double>(kRadius));
      }
    }
    if (count) {
      // numpy float32 mean: sum in float32 then divide, then / 3
      row[9] = static_cast<float>(static_cast<float>(static_cast<float>(sum_dx) / count) / kRadius);
      row[10] = static_cast<float>(static_cast<float>(static_cast<float>(sum_dy) / count) / kRadius);
      row[11] = static_cast<float>(min_dist / kRadius);
    }
    const bool ally = parts[0]->team == team_;
    row[12] = ally;
    row[13] = !ally;
    row[14] = other_id < dragon_id_;
    row[15] = other_id > dragon_id_;
  }
}

void LifetimeState::Prepare(const Window& w, const Buffers& out) {
  const int W = facts.width(), H = facts.height();
  const size_t n = static_cast<size_t>(W) * H;
  const bool redelivery = w.round == last_processed_round;
  // dense parts are optional: the trainer derives them on the GPU
  const bool dense = out.facts != nullptr;
  if (dense && !(redelivery && prewrite_round_ == w.round)) {
    std::vector<float> pre(kFactChannels * n);
    Materialize(w.round, pre.data());
    prewrite_prev_.assign(kTiles * kFactChannels, 0.0f);
    prewrite_age_.assign(kTiles, 0.0f);
    for (int i = 0; i < kTiles; ++i) {
      const size_t k = facts.Index(w.xs[i], w.ys[i]);
      for (int c = 0; c < kFactChannels; ++c) {
        prewrite_prev_[static_cast<size_t>(i) * kFactChannels + c] = pre[static_cast<size_t>(c) * n + k];
      }
      const int last = facts.last_seen_round[k];
      prewrite_age_[i] = last >= 0
          ? static_cast<float>(std::min(std::max(w.round - last, 0), kMaxRounds)) / kMaxRounds
          : 0.0f;
    }
    prewrite_round_ = w.round;
  }
  if (dense) {
    std::copy(prewrite_prev_.begin(), prewrite_prev_.end(), out.visible_prev);
    std::copy(prewrite_age_.begin(), prewrite_age_.end(), out.visible_age);
  }

  head = Cell{static_cast<int16_t>(w.head_x), static_cast<int16_t>(w.head_y)};
  heading = w.facing;
  length = w.length;
  unit_count = w.unit_count;
  round = w.round;
  split_legal = (w.length - kMinSize >= kMinSize) && (w.unit_count < unit_limit);
  if (!redelivery) {
    if (!body) {
      body.emplace(w.length, head);
    } else {
      CompletePending(w);
      body->Reconcile(w, dragon_id_);
    }
    facts.Observe(w, dragon_id_, team_);
    facts.MarkVisited(head, w.round);
    inbox = w.inbox;
    last_processed_round = w.round;
  }

  if (dense) {
    Materialize(w.round, out.facts);
    for (int c = 0; c < kFactChannels; ++c) {
      for (int i = 0; i < kTiles; ++i) {
        out.local[static_cast<size_t>(c) * kTiles + i] = out.facts[static_cast<size_t>(c) * n + facts.Index(w.xs[i], w.ys[i])];
      }
    }
  }
  for (int i = 0; i < kTiles; ++i) {
    out.visible_xy[2 * i] = w.xs[i];
    out.visible_xy[2 * i + 1] = w.ys[i];
  }
  Status(w, out.status);
  HistoryFeatures(w.round, out.history, out.history_mask);
  Entities(w, out.entities, out.entity_mask);
  portal_pairs.clear();
  portal_axis.clear();
  for (const auto& [a, b] : facts.ConfirmedPortalPairs()) {
    portal_pairs.push_back({a.y, a.x, b.y, b.x});
    portal_axis.push_back(a.axis);
  }
  portal_unpaired.clear();
  for (const CanonicalEdge& e : facts.UnpairedPortals()) portal_unpaired.push_back({e.y, e.x});
}

}  // namespace dragons::spatial
