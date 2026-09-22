#include "arena.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <sstream>

#include <torch/cuda.h>
#ifdef USE_CUDA
#include <c10/cuda/CUDAStream.h>
#endif

#include "agent/encoder.hpp"

namespace dragons::train {

namespace sp = spatial;
using sp::Cell;
using sp::LifetimeState;

namespace {

constexpr double kPi = 3.14159265358979323846;

std::vector<int> PortalIds(const std::string& text, int w, int h) {
  std::vector<int> ids(2 * w * h, -1);
  std::istringstream in(text);
  std::string line;
  const int stride = 2 * (w + 1);
  while (std::getline(in, line)) {
    if (line.rfind("EDGE ", 0) != 0) continue;
    std::istringstream f(line);
    std::string key;
    int index, kind, pid;
    f >> key >> index >> kind >> pid;
    if (kind != 2) continue;
    int y = index / stride, off = index % stride;
    const bool vertical = off >= w + 1;
    if (vertical) off -= w + 1;
    ids[(vertical ? w * h : 0) + (y % h) * w + (off % w)] = pid;
  }
  return ids;
}

}  // namespace

ArenaGroup::ArenaGroup(int width, int height, std::vector<std::string> maps, const GroupOptions& opts,
                       Model& model, torch::Device device)
    : W_(width), H_(height), opts_(opts), model_(model), device_(device), maps_(std::move(maps)),
      rng_(opts.seed) {
  const int A = opts_.arenas, S = opts_.max_slots;
  for (size_t m = 0; m < maps_.size(); ++m) {
    // parsed once; the engine itself seeds pearls from a constant, so copies
    // sharing a stream match its determinism and variety comes from the maps
    prototypes_.push_back(Game::FromMapText(maps_[m], opts_.seed * 7919 + m));
    prototype_portals_.push_back(PortalIds(maps_[m], W_, H_));
  }
  games_.reserve(A);
  for (int a = 0; a < A; ++a) {
    games_.push_back(prototypes_[a % maps_.size()]);
    portal_ids_.push_back(prototype_portals_[a % maps_.size()]);
  }
  slots_.resize(A);
  for (auto& row : slots_) row.resize(S);
  slot_of_.assign(A, {});
  free_slots_.resize(A);
  for (auto& f : free_slots_) { f.clear(); for (int i = S - 1; i >= 0; --i) f.push_back(i); }
  known_dragons_.assign(A, 0);
  known_deaths_.assign(A, 0);
  cursor_.assign(A, -1);
  rounds_left_.assign(A, 0);
  record_of_.assign(A, -1);
  for (int a = 0; a < std::min(A, opts_.record_arenas); ++a) {
    record_of_[a] = static_cast<int>(records_.size());
    records_.push_back(ArenaRecord{{}, {}, a});
  }
  const auto fopt = torch::TensorOptions().dtype(torch::kFloat).device(device_);
  raw_ = torch::zeros({static_cast<int64_t>(A) * S, H_, W_, kRaw}, fopt);   // channel-last
  ResetSlots(raw_, torch::arange(static_cast<int64_t>(A) * S, torch::kLong));
  M_ = torch::zeros({static_cast<int64_t>(A) * S, kLatent, H_, W_}, fopt);
  h_ = torch::zeros({static_cast<int64_t>(A) * S, kHidden}, fopt);
  for (int a = 0; a < A; ++a) SpawnNew(a);
}

void ArenaGroup::ResetArena(int a) {
  const int m = static_cast<int>(rng_() % maps_.size());
  games_[a] = prototypes_[m];
  portal_ids_[a] = prototype_portals_[m];
  for (Slot& s : slots_[a]) s = Slot{};
  slot_of_[a].clear();
  free_slots_[a].clear();
  for (int i = opts_.max_slots - 1; i >= 0; --i) free_slots_[a].push_back(i);
  known_dragons_[a] = 0;
  known_deaths_[a] = 0;
  cursor_[a] = -1;
  const int S = opts_.max_slots;
  for (int i = 0; i < S; ++i) pending_reset_.push_back(static_cast<int64_t>(a) * S + i);
  if (record_of_[a] >= 0) {
    records_[record_of_[a]] = ArenaRecord{{}, {}, a};
  }
  SpawnNew(a);
}

int ArenaGroup::ResetFinished(bool all) {
  const auto t0 = std::chrono::steady_clock::now();
  int n = 0;
  for (int a = 0; a < arenas(); ++a) {
    if (all || games_[a].Finished() || games_[a].round() >= Game::kMaxRounds) {
      ResetArena(a);
      ++n;
    }
  }
  timing_.reset += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return n;
}

bool ArenaGroup::AllFinished() const {
  for (const Game& g : games_) if (!g.Finished() && g.round() < Game::kMaxRounds) return false;
  return true;
}

void ArenaGroup::SpawnNew(int a) {
  Game& game = games_[a];
  while (known_dragons_[a] < static_cast<int>(game.dragons().size())) {
    const int i = known_dragons_[a];
    const Dragon& d = game.dragons()[i];
    if (free_slots_[a].empty()) throw std::runtime_error("more living dragons than slots; lower UNIT_LIMIT");
    slot_of_[a].push_back(free_slots_[a].back());
    free_slots_[a].pop_back();
    max_index_ = std::max(max_index_, i + 1);
    Slot& s = SlotFor(a, i);
    s = Slot{};
    s.state = std::make_unique<LifetimeState>(d.id, d.team, W_, H_, game.unit_limit(), opts_.max_steps);
    // GPU-side reset of the slot is batched: one call per batch, not per child
    pending_reset_.push_back(SlotIndex(a, i));
    if (record_of_[a] >= 0) {
      const bool learner = d.team == 0 || opts_.self_play;
      records_[record_of_[a]].lifetimes[d.id] = LifetimeRecord{d.id, d.team, learner, {}, {}, false};
    }
    ++known_dragons_[a];
  }
}

void ArenaGroup::Reap(int a) {
  const auto& deaths = games_[a].deaths();
  while (known_deaths_[a] < static_cast<int>(deaths.size())) {
    const DeathEvent& e = deaths[known_deaths_[a]];
    // free the dead dragon's slot; its index stays mapped to nothing usable
    for (int i = 0; i < static_cast<int>(games_[a].dragons().size()); ++i) {
      if (games_[a].dragons()[i].id == e.dragon_id && i < static_cast<int>(slot_of_[a].size()) && slot_of_[a][i] >= 0) {
        slots_[a][slot_of_[a][i]] = Slot{};
        free_slots_[a].push_back(slot_of_[a][i]);
        slot_of_[a][i] = -1;
      }
    }
    if (record_of_[a] >= 0) {
      auto it = records_[record_of_[a]].lifetimes.find(e.dragon_id);
      if (it != records_[record_of_[a]].lifetimes.end()) it->second.dead = true;
    }
    ++known_deaths_[a];
  }
}

void ArenaGroup::RecordEvent(int a, int index) {
  const Game& game = games_[a];
  ArenaRecord& rec = records_[record_of_[a]];
  EventRecord ev;
  ev.round = game.round();
  ev.acting_team = game.dragons()[index].team;
  const double area = static_cast<double>(W_) * H_;
  torch::Tensor grid = torch::zeros({20, H_, W_}, torch::kFloat);
  auto g = grid.accessor<float, 3>();
  std::vector<uint8_t> pearl(W_ * H_), north(W_ * H_), west(W_ * H_);
  std::vector<int32_t> countdown(W_ * H_);
  DumpTiles(game, pearl.data(), countdown.data());
  DumpEdges(game, north.data(), west.data());
  for (int y = 0; y < H_; ++y) {
    for (int x = 0; x < W_; ++x) {
      const int k = y * W_ + x;
      g[0][y][x] = pearl[k];
      g[1][y][x] = countdown[k] >= 0 ? 1.0f / (1.0f + countdown[k]) : 0.0f;
      g[2][y][x] = countdown[k] < 0;
      const uint8_t kinds[4] = {north[k], west[y * W_ + (x + 1) % W_], north[((y + 1) % H_) * W_ + x], west[k]};
      for (int d = 0; d < 4; ++d) {
        g[3 + d][y][x] = kinds[d] == 1;
        g[7 + d][y][x] = kinds[d] == 2;
      }
    }
  }
  std::vector<int> alive;
  for (int i = 0; i < static_cast<int>(game.dragons().size()); ++i) if (game.dragons()[i].alive) alive.push_back(i);
  const int N = std::max<int>(1, static_cast<int>(alive.size()));
  ev.dragons = torch::zeros({N, 16}, torch::kFloat);
  ev.dragon_team = torch::zeros({N}, torch::kLong);
  ev.team_feats = torch::zeros({2, 4}, torch::kFloat);
  auto df = ev.dragons.accessor<float, 2>();
  auto dt = ev.dragon_team.accessor<int64_t, 1>();
  auto tf = ev.team_feats.accessor<float, 2>();
  for (size_t s = 0; s < alive.size(); ++s) {
    const int i = alive[s];
    const Dragon& d = game.dragons()[i];
    const Point head = d.head();
    const int team = d.team;
    g[team == 0 ? 11 : 13][head.y][head.x] = 1.0f;
    for (size_t q = 1; q < d.segments.size(); ++q) g[team == 0 ? 12 : 14][d.segments[q].y][d.segments[q].x] = 1.0f;
    g[15 + static_cast<int>(d.facing)][head.y][head.x] = 1.0f;
    if (i == index) g[19][head.y][head.x] = 1.0f;
    const int L = d.length();
    const double ax = 2 * kPi * head.x / W_, ay = 2 * kPi * head.y / H_;
    df[s][0] = std::sin(ax); df[s][1] = std::cos(ax); df[s][2] = std::sin(ay); df[s][3] = std::cos(ay);
    df[s][4 + static_cast<int>(d.facing)] = 1.0f;
    df[s][8] = L / area;
    df[s][9] = std::log1p(static_cast<double>(L)) / std::log1p(area);
    df[s][10] = static_cast<double>(i) / std::max<int>(1, static_cast<int>(game.dragons().size()));
    df[s][11] = i == index;
    const Slot& sl = slots_[a][slot_of_[a][i]];
    if (sl.state && sl.state->body) {
      double seen = 0;
      for (uint8_t v : sl.state->facts.seen) seen += v;
      df[s][12] = seen / area;
      df[s][13] = sl.state->body->KnownFraction();
    }
    // |h| summaries are filled in from the GPU by the caller (PlaySlot)
    dt[s] = team;
    tf[team][0] = std::max<float>(tf[team][0], L / area);
    tf[team][1] += L / area;
  }
  for (int team = 0; team < 2; ++team) {
    tf[team][2] = static_cast<float>(game.AliveUnitCount(team)) / game.unit_limit();
    tf[team][3] = 0.0f;   // opponent policy id; self-play only for now
  }
  ev.phase = torch::tensor({game.round() / 500.0f, std::max(499 - game.round(), 0) / 500.0f,
                            static_cast<float>(ev.acting_team == 0)}, torch::kFloat);
  ev.map_grid = grid.to(torch::kHalf);
  rec.events.push_back(std::move(ev));
}

void ArenaGroup::EndRoundFor(int a) {
  Game& game = games_[a];
  game.EndRound();
  --rounds_left_[a];
  cursor_[a] = -1;
  if (record_of_[a] >= 0 && (game.Finished() || game.round() >= Game::kMaxRounds)) {
    ArenaRecord& rec = records_[record_of_[a]];
    if (!rec.events.empty()) {
      EventRecord& last = rec.events.back();
      last.terminal = true;
      const Outcome o = game.Result(true);
      if (o == Outcome::TeamA) { last.reward[0] = 1.0f; last.reward[1] = -1.0f; }
      else if (o == Outcome::TeamB) { last.reward[0] = -1.0f; last.reward[1] = 1.0f; }
    }
  }
}

// Every arena contributes its next actor, so batches stay full while each
// arena still acts in exact id order and advances its own rounds.
void ArenaGroup::PlaySegment(int rounds) {
  using Clock = std::chrono::steady_clock;
  const auto seg_start = Clock::now();
  rounds_left_.assign(arenas(), rounds);
  std::vector<int> rows, indices;
  while (true) {
    const auto scan_start = Clock::now();
    rows.clear();
    indices.clear();
    for (int a = 0; a < arenas(); ++a) {
      Game& game = games_[a];
      while (true) {
        // Finished() is a pure function of state and turns true the moment a
        // team is eliminated; the engine still plays the round out and decides
        // afterwards, so a round in progress always completes
        if (cursor_[a] < 0) {
          if (game.Finished() || game.round() >= Game::kMaxRounds || rounds_left_[a] <= 0) break;
          game.BeginRound();
          cursor_[a] = 0;
        }
        SpawnNew(a);
        int k = cursor_[a];
        while (k < static_cast<int>(game.dragons().size()) &&
               !(game.dragons()[k].alive && slot_of_[a][k] >= 0)) ++k;
        if (k < static_cast<int>(game.dragons().size())) {
          rows.push_back(a);
          indices.push_back(k);
          break;
        }
        EndRoundFor(a);
      }
    }
    timing_.scan += std::chrono::duration<double>(Clock::now() - scan_start).count();
    if (rows.empty()) break;
    const auto batch_start = Clock::now();
    PlayBatch(rows, indices);
    timing_.batch_total += std::chrono::duration<double>(Clock::now() - batch_start).count();
    for (size_t r = 0; r < rows.size(); ++r) cursor_[rows[r]] = indices[r] + 1;
  }
  timing_.segment_total += std::chrono::duration<double>(Clock::now() - seg_start).count();
}

void ArenaGroup::PlayRound() { PlaySegment(1); }

void ArenaGroup::FlushResets() {
  if (pending_reset_.empty()) return;
  torch::Tensor idx = torch::tensor(pending_reset_, torch::kLong).to(device_);
  ResetSlots(raw_, idx);
  M_.index_fill_(0, idx, 0.0f);
  h_.index_fill_(0, idx, 0.0f);
  pending_reset_.clear();
}

void ArenaGroup::PlayBatch(const std::vector<int>& rows, const std::vector<int>& indices) {
  if (rows.empty()) return;
  FlushResets();
  const int64_t R = static_cast<int64_t>(rows.size());
  const int K = opts_.max_steps;
  using Clock = std::chrono::steady_clock;
  auto tick = Clock::now();
  auto lap = [&](double& acc) { const auto now = Clock::now(); acc += std::chrono::duration<double>(now - tick).count(); tick = now; };
  // profiling laps synchronise this thread's stream only
  auto sync = [&] {
#ifdef USE_CUDA
    if (device_.is_cuda()) c10::cuda::getCurrentCUDAStream(device_.index()).synchronize();
#endif
  };

  // CPU: windows, private state, changed cells, small tensors
  std::vector<LifetimeState*> states(R);
  std::vector<RawUpdate>& updates = updates_;
  updates.clear();
  std::vector<size_t> update_begin(R), update_end(R);
  std::vector<int64_t> slots(R), rounds(R);
  torch::Tensor status = torch::zeros({R, sp::kStatus}, torch::kFloat);
  torch::Tensor history = torch::zeros({R, sp::kHistoryTurns, sp::kHistoryFeatures}, torch::kFloat);
  torch::Tensor history_mask = torch::zeros({R, sp::kHistoryTurns}, torch::kBool);
  torch::Tensor entities = torch::zeros({R, sp::kMaxEntities, sp::kEntityFeatures}, torch::kFloat);
  torch::Tensor entity_mask = torch::zeros({R, sp::kMaxEntities}, torch::kBool);
  torch::Tensor xy = torch::zeros({R, sp::kTiles, 2}, torch::kLong);
  torch::Tensor head_xy = torch::zeros({R, 2}, torch::kLong);
  int64_t max_pairs = 0, max_loose = 0, max_inbox = 0;
  std::vector<int> snapshot_rows;
  for (int64_t r = 0; r < R; ++r) {
    const int a = rows[r];
    const int k = indices[r];
    Slot& s = SlotFor(a, k);
    states[r] = s.state.get();
    slots[r] = SlotIndex(a, k);
    rounds[r] = games_[a].round();
    if (record_of_[a] >= 0 && s.turns % opts_.snapshot_every == 0) snapshot_rows.push_back(static_cast<int>(r));
  }
  // snapshots of recorded lifetimes are taken before this turn's update
  for (int r : snapshot_rows) {
    const int a = rows[r];
    const int k = indices[r];
    const int64_t slot = slots[r];
    LifetimeRecord& lr = records_[record_of_[a]].lifetimes[games_[a].dragons()[k].id];
    // clone: on a CPU device .to(kCPU) would alias the live slot
    lr.snapshots.push_back(Snapshot{SlotFor(a, k).turns, raw_[slot].to(torch::kCPU).clone(),
                                    M_[slot].to(torch::kCPU).clone(), h_[slot].to(torch::kCPU).clone()});
  }
  for (int64_t r = 0; r < R; ++r) {
    const int a = rows[r];
    const int k = indices[r];
    Slot& s = SlotFor(a, k);
    LifetimeState& st = *s.state;
    const sp::Window w = sp::ReadWindow(games_[a], k, portal_ids_[a]);
    LifetimeState::Buffers buf{nullptr, nullptr, xy[r].data_ptr<int64_t>(), nullptr, nullptr,
                               status[r].data_ptr<float>(), history[r].data_ptr<float>(),
                               reinterpret_cast<uint8_t*>(history_mask[r].data_ptr<bool>()),
                               entities[r].data_ptr<float>(),
                               reinterpret_cast<uint8_t*>(entity_mask[r].data_ptr<bool>())};
    st.Prepare(w, buf);
    head_xy.data_ptr<int64_t>()[2 * r] = st.head.x;
    head_xy.data_ptr<int64_t>()[2 * r + 1] = st.head.y;
    thread_local std::vector<Cell> cells;
    ChangedCells(st, s.old_head, s.had_head, s.old_body, cells);
    const auto body_index = st.body->IndexOf();
    update_begin[r] = updates.size();
    for (const Cell& c : cells) {
      updates.emplace_back();
      RawUpdate& u = updates.back();
      u.slot = static_cast<int32_t>(slots[r]);
      u.y = c.y;
      u.x = c.x;
      RawCell(st, body_index, c.x, c.y, u.values);
    }
    update_end[r] = updates.size();
    s.old_head = st.head;
    s.had_head = true;
    s.old_body.clear();
    for (const auto& c : st.body->cells) if (c) s.old_body.push_back(*c);
    max_pairs = std::max<int64_t>(max_pairs, st.portal_pairs.size());
    max_loose = std::max<int64_t>(max_loose, st.portal_unpaired.size());
    max_inbox = std::max<int64_t>(max_inbox, st.inbox.size());
    if (record_of_[a] >= 0) RecordEvent(a, k);
  }
  lap(timing_.cpu_prepare);
  torch::Tensor pairs = torch::zeros({R, max_pairs, 2, 2}, torch::kLong);
  torch::Tensor pair_mask = torch::zeros({R, max_pairs}, torch::kBool);
  torch::Tensor axis = torch::zeros({R, max_pairs}, torch::kLong);
  torch::Tensor loose = torch::zeros({R, max_loose, 2}, torch::kLong);
  torch::Tensor loose_mask = torch::zeros({R, max_loose}, torch::kBool);
  torch::Tensor inbox = torch::zeros({R, max_inbox, 4}, torch::kLong);
  torch::Tensor inbox_mask = torch::zeros({R, max_inbox}, torch::kBool);
  {
    int64_t* pp = pairs.data_ptr<int64_t>();
    bool* pm = pair_mask.data_ptr<bool>();
    int64_t* ax = axis.data_ptr<int64_t>();
    int64_t* lo = loose.data_ptr<int64_t>();
    bool* lm = loose_mask.data_ptr<bool>();
    int64_t* ib = inbox.data_ptr<int64_t>();
    bool* im = inbox_mask.data_ptr<bool>();
    for (int64_t r = 0; r < R; ++r) {
      const LifetimeState& st = *states[r];
      for (size_t i = 0; i < st.portal_pairs.size(); ++i) {
        for (int j = 0; j < 4; ++j) pp[(r * max_pairs + i) * 4 + j] = st.portal_pairs[i][j];
        pm[r * max_pairs + i] = true;
        ax[r * max_pairs + i] = st.portal_axis[i];
      }
      for (size_t i = 0; i < st.portal_unpaired.size(); ++i) {
        lo[(r * max_loose + i) * 2] = st.portal_unpaired[i][0];
        lo[(r * max_loose + i) * 2 + 1] = st.portal_unpaired[i][1];
        lm[r * max_loose + i] = true;
      }
      for (size_t i = 0; i < st.inbox.size(); ++i) {
        for (int b = 0; b < 4; ++b) ib[(r * max_inbox + i) * 4 + b] = static_cast<int64_t>((st.inbox[i] >> (8 * b)) & 0xFF);
        im[r * max_inbox + i] = true;
      }
    }
  }
  lap(timing_.cpu_pack);

  // GPU: pre-write features, sparse update, materialise, encode
  torch::NoGradGuard no_grad;
  torch::Tensor slots_t = torch::tensor(slots, torch::kLong).to(device_);
  torch::Tensor rounds_t = torch::tensor(rounds, torch::kLong).to(device_);
  torch::Tensor xy_d = xy.to(device_);
  torch::Tensor head_d = head_xy.to(device_);
  torch::Tensor raw_rows = raw_.index_select(0, slots_t);
  torch::Tensor pre = Materialize(raw_rows, rounds_t);
  torch::Tensor visible_prev = GatherCells(pre, xy_d);                               // [R, 49, 42]
  torch::Tensor last = GatherCells(raw_rows.select(3, R_LAST_SEEN).unsqueeze(1), xy_d).squeeze(-1);
  torch::Tensor visible_age = torch::where(last >= 0, torch::clamp(rounds_t.to(torch::kFloat).unsqueeze(1) - last, 0, 500) / 500.0,
                                           torch::zeros_like(last));
  sync(); lap(timing_.materialize);
  ApplyUpdates(raw_, updates);
  sync(); lap(timing_.upd_apply);
  raw_rows = raw_.index_select(0, slots_t);
  torch::Tensor facts = Materialize(raw_rows, rounds_t);
  EncoderBatch x;
  x.local = CropWindows(facts, head_d);
  x.facts = facts;
  x.board_valid = torch::ones({R, H_, W_}, torch::TensorOptions().dtype(torch::kBool).device(device_));
  x.visible_xy = xy_d;
  x.visible_prev = visible_prev;
  x.visible_age = visible_age;
  sync(); lap(timing_.materialize);
  x.status = status.to(device_);
  x.history = history.to(device_);
  x.history_mask = history_mask.to(device_);
  x.entities = entities.to(device_);
  x.entity_mask = entity_mask.to(device_);
  x.portal_pairs = pairs.to(device_);
  x.pair_mask = pair_mask.to(device_);
  x.portal_axis = axis.to(device_);
  x.portal_unpaired = loose.to(device_);
  x.unpaired_mask = loose_mask.to(device_);
  x.inbox = inbox.to(device_);
  x.inbox_mask = inbox_mask.to(device_);
  x.head_xy = head_d;
  sync(); lap(timing_.h2d);
  auto [h_new, M_new, fused] = model_.Encode(x, M_.index_select(0, slots_t), h_.index_select(0, slots_t));
  M_.index_copy_(0, slots_t, M_new);
  h_.index_copy_(0, slots_t, h_new);
  sync(); lap(timing_.forward);
  SampledCommands cmds = model_.Sample(h_new, states);
  lap(timing_.sample);

  // critic values for the recorded events of this slot (exact state, before acting)
  std::vector<int64_t> rec_rows;
  for (int64_t r = 0; r < R; ++r) if (record_of_[rows[r]] >= 0) rec_rows.push_back(r);
  if (!rec_rows.empty()) {
    torch::Tensor habs = h_new.abs();
    torch::Tensor hstat = torch::stack({habs.mean(1), habs.std(1)}, 1).to(torch::kCPU);   // [R, 2]
    int64_t N = 1;
    std::vector<EventRecord*> evs;
    for (int64_t r : rec_rows) {
      EventRecord& ev = records_[record_of_[rows[r]]].events.back();
      // the acting dragon's own |h| summary (other dragons keep zeros for now)
      auto df = ev.dragons.accessor<float, 2>();
      for (int64_t s = 0; s < ev.dragons.size(0); ++s) {
        if (df[s][11] > 0.5f) { df[s][14] = hstat[r][0].item<float>(); df[s][15] = hstat[r][1].item<float>(); }
      }
      N = std::max<int64_t>(N, ev.dragons.size(0));
      evs.push_back(&ev);
    }
    const int64_t E = static_cast<int64_t>(evs.size());
    torch::Tensor grid = torch::zeros({E, 20, H_, W_}, torch::kFloat);
    torch::Tensor dragons = torch::zeros({E, N, 16}, torch::kFloat);
    torch::Tensor team = torch::zeros({E, N}, torch::kLong);
    torch::Tensor mask = torch::zeros({E, N}, torch::kBool);
    torch::Tensor tfeat = torch::zeros({E, 2, 4}, torch::kFloat);
    torch::Tensor phase = torch::zeros({E, 3}, torch::kFloat);
    for (int64_t e = 0; e < E; ++e) {
      grid[e] = evs[e]->map_grid.to(torch::kFloat);
      const int64_t n = evs[e]->dragons.size(0);
      dragons[e].slice(0, 0, n) = evs[e]->dragons;
      team[e].slice(0, 0, n) = evs[e]->dragon_team;
      mask[e].slice(0, 0, n) = true;
      tfeat[e] = evs[e]->team_feats;
      phase[e] = evs[e]->phase;
    }
    torch::Tensor v = model_.CriticValues(grid.to(device_), dragons.to(device_), team.to(device_), mask.to(device_),
                                          tfeat.to(device_), phase.to(device_)).to(torch::kCPU);
    const float* vp = v.data_ptr<float>();
    for (int64_t e = 0; e < E; ++e) { evs[e]->value[0] = vp[2 * e]; evs[e]->value[1] = vp[2 * e + 1]; }
  }
  lap(timing_.critic);

  // act, record, reap
  torch::Tensor lp_cpu = cmds.log_prob.to(torch::kCPU).contiguous();
  StoredCommand& sc = cmds.stored;
  torch::Tensor top_c = sc.top.to(torch::kCPU), tokens_c = sc.tokens.to(torch::kCPU), active_c = sc.active.to(torch::kCPU);
  torch::Tensor masks_c = sc.masks.to(torch::kCPU), prefix_c = sc.prefix.to(torch::kCPU);
  torch::Tensor sidx_c = sc.split_index.to(torch::kCPU), sonar_c = sc.sonar.to(torch::kCPU);
  torch::Tensor ssz_c = sc.split_sizes.to(torch::kCPU).contiguous(), sft_c = sc.split_feats.to(torch::kCPU).contiguous(), smk_c = sc.split_mask.to(torch::kCPU).contiguous();
  const int64_t* top_p = top_c.data_ptr<int64_t>();
  const int64_t* sidx_p = sidx_c.data_ptr<int64_t>();
  const int64_t* sonar_p = sonar_c.data_ptr<int64_t>();
  const float* lp_p = lp_cpu.data_ptr<float>();
  const int64_t Csz = ssz_c.size(1);
  for (int64_t r = 0; r < R; ++r) {
    const int a = rows[r];
    const int k = indices[r];
    Slot& s = SlotFor(a, k);
    LifetimeState& st = *s.state;
    Action action;
    action.sonar = cmds.sonar[r] != 0;
    action.payload = action.sonar ? cmds.sonar[r] - 1 : 0;
    if (cmds.is_split[r]) {
      st.CommitSplit(cmds.split_n[r]);
      action.kind = Action::Kind::Split;
      action.split_size = cmds.split_n[r];
    } else {
      st.CommitMove(cmds.dirs[r]);
      action.kind = Action::Kind::Move;
      for (char c : cmds.dirs[r]) action.dirs.push_back(static_cast<Dir>(std::string("NESW").find(c)));
    }
    if (record_of_[a] >= 0) {
      ArenaRecord& rec = records_[record_of_[a]];
      LifetimeRecord& lr = rec.lifetimes[games_[a].dragons()[k].id];
      TurnRecord t;
      t.event = static_cast<int>(rec.events.size()) - 1;
      t.round = games_[a].round();
      t.turn_index = s.turns;
      t.updates.assign(updates.begin() + update_begin[r], updates.begin() + update_end[r]);
      t.status.assign(status[r].data_ptr<float>(), status[r].data_ptr<float>() + sp::kStatus);
      t.history.assign(history[r].data_ptr<float>(), history[r].data_ptr<float>() + sp::kHistoryTurns * sp::kHistoryFeatures);
      t.history_mask.assign(reinterpret_cast<uint8_t*>(history_mask[r].data_ptr<bool>()),
                            reinterpret_cast<uint8_t*>(history_mask[r].data_ptr<bool>()) + sp::kHistoryTurns);
      t.entities.assign(entities[r].data_ptr<float>(), entities[r].data_ptr<float>() + sp::kMaxEntities * sp::kEntityFeatures);
      t.entity_mask.assign(reinterpret_cast<uint8_t*>(entity_mask[r].data_ptr<bool>()),
                           reinterpret_cast<uint8_t*>(entity_mask[r].data_ptr<bool>()) + sp::kMaxEntities);
      t.visible_xy.assign(xy[r].data_ptr<int64_t>(), xy[r].data_ptr<int64_t>() + 2 * sp::kTiles);
      t.head_x = st.head.x;
      t.head_y = st.head.y;
      t.portal_pairs = st.portal_pairs;
      t.portal_axis = st.portal_axis;
      t.portal_unpaired = st.portal_unpaired;
      t.inbox = st.inbox;
      t.top = top_p[r];
      t.split_index = sidx_p[r];
      t.sonar = sonar_p[r];
      t.tokens.assign(tokens_c[r].data_ptr<int64_t>(), tokens_c[r].data_ptr<int64_t>() + K);
      t.active.assign(reinterpret_cast<uint8_t*>(active_c[r].data_ptr<bool>()), reinterpret_cast<uint8_t*>(active_c[r].data_ptr<bool>()) + K);
      t.masks.assign(reinterpret_cast<uint8_t*>(masks_c[r].data_ptr<bool>()), reinterpret_cast<uint8_t*>(masks_c[r].data_ptr<bool>()) + K * sp::kTokens);
      t.prefix.assign(prefix_c[r].data_ptr<float>(), prefix_c[r].data_ptr<float>() + K * sp::kPrefixFeatures);
      for (int64_t c = 0; c < Csz; ++c) {
        if (!smk_c.data_ptr<bool>()[r * Csz + c]) break;
        t.split_sizes.push_back(ssz_c.data_ptr<int64_t>()[r * Csz + c]);
        for (int j = 0; j < 4; ++j) t.split_feats.push_back(sft_c.data_ptr<float>()[(r * Csz + c) * 4 + j]);
      }
      t.split_legal = st.split_legal;
      t.log_prob = lp_p[r];
      lr.turns.push_back(std::move(t));
    }
    games_[a].TakeTurn(k, action);
    ++s.turns;
    Reap(a);
    SpawnNew(a);
  }
  lap(timing_.act_record);
  decisions_ += R;
}

std::vector<ArenaRecord> ArenaGroup::TakeRecords() {
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<ArenaRecord> out;
  for (ArenaRecord& rec : records_) {
    for (size_t e = 0; e + 1 < rec.events.size(); ++e) rec.events[e].delta_rounds = rec.events[e + 1].round - rec.events[e].round;
    if (!rec.events.empty()) rec.events.back().delta_rounds = 1;
    out.push_back(std::move(rec));
    rec = ArenaRecord{{}, {}, out.back().arena};
    // living lifetimes continue; their next turn starts a fresh record and the
    // snapshot rule (turn_index % snapshot_every == 0) will not fire for them,
    // so force one by aligning the counter
    for (Slot& s : slots_[out.back().arena]) {
      if (s.state) s.turns = 0;
    }

    for (const auto& [id, lr] : out.back().lifetimes) {
      if (!lr.dead) rec.lifetimes[id] = LifetimeRecord{lr.dragon_id, lr.team, lr.learner, {}, {}, false};
    }
  }
  timing_.take += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return out;
}

std::vector<std::array<float, 2>> ArenaGroup::BootstrapValues() {
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::array<float, 2>> out(records_.size(), {0.0f, 0.0f});
  // the value of the state after the last recorded action of every recorded
  // arena, as one critic call; built as events without recording them
  std::vector<EventRecord> evs;
  for (ArenaRecord& rec : records_) {
    const int a = rec.arena;
    const Game& game = games_[a];
    int index = 0;
    for (int i = 0; i < static_cast<int>(game.dragons().size()); ++i) if (game.dragons()[i].alive) { index = i; break; }
    const size_t before = rec.events.size();
    RecordEvent(a, index);
    evs.push_back(std::move(rec.events.back()));
    rec.events.resize(before);
  }
  if (!evs.empty()) {
    torch::NoGradGuard no_grad;
    const int64_t E = static_cast<int64_t>(evs.size());
    int64_t N = 1;
    for (const EventRecord& ev : evs) N = std::max<int64_t>(N, ev.dragons.size(0));
    torch::Tensor grid = torch::zeros({E, 20, H_, W_}, torch::kFloat), dragons = torch::zeros({E, N, 16}, torch::kFloat);
    torch::Tensor team = torch::zeros({E, N}, torch::kLong), mask = torch::zeros({E, N}, torch::kBool);
    torch::Tensor tfeat = torch::zeros({E, 2, 4}, torch::kFloat), phase = torch::zeros({E, 3}, torch::kFloat);
    for (int64_t e = 0; e < E; ++e) {
      grid[e] = evs[e].map_grid.to(torch::kFloat);
      const int64_t n = evs[e].dragons.size(0);
      dragons[e].slice(0, 0, n) = evs[e].dragons;
      team[e].slice(0, 0, n) = evs[e].dragon_team;
      mask[e].slice(0, 0, n) = true;
      tfeat[e] = evs[e].team_feats;
      phase[e] = evs[e].phase;
    }
    torch::Tensor v = model_.CriticValues(grid.to(device_), dragons.to(device_), team.to(device_), mask.to(device_),
                                          tfeat.to(device_), phase.to(device_)).to(torch::kCPU).contiguous();
    const float* vp = v.data_ptr<float>();
    for (int64_t e = 0; e < E; ++e) out[e] = {vp[2 * e], vp[2 * e + 1]};
  }
  timing_.bootstrap += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return out;
}

std::vector<std::string> ArenaGroup::Results() const {
  std::vector<std::string> out;
  for (const Game& g : games_) {
    switch (g.Result()) {
      case Outcome::TeamA: out.push_back("A"); break;
      case Outcome::TeamB: out.push_back("B"); break;
      case Outcome::Draw: out.push_back("draw"); break;
      default: out.push_back("running"); break;
    }
  }
  return out;
}

}  // namespace dragons::train
