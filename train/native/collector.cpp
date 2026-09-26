#include "collector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <unordered_map>
#include <stdexcept>
#include <string>

namespace rl {
namespace {

// Rows [lo, hi) of `n` for worker w of T.
template <typename F>
void Parallel(dragons::Pool& pool, int64_t n, F&& fn) {
  const int T = pool.size();
  pool.Run([&](int w) {
    const int64_t lo = n * w / T, hi = n * (w + 1) / T;
    for (int64_t i = lo; i < hi; ++i) fn(i);
  });
}

void Play(dragons::Game& game, const Command& c) {
  dragons::Sonar sonar;
  sonar.mask = c.sonar_mask;
  std::copy(c.words.begin(), c.words.end(), sonar.word);
  if (c.kind == 0) game.Act(dragons::kMove, c.dirs.data(), c.nsteps, 0, sonar, kProtocol);
  else game.Act(dragons::kSplit, nullptr, 0, c.split, sonar, kProtocol);
}

uint64_t Mix(uint64_t z) {
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

// The scripted opponents read only the legal observation.
Command ScriptedCommand(int32_t kind, const Observation& o, uint64_t seed) {
  constexpr int C = kWindow / 2;
  const int dx[4] = {0, 1, 0, -1}, dy[4] = {-1, 0, 1, 0};
  std::array<bool, kWindowTiles> body{};
  for (const Segment& s : o.segments)
    for (int k = 0; k < kWindowTiles; ++k)
      if (o.tile_x[k] == s.x && o.tile_y[k] == s.y) body[k] = true;
  const int32_t edge[4] = {o.hedge[C * kWindow + C], o.vedge[C * (kWindow + 1) + C + 1], o.hedge[(C + 1) * kWindow + C],
                           o.vedge[C * (kWindow + 1) + C]};
  uint64_t r = Mix(seed ^ Mix(static_cast<uint64_t>(o.spawn.id) << 32 | static_cast<uint32_t>(o.round)));
  int best = o.facing, best_score = 1 << 30;
  for (int d = 0; d < 4; ++d) {
    const int k = (C + dy[d]) * kWindow + C + dx[d];
    if (edge[d] == dragons::kEdgeKelp || body[k]) continue;
    int score = 0;
    if (kind == kGreedy) {
      int nearest = 99;
      for (int j = 0; j < kWindowTiles; ++j)
        if (o.pearl[j]) nearest = std::min(nearest, std::abs(j % kWindow - (C + dx[d])) + std::abs(j / kWindow - (C + dy[d])));
      score = nearest * 64;
    }
    r = Mix(r);
    score += static_cast<int>(r % 64);                 // random tie-break
    if (score < best_score) { best_score = score; best = d; }
  }
  Command c;
  c.kind = 0;
  c.nsteps = 1;
  c.dirs[0] = static_cast<uint8_t>(best);
  return c;
}

int32_t EchoBytes(const Observation& o) {
  size_t m = 0;
  for (const auto& f : o.echo) m = std::max(m, f.size());
  return static_cast<int32_t>(m);
}

}  // namespace

void EncodeObservation(const Lifetime& life, const Observation& o, const ActorBatch& out, int64_t b, bool reference) {
  // buffers are reused across steps: clear this row's parts an encoding does not always write
  std::fill(out.entity + b * kMaxEntities * kEntityFields, out.entity + (b + 1) * kMaxEntities * kEntityFields, 0);
  std::fill(out.segment + b * kWindowTiles * kSegmentFields, out.segment + (b + 1) * kWindowTiles * kSegmentFields, 0);
  std::fill(out.status + b * kStatusFields, out.status + (b + 1) * kStatusFields, 0);
  std::fill(out.inbox + b * out.k_pad * 8, out.inbox + (b + 1) * out.k_pad * 8, 0);
  std::fill(out.echo + b * 5 * out.e_pad, out.echo + (b + 1) * 5 * out.e_pad, 0);
  WindowRows win{out.win + b * kWinTiles * kTileFields, out.wdest + b * kWinTiles * 4,
                 out.grid + b * kGridCells * kGridFields};
  if (reference) life.Rebuilt(o).EncodeWindow(win);
  else life.EncodeWindow(win);
  DecisionRows rows{nullptr, nullptr,
                    out.entity + b * kMaxEntities * kEntityFields,
                    out.segment + b * kWindowTiles * kSegmentFields,
                    out.status + b * kStatusFields,
                    out.prev_dirs + b * kPrevDirsFields,
                    out.prev_words + b * 32,
                    0};
  life.EncodeSmall(o, rows);
  // the latest kMaxInbox words and the first kMaxEcho bytes of each echo field (the same rule everywhere)
  const size_t words = std::min(o.inbox.size(), static_cast<size_t>(kMaxInbox));
  if (static_cast<int32_t>(words) > out.k_pad) throw std::invalid_argument("k_pad below the inbox size");
  uint8_t* inbox = out.inbox + b * out.k_pad * 8;
  const size_t skip = o.inbox.size() - words;
  for (size_t k = 0; k < words; ++k)
    for (int i = 0; i < 8; ++i) inbox[k * 8 + i] = static_cast<uint8_t>(o.inbox[skip + k] >> (8 * i));
  uint8_t* echo = out.echo + b * 5 * out.e_pad;
  for (int f = 0; f < 5; ++f) {
    const std::string& s = o.echo[f];
    const size_t n = std::min(s.size(), static_cast<size_t>(kMaxEcho));
    if (static_cast<int32_t>(n) > out.e_pad) throw std::invalid_argument("e_pad below an echo field");
    std::memcpy(echo + f * out.e_pad, s.data(), n);
    out.echo_len[b * 5 + f] = static_cast<int32_t>(n);
  }
}

// ---- collection -------------------------------------------------------------------

void Collector::Drop(Live& L) {
  L.life.reset();
}

void Collector::DropAll(int32_t m) {
  for (Live& L : live_[m]) Drop(L);
  live_[m].clear();
}

Collector::Collector(std::vector<MatchSetup> setups, int threads, bool keep_observations)
    : keep_observations_(keep_observations), pool_(std::max(1, threads)) {
  const size_t M = setups.size();
  games_.resize(M);
  live_.resize(M);
  records_.resize(M);
  prepared_.assign(M, 0);
  harvested_.assign(M, 0);
  for (size_t m = 0; m < M; ++m) Start(static_cast<int32_t>(m), setups[m]);
}

void Collector::Start(int32_t m, const MatchSetup& setup) {
  auto board = dragons::LoadBoard(setup.map_text);
  if (board->width < kMinSide || board->width > kMaxSide || board->height < kMinSide || board->height > kMaxSide)
    throw std::invalid_argument("boards are 7..64 a side");
  games_[m].Reset(board);
  records_[m] = MatchRecord{};
  records_[m].board = board;
  records_[m].setup = setup;
  DropAll(m);
  prepared_[m] = 0;
  harvested_[m] = 0;
  if (games_[m].done()) {
    records_[m].finished = true;
    records_[m].winner = games_[m].result().winner;
  }
}

void Collector::Reset(int32_t m, const MatchSetup& setup) {
  if (!records_[m].finished) throw std::logic_error("reset of a match still being played");
  if (!harvested_[m]) pending_.push_back(std::move(records_[m]));
  Start(m, setup);
}

std::shared_ptr<Pack> Collector::Harvest() {
  auto pack = std::make_shared<Pack>();
  pack->matches = std::move(pending_);
  pending_.clear();
  for (int32_t m = 0; m < num_matches(); ++m)
    if (records_[m].finished && !harvested_[m]) {
      pack->matches.push_back(records_[m]);        // a copy: results() still reads the slot
      harvested_[m] = 1;
    }
  return pack;
}

bool Collector::done() const {
  for (const MatchRecord& r : records_)
    if (!r.finished) return false;
  return true;
}

// Scripted dragons act until a policy's dragon is up. Their turns are not
// events (a match with a scripted team is for evaluation, not training).
void Collector::PlayScripted(int32_t m) {
  dragons::Game& g = games_[m];
  const MatchSetup& setup = records_[m].setup;
  while (!g.done()) {
    const int32_t team = g.dragons()[g.actor()].team;
    const int32_t kind = setup.policy[team];
    if (kind >= 0) return;
    if (setup.record[team]) throw std::invalid_argument("a scripted team cannot be recorded");
    Play(g, ScriptedCommand(kind, Observe(g), setup.seed));
  }
  records_[m].finished = true;
  records_[m].winner = g.result().winner;
  records_[m].rounds = g.round();
  DropAll(m);
}

std::vector<Pending> Collector::Prepare() {
  const int32_t M = num_matches();
  std::vector<int32_t> active;
  Parallel(pool_, M, [&](int64_t m) {
    if (!records_[m].finished && !prepared_[m]) PlayScripted(static_cast<int32_t>(m));
  });
  // serial: new lifetimes and their records
  for (int32_t m = 0; m < M; ++m) {
    MatchRecord& mr = records_[m];
    if (mr.finished) continue;
    active.push_back(m);
    const int32_t d = games_[m].actor();
    auto& live = live_[m];
    if (static_cast<int32_t>(live.size()) <= d) live.resize(d + 1);
    if (static_cast<int32_t>(mr.lifetimes.size()) <= d) mr.lifetimes.resize(d + 1, -1);
    if (!live[d].life) {
      if (mr.lifetimes[d] >= 0) throw std::logic_error("a dragon's lifetime started twice");
      live[d].life = std::make_unique<Lifetime>(SpawnOf(games_[m], d));
      LifetimeRecord rec;
      rec.match = m;
      rec.dragon = d;
      rec.team = games_[m].dragons()[d].team;
      rec.recorded = mr.setup.record[rec.team];
      mr.lifetimes[d] = static_cast<int32_t>(mr.lives.size());
      mr.lives.push_back(std::move(rec));
    }
    // snapshots serve this process's own Replay; serialized packs rebuild theirs
    if (keep_observations_ && mr.events.size() % kSnapshotEvery == 0 && !prepared_[m]) mr.snapshots.push_back(games_[m]);
  }
  std::vector<Pending> out(active.size());
  Parallel(pool_, static_cast<int64_t>(active.size()), [&](int64_t i) {
    const int32_t m = active[i];
    const dragons::Game& g = games_[m];
    const int32_t d = g.actor();
    Live& L = live_[m][d];
    if (!prepared_[m]) {
      L.obs = Observe(g);
      L.life->Observe(L.obs);
      prepared_[m] = 1;
    }
    Pending& p = out[i];
    p.match = m;
    p.dragon = d;
    p.team = g.dragons()[d].team;
    p.policy = records_[m].setup.policy[p.team];
    p.record = records_[m].setup.record[p.team];
    p.tiles = g.board().width * g.board().height;
    p.length = L.obs.length;
    p.inbox = static_cast<int32_t>(L.obs.inbox.size());
    p.echo_bytes = EchoBytes(L.obs);
    p.alive = g.alive_count(0) + g.alive_count(1);
  });
  return out;
}

void Collector::EncodeActors(const std::vector<int32_t>& matches, const ActorBatch& out) {
  Parallel(pool_, static_cast<int64_t>(matches.size()), [&](int64_t b) {
    const int32_t m = matches[b];
    if (!prepared_[m]) throw std::logic_error("encode before prepare");
    const Live& L = live_[m][games_[m].actor()];
    EncodeObservation(*L.life, L.obs, out, b);
    std::copy(L.h.begin(), L.h.end(), out.h + b * kHidden);
  });
}

void Collector::EncodeCritics(const std::vector<int32_t>& matches, const CriticBatch& out) {
  Parallel(pool_, static_cast<int64_t>(matches.size()), [&](int64_t b) {
    const int32_t m = matches[b];
    const dragons::Game& g = games_[m];

    int32_t* rows = out.dragons + b * out.d_pad * kCriticDragonFields;
    const int32_t alive = EncodeEvent(g, out.cells + b * kCriticCells * kCriticCellFields, out.d_pad, rows,
                                      out.global + b * kCriticGlobalFields);
    if (alive > out.d_pad) throw std::invalid_argument("d_pad below the living dragons");
    float* sum = out.summary + b * out.d_pad * (kHidden + 1);
    std::fill(sum, sum + out.d_pad * (kHidden + 1), 0.0f);
    for (int32_t i = 0; i < alive; ++i) {
      const int32_t id = rows[i * kCriticDragonFields + kDId];
      if (id >= static_cast<int32_t>(live_[m].size()) || !live_[m][id].life) continue;   // not acted yet: zeros
      const Live& L = live_[m][id];
      std::copy(L.h_chunk.begin(), L.h_chunk.end(), sum + i * (kHidden + 1));
      sum[i * (kHidden + 1) + kHidden] = L.seen;
    }
  });
}

void Collector::Apply(const std::vector<int32_t>& matches, const std::vector<Command>& commands, const float* logp,
                      const float* h_new, const float* value) {
  if (commands.size() != matches.size()) throw std::invalid_argument("one command per match");
  Parallel(pool_, static_cast<int64_t>(matches.size()), [&](int64_t b) {
    const int32_t m = matches[b];
    if (!prepared_[m]) throw std::logic_error("apply before prepare");
    dragons::Game& g = games_[m];
    MatchRecord& mr = records_[m];
    const int32_t d = g.actor();
    Live& L = live_[m][d];
    LifetimeRecord& rec = mr.lives[mr.lifetimes[d]];
    const Command& c = commands[b];
    const int32_t event = static_cast<int32_t>(mr.events.size());
    std::copy(h_new + b * kHidden, h_new + (b + 1) * kHidden, L.h.begin());
    L.seen = static_cast<float>(L.life->seen_fraction());
    if (rec.recorded) {
      if (keep_observations_) {
        rec.observations.push_back(std::move(L.obs));
        rec.commands.push_back(c);
      }
      rec.logp.push_back(logp[b]);
    }
    rec.events.push_back(event);
    if (rec.events.size() % kChunk == 0) {        // a chunk ends: its successor starts from this h
      rec.h.push_back(L.h);
      L.h_chunk = L.h;
    }
    rec.seen.push_back(L.seen);
    L.life->Commit(c);
    EventRecord ev;
    ev.dragon = d;
    ev.command = c;
    ev.value[rec.team] = value[b];
    ev.value[1 - rec.team] = -value[b];
    mr.events.push_back(ev);
    g.ClearDeaths();
    Play(g, c);
    for (const dragons::Death& death : g.deaths()) {   // closed: facts dropped
      ++mr.deaths[death.team * 5 + death.reason];
      if (death.id < static_cast<int32_t>(mr.lifetimes.size()) && mr.lifetimes[death.id] >= 0)
        mr.lives[mr.lifetimes[death.id]].death = static_cast<int8_t>(death.reason);
      if (death.id < static_cast<int32_t>(live_[m].size())) Drop(live_[m][death.id]);
    }
    g.ClearDeaths();
    prepared_[m] = 0;
    if (g.done()) {
      mr.finished = true;
      mr.winner = g.result().winner;
      mr.rounds = g.round();
      DropAll(m);
    }
  });
}

// ---- the native step loop -----------------------------------------------------------

void Collector::PushMaps(std::vector<std::string> maps) {
  for (auto& m : maps) {
    if (recent_maps_.size() < 64) recent_maps_.push_back(m);
    else recent_maps_[rng_() % recent_maps_.size()] = m;
    maps_.push_back(std::move(m));
  }
}

void Collector::SetMatchmaking(double self_play, uint64_t seed) {
  self_play_ = self_play;
  rng_.seed(seed);
}

MatchSetup Collector::NextSetup() {
  MatchSetup s;
  if (!maps_.empty()) {
    s.map_text = std::move(maps_.front());
    maps_.pop_front();
  } else if (!recent_maps_.empty()) {
    s.map_text = recent_maps_[rng_() % recent_maps_.size()];
  } else {
    throw std::logic_error("no maps to start a match on");
  }
  const int side = static_cast<int>(rng_() % 2);
  const bool self = std::uniform_real_distribution<double>(0.0, 1.0)(rng_) < self_play_;
  s.policy[side] = 0;
  s.policy[1 - side] = self ? 0 : 1;
  s.record[side] = true;
  s.record[1 - side] = self;
  return s;
}

void Collector::StepEncode(const ActorBatch& learner, int32_t l_cap, const ActorBatch& opponent, int32_t o_cap,
                           const CriticBatch& critic, int32_t critic_cap, int32_t critic_every, int32_t* crow,
                           int32_t* cteam, int32_t* counts) {
  const int32_t M = num_matches();
  critic_every_ = critic_every;
  if (static_cast<int32_t>(v_age_.size()) != M) {
    v_value_.assign(M, 0.0f);
    v_team_.assign(M, 0);
    v_age_.assign(M, critic_every);
  }
  for (int32_t m = 0; m < M; ++m)
    if (records_[m].finished) {
      Reset(m, NextSetup());
      v_age_[m] = critic_every;
    }
  const std::vector<Pending> pend = Prepare();
  // rows: the learner's first-come up to l_cap, the opponent's up to o_cap (the rest wait, still prepared)
  step_active_.clear();
  step_row_.clear();
  std::vector<int32_t> slot_of;                   // per selected decision: its row within its buffer
  std::vector<uint8_t> is_opp;
  int32_t nl = 0, no = 0;
  for (const Pending& p : pend) {
    const bool o = p.policy != 0;
    if (o ? no >= o_cap : nl >= l_cap) continue;
    const int32_t r = o ? no++ : nl++;
    step_active_.push_back(p.match);
    step_row_.push_back(o ? l_cap + r : r);
    slot_of.push_back(r);
    is_opp.push_back(o);
  }
  Parallel(pool_, static_cast<int64_t>(step_active_.size()), [&](int64_t i) {
    const int32_t m = step_active_[i];
    const Live& L = live_[m][games_[m].actor()];
    const ActorBatch& out = is_opp[i] ? opponent : learner;
    EncodeObservation(*L.life, L.obs, out, slot_of[i]);
    std::copy(L.h.begin(), L.h.end(), out.h + static_cast<int64_t>(slot_of[i]) * kHidden);
  });
  // the critic: the stalest learner decisions first
  std::vector<std::pair<int32_t, int32_t>> due;       // (-age, index in step_active_)
  for (size_t i = 0; i < step_active_.size(); ++i) {
    const int32_t m = step_active_[i];
    const dragons::Game& g = games_[m];
    if (records_[m].setup.record[g.dragons()[g.actor()].team] && v_age_[m] >= critic_every)
      due.emplace_back(-v_age_[m], static_cast<int32_t>(i));
  }
  std::sort(due.begin(), due.end());
  if (static_cast<int32_t>(due.size()) > critic_cap) due.resize(critic_cap);
  step_critic_.clear();
  for (size_t k = 0; k < due.size(); ++k) {
    const int32_t i = due[k].second, m = step_active_[i];
    step_critic_.push_back(m);
    crow[k] = step_row_[i];
    cteam[k] = games_[m].dragons()[games_[m].actor()].team;
  }
  if (!step_critic_.empty()) EncodeCritics(step_critic_, critic);
  counts[0] = nl;
  counts[1] = no;
  counts[2] = static_cast<int32_t>(step_critic_.size());
}

void Collector::StepApply(const int32_t* kind, const int32_t* nsteps, const int32_t* dirs, const int32_t* split,
                          const int32_t* sonar, const float* logp, float* h_new, const float* value) {
  const size_t B = step_active_.size();
  std::vector<Command> cmds(B);
  std::vector<float> lp(B), v(B), h(B * kHidden);
  std::vector<int32_t> fresh(num_matches(), -1);
  for (size_t i = 0; i < step_critic_.size(); ++i) fresh[step_critic_[i]] = static_cast<int32_t>(i);
  for (size_t b = 0; b < B; ++b) {
    const int32_t m = step_active_[b], r = step_row_[b];
    Command& c = cmds[b];
    c.kind = static_cast<uint8_t>(kind[r]);
    c.nsteps = static_cast<uint8_t>(c.kind == 0 ? nsteps[r] : 0);
    for (int k = 0; k < kMaxMoveTokens; ++k) c.dirs[k] = static_cast<uint8_t>(std::max(0, dirs[r * kMaxMoveTokens + k]));
    c.split = split[r];
    c.sonar_mask = 0;
    for (int d = 0; d < 4; ++d) {
      const int32_t s = sonar[r * 4 + d];
      if (s > 0) {
        c.sonar_mask |= 1 << d;
        c.words[d] = static_cast<uint64_t>(s - 1);
      }
    }
    float* hm = h_new + static_cast<int64_t>(r) * kHidden;
    bool ok = std::isfinite(logp[r]);
    for (int k = 0; k < kHidden && ok; ++k) ok = std::isfinite(hm[k]);
    if (!ok) {                                        // a nonfinite h would poison the dragon for life
      std::fill(hm, hm + kHidden, 0.0f);
      ++nonfinite_;
    }
    lp[b] = ok ? logp[r] : std::numeric_limits<float>::quiet_NaN();
    std::copy(hm, hm + kHidden, h.begin() + static_cast<int64_t>(b) * kHidden);
    // V_old: fresh where the critic ran, the match's latest otherwise (a baseline: unbiased)
    const dragons::Game& g = games_[m];
    const int32_t team = g.dragons()[g.actor()].team;
    const bool rec = records_[m].setup.record[team];
    if (fresh[m] >= 0) {
      v_value_[m] = value[fresh[m]];
      v_team_[m] = team;
      v_age_[m] = 0;
    }
    v[b] = !rec ? std::numeric_limits<float>::quiet_NaN()
                : (v_team_[m] == team ? v_value_[m] : -v_value_[m]);
    ++v_age_[m];
  }
  turns_ += static_cast<int64_t>(B);
  Apply(step_active_, cmds, lp.data(), h.data(), v.data());
}

// ---- packs between processes -----------------------------------------------------------

namespace {

struct Writer {
  std::string out;
  template <typename T> void put(const T& v) { out.append(reinterpret_cast<const char*>(&v), sizeof(T)); }
  template <typename T> void vec(const std::vector<T>& v) {
    put(static_cast<uint64_t>(v.size()));
    out.append(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
  }
  void str(const std::string& s) {
    put(static_cast<uint64_t>(s.size()));
    out.append(s);
  }
};

struct Reader {
  const std::string& in;
  size_t at = 0;
  void need(size_t n) const {
    if (at + n > in.size()) throw std::invalid_argument("truncated pack bytes");
  }
  template <typename T> T get() {
    need(sizeof(T));
    T v;
    std::memcpy(&v, in.data() + at, sizeof(T));
    at += sizeof(T);
    return v;
  }
  template <typename T> std::vector<T> vec() {
    const uint64_t n = get<uint64_t>();
    need(n * sizeof(T));
    std::vector<T> v(n);
    std::memcpy(v.data(), in.data() + at, n * sizeof(T));
    at += n * sizeof(T);
    return v;
  }
  std::string str() {
    const uint64_t n = get<uint64_t>();
    need(n);
    std::string s = in.substr(at, n);
    at += n;
    return s;
  }
};

constexpr uint32_t kPackMagic = 0x4b435044;   // "DPCK"

}  // namespace

std::string SerializePack(const Pack& pack) {
  Writer w;
  w.put(kPackMagic);
  w.put(static_cast<uint64_t>(pack.matches.size()));
  for (const MatchRecord& mr : pack.matches) {
    if (!mr.finished) throw std::logic_error("only finished matches travel");
    w.str(mr.setup.map_text);
    w.put(mr.setup.policy[0]); w.put(mr.setup.policy[1]);
    w.put(static_cast<uint8_t>(mr.setup.record[0])); w.put(static_cast<uint8_t>(mr.setup.record[1]));
    w.put(mr.setup.seed);
    w.vec(mr.events);                                   // EventRecord is trivially copyable
    w.put(static_cast<uint64_t>(mr.lives.size()));
    for (const LifetimeRecord& r : mr.lives) {
      w.put(r.match); w.put(r.dragon); w.put(r.team);
      w.put(static_cast<uint8_t>(r.recorded)); w.put(r.death);
      w.vec(r.logp); w.vec(r.events); w.vec(r.h); w.vec(r.seen);
    }
    w.put(mr.winner); w.put(mr.rounds); w.put(mr.deaths);
  }
  return std::move(w.out);
}

std::shared_ptr<Pack> DeserializePack(const std::string& bytes) {
  Reader r{bytes};
  if (r.get<uint32_t>() != kPackMagic) throw std::invalid_argument("not pack bytes");
  auto pack = std::make_shared<Pack>();
  const uint64_t M = r.get<uint64_t>();
  pack->matches.resize(M);
  for (uint64_t i = 0; i < M; ++i) {
    MatchRecord& mr = pack->matches[i];
    mr.setup.map_text = r.str();
    mr.setup.policy[0] = r.get<int32_t>(); mr.setup.policy[1] = r.get<int32_t>();
    mr.setup.record[0] = r.get<uint8_t>(); mr.setup.record[1] = r.get<uint8_t>();
    mr.setup.seed = r.get<uint64_t>();
    mr.events = r.vec<EventRecord>();
    mr.lives.resize(r.get<uint64_t>());
    for (LifetimeRecord& rec : mr.lives) {
      rec.match = r.get<int32_t>(); rec.dragon = r.get<int32_t>(); rec.team = r.get<int32_t>();
      rec.recorded = r.get<uint8_t>(); rec.death = r.get<int8_t>();
      rec.logp = r.vec<float>(); rec.events = r.vec<int32_t>();
      rec.h = r.vec<std::array<float, kHidden>>(); rec.seen = r.vec<float>();
    }
    mr.winner = r.get<int8_t>(); mr.rounds = r.get<int32_t>(); mr.deaths = r.get<std::array<int32_t, 2 * 5>>();
    mr.finished = true;
    for (size_t k = 0; k < mr.lives.size(); ++k) {
      const int32_t d = mr.lives[k].dragon;
      if (static_cast<int32_t>(mr.lifetimes.size()) <= d) mr.lifetimes.resize(d + 1, -1);
      mr.lifetimes[d] = static_cast<int32_t>(k);
    }
  }
  RebuildPack(*pack);
  return pack;
}

void RebuildMatch(MatchRecord& mr) {
  if (!mr.snapshots.empty()) return;              // kept at collection: nothing to rebuild
  mr.board = dragons::LoadBoard(mr.setup.map_text);
  dragons::Game g;
  g.Reset(mr.board);
  for (size_t e = 0; e < mr.events.size(); ++e) {
    if (e % kSnapshotEvery == 0) mr.snapshots.push_back(g);
    const EventRecord& ev = mr.events[e];
    if (g.actor() != ev.dragon) throw std::logic_error("pack replay diverged from the recorded match");
    LifetimeRecord& rec = mr.lives[mr.lifetimes[ev.dragon]];
    if (rec.recorded) {
      rec.observations.push_back(Observe(g));
      rec.commands.push_back(ev.command);
    }
    g.ClearDeaths();
    Play(g, ev.command);
    g.ClearDeaths();
  }
}

void RebuildPack(Pack& pack) {
  for (MatchRecord& mr : pack.matches) RebuildMatch(mr);
}

// ---- replay ----------------------------------------------------------------------

Replay::Replay(std::vector<std::shared_ptr<const Pack>> packs, int threads,
               const std::optional<std::vector<int32_t>>& keep)
    : packs_(std::move(packs)), pool_(std::max(1, threads)) {
  std::vector<const LifetimeRecord*> recorded;
  std::vector<int32_t> recorded_match;
  for (const auto& pack : packs_)
    for (const MatchRecord& mr : pack->matches) {
      const int32_t mi = static_cast<int32_t>(matches_.size());
      matches_.push_back(&mr);
      for (const LifetimeRecord& r : mr.lives)
        if (r.recorded && !r.observations.empty()) {
          recorded.push_back(&r);
          recorded_match.push_back(mi);
        }
    }
  if (!keep) {
    lives_ = recorded;
    life_match_ = recorded_match;
  } else {
    for (int32_t i : *keep) {
      lives_.push_back(recorded.at(i));
      life_match_.push_back(recorded_match.at(i));
    }
  }
  facts_.resize(lives_.size());
  next_.assign(lives_.size(), 0);
}

void Replay::Reset() {
  for (auto& f : facts_) f.reset();
  std::fill(next_.begin(), next_.end(), 0);
}

void Replay::EncodeChunk(const std::vector<int32_t>& which, const std::vector<int32_t>& start, int32_t steps,
                         const ActorBatch& out, bool reference) {
  if (which.size() != start.size()) throw std::invalid_argument("which and start differ in length");
  Parallel(pool_, static_cast<int64_t>(which.size()), [&](int64_t b) {
    const int32_t i = which[b];
    const LifetimeRecord& rec = *lives_[i];
    const int32_t n = static_cast<int32_t>(rec.observations.size());
    if (start[b] != next_[i]) throw std::logic_error("lifetime chunks must be replayed in order");
    if (start[b] == 0) facts_[i] = std::make_unique<Lifetime>(rec.observations[0].spawn);
    const int32_t count = std::min(steps, n - start[b]);
    for (int32_t t = 0; t < count; ++t) {
      const Observation& o = rec.observations[start[b] + t];
      facts_[i]->Observe(o);
      EncodeObservation(*facts_[i], o, out, b * steps + t, reference);
      facts_[i]->Commit(rec.commands[start[b] + t]);
    }
    next_[i] = start[b] + count;
    if (next_[i] == n) facts_[i].reset();
  });
}

void Replay::EncodeList(const std::vector<int32_t>& which, const std::vector<int32_t>& step, const ActorBatch& out) {
  if (which.size() != step.size()) throw std::invalid_argument("which and step differ in length");
  // entries per lifetime, in list order; each lifetime is advanced by one worker
  std::vector<int32_t> lives;
  std::unordered_map<int32_t, std::vector<int32_t>> entries;
  for (size_t e = 0; e < which.size(); ++e) {
    if (which[e] < 0) continue;
    auto& v = entries[which[e]];
    if (v.empty()) lives.push_back(which[e]);
    v.push_back(static_cast<int32_t>(e));
  }
  Parallel(pool_, static_cast<int64_t>(lives.size()), [&](int64_t k) {
    const int32_t i = lives[k];
    const LifetimeRecord& rec = *lives_[i];
    for (const int32_t e : entries.at(i)) {
      const int32_t t = step[e];
      if (t != next_[i]) throw std::logic_error("lifetime decisions must be encoded in order");
      if (t == 0) facts_[i] = std::make_unique<Lifetime>(rec.observations[0].spawn);
      const Observation& o = rec.observations[t];
      facts_[i]->Observe(o);
      EncodeObservation(*facts_[i], o, out, e);
      facts_[i]->Commit(rec.commands[t]);
      next_[i] = t + 1;
      if (next_[i] == static_cast<int32_t>(rec.observations.size())) facts_[i].reset();
    }
  });
}

void Replay::EncodeEvents(int32_t match, const std::vector<int32_t>& events, const CriticBatch& out) const {
  if (events.empty()) return;
  const MatchRecord& mr = *matches_[match];
  const int32_t first = events.front() / kSnapshotEvery;
  dragons::Game g = mr.snapshots.at(first);
  int32_t e = first * kSnapshotEvery;
  for (size_t r = 0; r < events.size(); ++r) {
    if (r && events[r] <= events[r - 1]) throw std::invalid_argument("events must be sorted and distinct");
    for (; e < events[r]; ++e) Play(g, mr.events[e].command);
    if (g.actor() != mr.events[e].dragon) throw std::logic_error("event replay diverged");

    int32_t* rows = out.dragons + r * out.d_pad * kCriticDragonFields;
    const int32_t alive = EncodeEvent(g, out.cells + r * kCriticCells * kCriticCellFields, out.d_pad, rows,
                                      out.global + r * kCriticGlobalFields);
    if (alive > out.d_pad) throw std::invalid_argument("d_pad below the living dragons");
    float* sum = out.summary + r * out.d_pad * (kHidden + 1);
    std::fill(sum, sum + out.d_pad * (kHidden + 1), 0.0f);
    for (int32_t i = 0; i < alive; ++i) {
      const int32_t id = rows[i * kCriticDragonFields + kDId];
      if (id >= static_cast<int32_t>(mr.lifetimes.size()) || mr.lifetimes[id] < 0) continue;
      const LifetimeRecord& rec = mr.lives[mr.lifetimes[id]];
      // its latest decision before this event (the h it carries into it)
      const auto it = std::lower_bound(rec.events.begin(), rec.events.end(), events[r]);
      if (it == rec.events.begin()) continue;
      const size_t k = static_cast<size_t>(it - rec.events.begin()) - 1;
      const size_t done = k + 1;                   // its h at the start of the current chunk (0 in the first)
      if (done >= static_cast<size_t>(kChunk))
        std::copy(rec.h[done / kChunk - 1].begin(), rec.h[done / kChunk - 1].end(), sum + i * (kHidden + 1));
      sum[i * (kHidden + 1) + kHidden] = rec.seen[k];
    }
  }
}

}  // namespace rl
