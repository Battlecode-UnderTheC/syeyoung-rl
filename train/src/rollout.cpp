#include "rollout.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "actions.hpp"

namespace train {

namespace dg = dragons;
using Clock = std::chrono::steady_clock;

namespace {
double Since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
}  // namespace

void Storage::Clear() {
  records.clear();
  action.clear();
  round.clear();
  log_prob.clear();
  value.clear();
  advantage.clear();
  returns.clear();
  next.clear();
  end.clear();
  end_value.clear();
  end_round.clear();
  chunks.clear();
  start_h.clear();
  start_map.clear();
  start_bins.clear();
}

Rollout::Rollout(const pjrt::Runtime& rt, const Artifacts& art, const std::vector<std::string>& maps, int device,
                 const RolloutOptions& opts)
    : rt_(rt),
      art_(art),
      device_(device),
      opts_(opts),
      N_(art.Get("num_envs")),
      S_(art.Get("pool")),
      R_(art.Get("record_bytes")),
      G_(art.Get("gather")),
      H_(art.Get("hidden")),
      C_(art.Get("map_cells")),
      M_(art.Get("map_channels")),
      B_(art.Get("bins")),
      chunk_(art.Get("chunk")),
      env_(maps, static_cast<int>(art.Get("num_envs")), opts.seed, opts.threads, static_cast<int>(art.Get("inbox_cap")),
           /*state=*/true),
      act_(rt, art.dir, "act"),
      value_(rt, art.dir, "value"),
      gather_(rt, art.dir, "gather"),
      rng_(opts.seed) {
  if (env_.layout().bytes != R_) throw std::runtime_error("the env's record size does not match the exported programs");
  // empty pools; the slot state of a dragon's first turn is zero anyway
  {
    std::vector<uint8_t> zeros(static_cast<size_t>(S_) * C_ * M_ * 2, 0);
    pool_h_ = rt.Upload(device, zeros.data(), pjrt::DType::F32, {S_, H_});
    pool_map_ = rt.Upload(device, zeros.data(), pjrt::DType::F16, {S_, C_, M_});
    pool_bins_ = rt.Upload(device, zeros.data(), pjrt::DType::F32, {S_, B_, M_});
  }
  const uint32_t key[2] = {static_cast<uint32_t>(opts.seed), static_cast<uint32_t>(opts.seed >> 32)};
  key_ = rt.Upload(device, key, pjrt::DType::U32, {2});
  scratch_ = static_cast<int32_t>(S_ - 1);
  for (int32_t s = scratch_ - 1; s >= 0; --s) free_slots_.push_back(s);
  tracks_.resize(N_);
}

int32_t Rollout::AllocSlot() {
  if (free_slots_.empty()) return scratch_;
  const int32_t s = free_slots_.back();
  free_slots_.pop_back();
  return s;
}

void Rollout::FreeSlot(int32_t slot) {
  if (slot >= 0 && slot != scratch_) free_slots_.push_back(slot);
}

void Rollout::EndGame(int i, Storage& out, RolloutStats& stats) {
  const int8_t winner = env_.winner[i];
  stats.games += 1;
  stats.rounds += env_.final_round[i] + 1;
  stats.eliminations += env_.end_reason[i] == 0;
  if (winner < 0) stats.draws += 1;
  else stats.wins[winner] += 1;
  for (auto& [id, tr] : tracks_[i]) {
    if (tr.last >= 0) {
      out.end[tr.last] = 1;
      out.end_value[tr.last] = winner < 0 ? 0.0f : (winner == tr.team ? 1.0f : -1.0f);
      out.end_round[tr.last] = env_.final_round[i];
    }
    FreeSlot(tr.slot);
  }
  tracks_[i].clear();
}

void Rollout::RunGathers(std::vector<PendingGather>& pending, Storage& out) {
  if (pending.empty()) return;
  std::vector<int32_t> slots(G_);
  std::vector<uint8_t> fresh(G_);
  std::vector<float> h(G_ * H_), bins(G_ * B_ * M_);
  std::vector<uint16_t> map(G_ * C_ * M_);
  const int64_t max_state = pending.back().state;
  out.start_h.resize((max_state + 1) * H_);
  out.start_map.resize((max_state + 1) * C_ * M_);
  out.start_bins.resize((max_state + 1) * B_ * M_);
  for (size_t at = 0; at < pending.size(); at += G_) {
    const size_t k = std::min<size_t>(G_, pending.size() - at);
    for (size_t r = 0; r < static_cast<size_t>(G_); ++r) {
      const bool real = r < k;
      slots[r] = real ? pending[at + r].slot : scratch_;
      fresh[r] = real ? pending[at + r].fresh : 1;
    }
    pjrt::Buffer sb = rt_.Upload(device_, slots.data(), pjrt::DType::S32, {G_});
    pjrt::Buffer fb = rt_.Upload(device_, fresh.data(), pjrt::DType::Pred, {G_});
    auto outs = gather_.Run(device_, {&pool_h_, &pool_map_, &pool_bins_, &sb, &fb});
    // only the first k rows are wanted: copy just those
    outs[0].RawToHost(h.data(), 0, static_cast<int64_t>(k) * H_ * 4);
    outs[1].RawToHost(map.data(), 0, static_cast<int64_t>(k) * C_ * M_ * 2);
    outs[2].RawToHost(bins.data(), 0, static_cast<int64_t>(k) * B_ * M_ * 4);
    for (size_t r = 0; r < k; ++r) {
      const int64_t s = pending[at + r].state;
      std::memcpy(&out.start_h[s * H_], &h[r * H_], H_ * 4);
      std::memcpy(&out.start_map[s * C_ * M_], &map[r * C_ * M_], C_ * M_ * 2);
      std::memcpy(&out.start_bins[s * B_ * M_], &bins[r * B_ * M_], B_ * M_ * 4);
    }
  }
  pending.clear();
}

RolloutStats Rollout::Collect(const std::vector<pjrt::Buffer>& params, Storage& out) {
  const auto t_start = Clock::now();
  out.Clear();
  out.record_bytes = R_;
  RolloutStats stats;
  const dg::Layout& L = env_.layout();
  std::vector<int32_t> slots(N_), action(N_);
  std::vector<uint8_t> fresh(N_);
  std::vector<float> log_prob(N_), value(N_);
  std::vector<PendingGather> pending;
  std::vector<uint8_t> kind(N_), steps(N_ * 2);
  std::vector<int32_t> nsteps(N_), split(N_);
  const std::vector<int64_t> sonar(N_, -1);

  auto scalars = [&](int64_t i) { return reinterpret_cast<const int32_t*>(env_.records() + i * R_ + L.scalars); };

  for (int step = 0; step < opts_.steps; ++step) {
    const int64_t base = out.size();
    const int64_t n = base + N_;
    out.action.resize(n);
    out.round.resize(n);
    out.log_prob.resize(n);
    out.value.resize(n);
    out.next.resize(n, -1);
    out.end.resize(n, 0);
    out.end_value.resize(n, 0.0f);
    out.end_round.resize(n, 0);
    // slots and chunks of this step's actors
    for (int64_t i = 0; i < N_; ++i) {
      const int32_t* sc = scalars(i);
      Track& tr = tracks_[i][sc[dg::kSId]];
      bool reset = false;
      if (tr.slot < 0 || tr.slot == scratch_) {
        // a dragon waiting for a slot takes one as soon as one is free, starting from zero
        reset = tr.slot == scratch_;
        tr.slot = AllocSlot();
        tr.team = static_cast<uint8_t>(sc[dg::kSTeam]);
      }
      const bool scratch = tr.slot == scratch_;
      fresh[i] = sc[dg::kSTurn] == 0 || scratch || reset;
      slots[i] = tr.slot;
      const int64_t j = base + i;
      out.round[j] = sc[dg::kSRound];
      if (scratch) {
        // pool full: the dragon plays from a zero state, and the turn is not trained on
        ++stats.overflow;
        out.end[j] = 3;
        continue;
      }
      if (tr.chunk < 0 || tr.chunk_turns >= chunk_ || reset) {
        tr.chunk = static_cast<int64_t>(out.chunks.size());
        tr.chunk_turns = 0;
        out.chunks.push_back({tr.chunk, {}});
        pending.push_back({tr.slot, fresh[i] != 0, tr.chunk});
      }
      out.chunks[tr.chunk].rows.push_back(j);
      ++tr.chunk_turns;
      if (tr.last >= 0) out.next[tr.last] = j;
      tr.last = j;
    }
    stats.max_slots = std::max<int64_t>(stats.max_slots, scratch_ - static_cast<int64_t>(free_slots_.size()));
    auto t0 = Clock::now();
    RunGathers(pending, out);
    stats.gather_seconds += Since(t0);

    // the policy
    t0 = Clock::now();
    pjrt::Buffer rec = rt_.Upload(device_, env_.records(), pjrt::DType::U8, {N_, R_});
    pjrt::Buffer sb = rt_.Upload(device_, slots.data(), pjrt::DType::S32, {N_});
    pjrt::Buffer fb = rt_.Upload(device_, fresh.data(), pjrt::DType::Pred, {N_});
    std::vector<const pjrt::Buffer*> args;
    for (const pjrt::Buffer& p : params) args.push_back(&p);
    for (const pjrt::Buffer* b : {&rec, &sb, &fb, &pool_h_, &pool_map_, &pool_bins_, &key_}) args.push_back(b);
    auto outs = act_.Run(device_, args);
    pool_h_ = std::move(outs[3]);
    pool_map_ = std::move(outs[4]);
    pool_bins_ = std::move(outs[5]);
    key_ = std::move(outs[6]);
    outs[0].ToHost(action.data(), N_ * 4);
    outs[1].ToHost(log_prob.data(), N_ * 4);
    outs[2].ToHost(value.data(), N_ * 4);
    stats.act_seconds += Since(t0);

    // store and act
    t0 = Clock::now();
    out.records.resize(n * R_);
    std::memcpy(&out.records[base * R_], env_.records(), N_ * R_);
    for (int64_t i = 0; i < N_; ++i) {
      const int64_t j = base + i;
      out.action[j] = action[i];
      out.log_prob[j] = log_prob[i];
      out.value[j] = value[i];
      stats.splits += action[i] >= 20;
      stats.sprints += action[i] >= 4 && action[i] < 20;
      const Command c = ActionCommand(action[i], scalars(i)[dg::kSLength]);
      kind[i] = c.kind;
      steps[2 * i] = c.steps[0];
      steps[2 * i + 1] = c.steps[1];
      nsteps[i] = c.nsteps;
      split[i] = c.split;
    }
    env_.Step(dg::StepActions{kind.data(), steps.data(), 2, nsteps.data(), split.data(), sonar.data()});
    for (size_t d = 0; d < env_.death_env.size(); ++d) {
      stats.deaths[env_.death_reason[d]] += 1;
      auto it = tracks_[env_.death_env[d]].find(env_.death_id[d]);
      if (it != tracks_[env_.death_env[d]].end()) {
        FreeSlot(it->second.slot);
        it->second.slot = -2;       // dead: keeps its track (team, last turn) until the game ends
      }
    }
    for (int64_t i = 0; i < N_; ++i)
      if (env_.done[i]) EndGame(static_cast<int>(i), out, stats);
    stats.env_seconds += Since(t0);
  }

  // bootstrap every trajectory still open from the critic's value of the position now
  {
    for (int64_t i = 0; i < N_; ++i) {
      const int32_t* sc = scalars(i);
      auto it = tracks_[i].find(sc[dg::kSId]);
      const bool known = it != tracks_[i].end() && it->second.slot >= 0;
      slots[i] = known ? it->second.slot : scratch_;
      fresh[i] = !known || sc[dg::kSTurn] == 0 || slots[i] == scratch_;
    }
    pjrt::Buffer rec = rt_.Upload(device_, env_.records(), pjrt::DType::U8, {N_, R_});
    pjrt::Buffer sb = rt_.Upload(device_, slots.data(), pjrt::DType::S32, {N_});
    pjrt::Buffer fb = rt_.Upload(device_, fresh.data(), pjrt::DType::Pred, {N_});
    std::vector<const pjrt::Buffer*> args;
    for (const pjrt::Buffer& p : params) args.push_back(&p);
    for (const pjrt::Buffer* b : {&rec, &sb, &fb, &pool_h_, &pool_map_, &pool_bins_}) args.push_back(b);
    auto outs = value_.Run(device_, args);
    outs[0].ToHost(value.data(), N_ * 4);
    for (int64_t i = 0; i < N_; ++i) {
      const int32_t* sc = scalars(i);
      for (auto& [id, tr] : tracks_[i]) {
        if (tr.last >= 0) {
          out.end[tr.last] = 2;
          out.end_value[tr.last] = tr.team == sc[dg::kSTeam] ? value[i] : -value[i];
          out.end_round[tr.last] = sc[dg::kSRound];
        }
        tr.chunk = -1;
        tr.chunk_turns = 0;
        tr.last = -1;
      }
    }
  }
  ComputeAdvantages(out);
  stats.transitions = out.size();
  stats.seconds = Since(t_start);
  return stats;
}

void Rollout::ComputeAdvantages(Storage& out) const {
  const int64_t n = out.size();
  out.advantage.assign(n, 0.0f);
  out.returns.assign(n, 0.0f);
  for (int64_t j = n - 1; j >= 0; --j) {
    double a;
    const int64_t k = out.next[j];
    if (out.end[j] == 3) {            // an overflow turn: not trained on
      out.advantage[j] = 0.0f;
      out.returns[j] = out.value[j];
      continue;
    }
    if (k >= 0) {
      const double g = std::pow(opts_.gamma, out.round[k] - out.round[j]);
      a = g * out.value[k] - out.value[j] + g * opts_.lambda * out.advantage[k];
    } else {
      if (out.end[j] == 0) throw std::logic_error("a trajectory ended with neither an outcome nor a bootstrap");
      const double g = std::pow(opts_.gamma, out.end_round[j] - out.round[j]);
      a = g * out.end_value[j] - out.value[j];
    }
    out.advantage[j] = static_cast<float>(a);
    out.returns[j] = static_cast<float>(a + out.value[j]);
  }
}

}  // namespace train
