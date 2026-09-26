#include "learner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "worker.hpp"

namespace rt {

namespace {

constexpr int kT = rl::kChunk;
constexpr int kSmallSplit = 32;                 // the split pads the actor_grad programs were exported for
constexpr int kWideSplit = 512;

double seconds_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

// first-fit decreasing: segments (lifetime, start, length <= T) into rows of T steps
struct Segment {
  int32_t life, start, length;
};
std::vector<std::vector<Segment>> pack_rows(std::vector<Segment> segs) {
  std::stable_sort(segs.begin(), segs.end(), [](const Segment& a, const Segment& b) { return a.length > b.length; });
  std::vector<std::vector<Segment>> rows;
  std::vector<int> free;
  std::vector<std::vector<int>> by_free(kT + 1);
  for (const Segment& s : segs) {
    int r = -1;
    for (int f = s.length; f <= kT && r < 0; ++f)
      while (!by_free[f].empty()) {
        const int cand = by_free[f].back();
        by_free[f].pop_back();
        if (free[cand] == f) {
          r = cand;
          break;
        }
      }
    if (r < 0) {
      r = static_cast<int>(rows.size());
      rows.emplace_back();
      free.push_back(kT);
    }
    rows[r].push_back(s);
    free[r] -= s.length;
    by_free[free[r]].push_back(r);
  }
  return rows;
}

rl::ActorBatch actor_rows(HostInputs& in, int64_t offset_rows) {
  auto at = [&](const std::string& n) {
    const TensorSpec& s = in.spec(n);
    const size_t row = s.bytes() / (s.dims[0] * s.dims[1]);          // [R, T, ...]: bytes of one decision
    return static_cast<uint8_t*>(in.data(n)) + offset_rows * row;
  };
  rl::ActorBatch b;
  b.k_pad = static_cast<int32_t>(in.spec("raw/inbox").dims[2]);
  b.e_pad = static_cast<int32_t>(in.spec("raw/echo").dims[3]);
  b.win = reinterpret_cast<int32_t*>(at("raw/win"));
  b.wdest = reinterpret_cast<int32_t*>(at("raw/wdest"));
  b.grid = reinterpret_cast<int32_t*>(at("raw/grid"));
  b.entity = reinterpret_cast<int32_t*>(at("raw/entity"));
  b.segment = reinterpret_cast<int32_t*>(at("raw/segment"));
  b.status = reinterpret_cast<int32_t*>(at("raw/status"));
  b.prev_dirs = reinterpret_cast<int32_t*>(at("raw/prev_dirs"));
  b.prev_words = at("raw/prev_words");
  b.inbox = at("raw/inbox");
  b.echo = at("raw/echo");
  b.echo_len = reinterpret_cast<int32_t*>(at("raw/echo_len"));
  b.h = reinterpret_cast<float*>(at("h0"));                          // unused by EncodeList (h comes from the batch)
  return b;
}

rl::CriticBatch critic_rows(HostInputs& in, int64_t offset) {
  auto at = [&](const std::string& n) {
    const TensorSpec& s = in.spec(n);
    return static_cast<uint8_t*>(in.data(n)) + offset * (s.bytes() / s.dims[0]);
  };
  rl::CriticBatch b;
  b.d_pad = static_cast<int32_t>(in.spec("raw/dragons").dims[1]);
  b.cells = reinterpret_cast<int32_t*>(at("raw/cells"));
  b.dragons = reinterpret_cast<int32_t*>(at("raw/dragons"));
  b.global = reinterpret_cast<int32_t*>(at("raw/global"));
  b.summary = reinterpret_cast<float*>(at("raw/summary"));
  return b;
}

// every input under `prefix` of a program: copy row `src` over rows [from, to) (padding rows, masked out)
void copy_rows(HostInputs& in, const Program& p, const std::string& prefix, int64_t rows, int64_t src, int64_t from,
               int64_t to) {
  for (const int i : in.indices()) {
    const TensorSpec& s = p.inputs()[i];
    if (s.name.compare(0, prefix.size(), prefix) != 0) continue;
    const size_t row = s.bytes() / rows;
    uint8_t* base = static_cast<uint8_t*>(in.by_index(i));
    for (int64_t r = from; r < to; ++r) std::memcpy(base + r * row, base + src * row, row);
  }
}

// a call's gradient outputs are all finite
bool finite_grads(const std::vector<std::vector<uint8_t>>& host, const std::vector<int>& gmap) {
  for (size_t o = 0; o < host.size(); ++o) {
    if (gmap[o] < 0) continue;
    const float* g = reinterpret_cast<const float*>(host[o].data());
    for (size_t q = 0, n = host[o].size() / sizeof(float); q < n; ++q)
      if (!std::isfinite(g[q])) return false;
  }
  return true;
}

// nonfinite values among a program's float inputs under `prefix` (diagnostics)
std::string nonfinite_inputs(HostInputs& in, const Program& p, const std::string& prefix) {
  std::string out;
  for (const int i : in.indices()) {
    const TensorSpec& s = p.inputs()[i];
    if (s.type != DType::F32 || s.name.compare(0, prefix.size(), prefix) != 0) continue;
    const float* x = static_cast<const float*>(in.by_index(i));
    size_t bad = 0;
    for (size_t q = 0, n = s.bytes() / sizeof(float); q < n; ++q) bad += !std::isfinite(x[q]);
    if (bad) out += " " + s.name + ":" + std::to_string(bad);
  }
  return out.empty() ? " none" : out;
}

// a call's inputs (host part and the weights) as named arrays, for reproducing it offline
void dump_call(const std::string& path, HostInputs& in, const Program& p, const Params& weights) {
  std::vector<HostArray> out;
  for (const int i : in.indices()) {
    HostArray a{p.inputs()[i], {}};
    const uint8_t* d = static_cast<const uint8_t*>(in.by_index(i));
    a.data.assign(d, d + a.spec.bytes());
    out.push_back(std::move(a));
  }
  for (HostArray a : weights.arrays) {
    a.spec.name = "params/" + a.spec.name;
    out.push_back(std::move(a));
  }
  write_arrays(path, out);
}

// one call in flight: inputs uploading, the program running, its outputs downloading
struct Flight {
  HostInputs* in = nullptr;
  std::vector<Buffer> up, outs;
  std::vector<std::vector<uint8_t>> host;       // the outputs, once landed
  std::unique_ptr<Transfers> tr, dl;
  std::vector<int64_t> dec;                     // actor calls: each row's decision (-1 padding)

  // `in` must stay put until land()
  static Flight launch(const Runtime& rt, const Program& prog, HostInputs& in, const Weights& w) {
    Flight f;
    f.in = &in;
    f.tr = std::make_unique<Transfers>(rt);
    std::vector<PJRT_Buffer*> args(prog.inputs().size());
    for (size_t i = 0; i < args.size(); ++i)
      if (w.has(static_cast<int>(i))) args[i] = w.get(static_cast<int>(i));
    for (const int i : in.indices()) {
      f.up.push_back(f.tr->upload(prog.device(), prog.inputs()[i], in.by_index(i)));
      args[i] = f.up.back().get();
    }
    f.outs = prog.run(args);                    // waits on the uploads on the device
    f.dl = std::make_unique<Transfers>(rt);
    f.host.resize(f.outs.size());
    for (size_t o = 0; o < f.outs.size(); ++o) {
      f.host[o].resize(prog.outputs()[o].bytes());
      f.dl->download(f.outs[o], f.host[o].data(), f.host[o].size());
    }
    return f;
  }
  void land() {
    dl->wait();
    tr->wait();
  }
};

}  // namespace

// ---- per device --------------------------------------------------------------------------------

struct Learner::Device {
  int id;
  std::unique_ptr<Program> actor32, actor512, critic;
  std::unique_ptr<HostInputs> in32[2], in512[2], inc;   // the actor's: two calls in flight
  std::vector<int> grad_param32, grad_param512, grad_paramc;          // output index -> parameter index (-1: not a grad)

  static std::vector<int> grad_map(const Program& p, const Params& params) {
    std::vector<int> m(p.outputs().size(), -1);
    for (size_t o = 0; o < p.outputs().size(); ++o) {
      const std::string& n = p.outputs()[o].name;
      if (n.compare(0, 6, "grads/") == 0) m[o] = params.index(n.substr(6));
    }
    return m;
  }
};

Learner::Learner(const Runtime& rt, const std::string& programs, const std::vector<int>& devices, LearnerConfig config)
    : rt_(rt), config_(config) {
  std::vector<std::thread> compile;
  devices_.resize(devices.size());
  for (size_t k = 0; k < devices.size(); ++k)
    compile.emplace_back([&, k] {
      auto d = std::make_unique<Device>();
      d->id = devices[k];
      d->actor32 = std::make_unique<Program>(rt, programs, "actor_grad_" + std::to_string(kSmallSplit), d->id);
      d->actor512 = std::make_unique<Program>(rt, programs, "actor_grad_" + std::to_string(kWideSplit), d->id);
      d->critic = std::make_unique<Program>(rt, programs, "critic_grad", d->id);
      for (int b = 0; b < 2; ++b) {
        d->in32[b] = std::make_unique<HostInputs>(*d->actor32, std::vector<std::string>{"params"});
        d->in512[b] = std::make_unique<HostInputs>(*d->actor512, std::vector<std::string>{"params"});
      }
      d->inc = std::make_unique<HostInputs>(*d->critic, std::vector<std::string>{"params"});
      devices_[k] = std::move(d);
    });
  for (auto& t : compile) t.join();
}

Learner::~Learner() = default;

// ---- one update -----------------------------------------------------------------------------------

UpdateStats Learner::update(std::vector<std::shared_ptr<rl::Pack>> packs, Params& actor, Params& critic,
                            Adam& actor_opt, Adam& critic_opt) {
  UpdateStats st;
  const int D = static_cast<int>(devices_.size());
  auto t0 = std::chrono::steady_clock::now();
  {                                                // packs collected without observations: rebuild (parallel)
    std::vector<rl::MatchRecord*> matches;
    for (auto& p : packs)
      for (auto& m : p->matches) matches.push_back(&m);
    std::atomic<size_t> next{0};
    std::vector<std::thread> pool;
    for (int w = 0; w < config_.threads; ++w)
      pool.emplace_back([&] {
        for (size_t i; (i = next++) < matches.size();) rl::RebuildMatch(*matches[i]);
      });
    for (auto& t : pool) t.join();
  }
  st.rebuild_s = seconds_since(t0);
  auto t1 = std::chrono::steady_clock::now();
  std::vector<std::shared_ptr<const rl::Pack>> cpacks(packs.begin(), packs.end());
  rl::Replay full(cpacks, 1);
  const int32_t L = full.num_lifetimes();
  // per lifetime and per decision
  std::vector<int64_t> start(L + 1, 0);
  std::vector<int32_t> wide(L);
  for (int32_t i = 0; i < L; ++i) {
    const rl::LifetimeRecord& r = full.lifetime(i);
    start[i + 1] = start[i] + static_cast<int64_t>(r.observations.size());
    int32_t longest = 0;
    for (const auto& o : r.observations) longest = std::max(longest, o.length);
    wide[i] = longest - 3 > kSmallSplit;             // the lifetime's calls: the wide split program
  }
  const int64_t N = start[L];
  st.decisions = N;
  std::vector<float> adv(N), logp_old(N), logp_new(N, std::numeric_limits<float>::quiet_NaN());
  std::vector<uint8_t> skip(N);
  for (int32_t i = 0; i < L; ++i) {
    const rl::LifetimeRecord& r = full.lifetime(i);
    const rl::MatchRecord& m = full.match(full.lifetime_match(i));
    const double G = m.winner < 0 ? 0.0 : (m.winner == r.team ? 1.0 : -1.0);
    const bool self_inflicted = r.death == 0 || r.death == 1 || r.death == 4;
    for (size_t t = 0; t < r.observations.size(); ++t) {
      const int64_t d = start[i] + static_cast<int64_t>(t);
      const double v = m.events[r.events[t]].value[r.team];
      adv[d] = static_cast<float>(G - v - (self_inflicted ? config_.death_penalty : 0.0));
      logp_old[d] = r.logp[t];
      skip[d] = !std::isfinite(adv[d]) || !std::isfinite(logp_old[d]);
      st.excluded += skip[d];
    }
  }
  // lifetimes to devices, largest first onto the least loaded
  std::vector<int32_t> order(L);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return start[a + 1] - start[a] > start[b + 1] - start[b]; });
  std::vector<std::vector<int32_t>> keep(D);
  std::vector<int64_t> load(D, 0);
  for (const int32_t i : order) {
    const int d = static_cast<int>(std::min_element(load.begin(), load.end()) - load.begin());
    keep[d].push_back(i);
    load[d] += start[i + 1] - start[i];
  }
  for (auto& k : keep) std::sort(k.begin(), k.end());
  std::vector<std::unique_ptr<rl::Replay>> replays(D);
  for (int d = 0; d < D; ++d)
    replays[d] = std::make_unique<rl::Replay>(cpacks, std::max(1, config_.threads / D), keep[d]);
  // the critic's events: a fraction of learner decisions, by device (match, event, team, target)
  struct CriticEvent {
    int32_t match, event, team;
    float target;
  };
  std::vector<std::vector<CriticEvent>> cevents(D);
  std::mt19937_64 rng(std::random_device{}());
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  {
    std::vector<int32_t> match_device(full.num_matches(), -1);
    std::vector<int64_t> cload(D, 0);
    for (int32_t i = 0; i < L; ++i) {
      const rl::LifetimeRecord& r = full.lifetime(i);
      const int32_t mi = full.lifetime_match(i);
      const rl::MatchRecord& m = full.match(mi);
      const float G = m.winner < 0 ? 0.0f : (m.winner == r.team ? 1.0f : -1.0f);
      for (size_t t = 0; t < r.observations.size(); ++t) {
        if (uni(rng) >= config_.critic_fraction) continue;
        if (match_device[mi] < 0) {
          match_device[mi] = static_cast<int32_t>(std::min_element(cload.begin(), cload.end()) - cload.begin());
        }
        cevents[match_device[mi]].push_back({mi, r.events[t], r.team, G});
        ++cload[match_device[mi]];
      }
    }
    for (auto& v : cevents) {
      std::shuffle(v.begin(), v.end(), rng);
      st.critic_events += static_cast<int64_t>(v.size());
    }
  }
  st.batch_s = seconds_since(t1);

  // ---- minibatches ----
  const int K = config_.minibatches;
  std::vector<std::vector<std::vector<int32_t>>> groups(D, std::vector<std::vector<int32_t>>(K));
  for (int d = 0; d < D; ++d) {
    std::vector<int32_t> g = keep[d];
    std::shuffle(g.begin(), g.end(), rng);
    for (size_t j = 0; j < g.size(); ++j) groups[d][j % K].push_back(g[j]);
    replays[d]->Reset();
  }
  // full lifetime index -> device-local index
  std::vector<int32_t> local(L, -1);
  for (int d = 0; d < D; ++d)
    for (size_t j = 0; j < keep[d].size(); ++j) local[keep[d][j]] = static_cast<int32_t>(j);

  bool actor_on = true;
  std::mutex note_mu;
  for (int k = 0; k < K; ++k) {
    int64_t n_k = 0;
    for (int d = 0; d < D; ++d)
      for (const int32_t i : groups[d][k]) n_k += start[i + 1] - start[i];
    int64_t m_k = 0;
    for (int d = 0; d < D; ++d) {
      const size_t a = cevents[d].size() * k / K, b = cevents[d].size() * (k + 1) / K;
      m_k += static_cast<int64_t>(b - a);
    }
    std::vector<Grads> agrads(D, Grads(actor)), cgrads(D, Grads(critic));
    std::vector<double> clipped(D, 0), hp(D, 0), hs(D, 0), closs(D, 0);
    std::vector<int> acalls(D, 0), ccalls(D, 0);
    std::vector<double> atime(D, 0), ctime(D, 0), aenc(D, 0), agpu(D, 0);
    auto drop = [&](const char* what, HostInputs& in, const Program& prog, const Params& weights) {
      std::lock_guard<std::mutex> lock(note_mu);
      ++st.dropped_calls;
      if (!st.note.empty()) return;
      st.note = std::string(what) + " call with a nonfinite gradient; nonfinite inputs:" + nonfinite_inputs(in, prog, "");
      if (!config_.dump_dir.empty() && dumps_ < 3) {
        const std::string path = config_.dump_dir + "/dropped_" + what + "_" + std::to_string(++dumps_) + ".bin";
        dump_call(path, in, prog, weights);
        st.note += "; inputs in " + path;
      }
    };
    // the actor's calls, two in flight: the next one is encoded while the GPU runs the last
    auto actor_work = [&](int d) {
      if (!actor_on || n_k == 0) return;
      Device& dev = *devices_[d];
      rl::Replay& rep = *replays[d];
      const auto ta = std::chrono::steady_clock::now();
      for (int kind = 0; kind < 2; ++kind) {
        Program& prog = kind ? *dev.actor512 : *dev.actor32;
        HostInputs* bufs[2] = {kind ? dev.in512[0].get() : dev.in32[0].get(), kind ? dev.in512[1].get() : dev.in32[1].get()};
        const std::vector<int>& gmap = kind ? dev.grad_param512 : dev.grad_param32;
        const int R = static_cast<int>(bufs[0]->spec("reset").dims[0]);
        std::vector<Segment> segs;
        for (const int32_t i : groups[d][k]) {
          if (wide[i] != kind) continue;
          const int32_t n = static_cast<int32_t>(start[i + 1] - start[i]);
          for (int32_t s = 0; s < n; s += kT) segs.push_back({i, s, std::min(kT, n - s)});
        }
        if (segs.empty()) continue;
        const auto rows = pack_rows(std::move(segs));
        Weights w;
        w.set(rt_, prog, "params", actor.arrays);
        auto land_call = [&](Flight& f) {
          const auto tw = std::chrono::steady_clock::now();
          f.land();
          agpu[d] += seconds_since(tw);
          if (!finite_grads(f.host, gmap)) {         // dropped (its decisions keep a NaN logp_new)
            drop("actor", *f.in, prog, actor);
            return;
          }
          for (size_t o = 0; o < f.host.size(); ++o) {
            const int pi = gmap[o];
            if (pi < 0) continue;
            const float* g = reinterpret_cast<const float*>(f.host[o].data());
            std::vector<float>& acc = agrads[d].g[pi];
            for (size_t q = 0; q < acc.size(); ++q) acc[q] += g[q];
          }
          const float* lp = reinterpret_cast<const float*>(f.host[prog.output("logp")].data());
          for (size_t e = 0; e < f.dec.size(); ++e)
            if (f.dec[e] >= 0) logp_new[f.dec[e]] = lp[e];
          clipped[d] += *reinterpret_cast<const int32_t*>(f.host[prog.output("stats/clipped")].data());
          hp[d] += *reinterpret_cast<const float*>(f.host[prog.output("stats/h_phys")].data());
          hs[d] += *reinterpret_cast<const float*>(f.host[prog.output("stats/h_sonar")].data());
          ++acalls[d];
        };
        std::optional<Flight> last;
        size_t c = 0;
        for (size_t r0 = 0; r0 < rows.size(); r0 += R, ++c) {
          HostInputs& in = *bufs[c % 2];              // the call before last (same buffer) has landed
          const auto te = std::chrono::steady_clock::now();
          const size_t r1 = std::min(rows.size(), r0 + R);
          std::vector<int32_t> which(R * kT, -1), step(R * kT, 0);
          std::vector<int64_t> dec(R * kT, -1);
          uint8_t* reset = in.as<uint8_t>("reset");
          uint8_t* mask = in.as<uint8_t>("mask");
          float* h0 = in.as<float>("h0");
          std::memset(reset, 0, R * kT);
          std::memset(mask, 0, R * kT);
          for (size_t r = r0; r < r1; ++r) {
            int t = 0;
            for (const Segment& s : rows[r]) {
              for (int j = 0; j < s.length; ++j) {
                const size_t e = (r - r0) * kT + t + j;
                which[e] = local[s.life];
                step[e] = s.start + j;
                dec[e] = start[s.life] + s.start + j;
              }
              reset[(r - r0) * kT + t] = 1;
              t += s.length;
            }
          }
          rep.EncodeList(which, step, actor_rows(in, 0));
          int64_t first = -1;
          for (int e = 0; e < R * kT; ++e)
            if (dec[e] >= 0) { first = e; break; }
          // per decision: h at chunk starts, the forced command, old logp, advantage, mask
          int32_t* kindp = in.as<int32_t>("forced/kind");
          int32_t* nsteps = in.as<int32_t>("forced/nsteps");
          int32_t* dirs = in.as<int32_t>("forced/dirs");
          int32_t* split = in.as<int32_t>("forced/split");
          int32_t* sonar = in.as<int32_t>("forced/sonar");
          float* lpo = in.as<float>("logp_old");
          float* ad = in.as<float>("adv");
          for (int e = 0; e < R * kT; ++e) {
            const int64_t dd = dec[e] >= 0 ? dec[e] : dec[first];
            const int32_t li = static_cast<int32_t>(std::upper_bound(start.begin(), start.end(), dd) - start.begin() - 1);
            const rl::LifetimeRecord& rec = full.lifetime(li);
            const int32_t ts = static_cast<int32_t>(dd - start[li]);
            const rl::Command& cm = rec.commands[ts];
            kindp[e] = cm.kind;
            nsteps[e] = cm.nsteps;
            for (int q = 0; q < 8; ++q) dirs[e * 8 + q] = cm.dirs[q];
            split[e] = cm.split;
            for (int q = 0; q < 4; ++q) sonar[e * 4 + q] = (cm.sonar_mask >> q) & 1 ? static_cast<int32_t>(cm.words[q]) + 1 : 0;
            float* h = h0 + static_cast<int64_t>(e) * rl::kHidden;
            if (ts > 0 && ts % kT == 0) std::copy(rec.h[ts / kT - 1].begin(), rec.h[ts / kT - 1].end(), h);
            else std::fill(h, h + rl::kHidden, 0.0f);
            lpo[e] = logp_old[dd];
            ad[e] = adv[dd];
            mask[e] = dec[e] >= 0 && !skip[dd];
          }
          // padding decisions: a real decision's inputs (masked out, but finite)
          for (int e = 0; e < R * kT; ++e)
            if (dec[e] < 0) copy_rows(in, prog, "raw/", R * kT, first, e, e + 1);
          *in.as<float>("denom") = static_cast<float>(n_k);
          aenc[d] += seconds_since(te);
          Flight f = Flight::launch(rt_, prog, in, w);
          f.dec = std::move(dec);
          if (last) land_call(*last);
          last = std::move(f);
        }
        if (last) land_call(*last);
      }
      atime[d] = seconds_since(ta);
    };
    // the critic's calls, concurrently with the actor's (EncodeEvents only reads the matches)
    auto critic_work = [&](int d) {
      const size_t a = cevents[d].size() * k / K, b = cevents[d].size() * (k + 1) / K;
      if (b <= a || m_k == 0) return;
      Device& dev = *devices_[d];
      const auto tc = std::chrono::steady_clock::now();
      Program& prog = *dev.critic;
      HostInputs& in = *dev.inc;
      const int C = static_cast<int>(in.spec("team").dims[0]);
      Weights w;
      w.set(rt_, prog, "params", critic.arrays);
      std::vector<CriticEvent> evs(cevents[d].begin() + a, cevents[d].begin() + b);
      std::sort(evs.begin(), evs.end(), [](const CriticEvent& x, const CriticEvent& y) {
        return x.match != y.match ? x.match < y.match : x.event < y.event;
      });
      for (size_t c0 = 0; c0 < evs.size(); c0 += C) {
        const size_t c1 = std::min(evs.size(), c0 + C);
        int32_t* tm = in.as<int32_t>("team");
        float* tg = in.as<float>("target");
        uint8_t* mk = in.as<uint8_t>("mask");
        std::memset(mk, 0, C);
        size_t at = 0;
        // one match's events at a time: each event is one dragon's decision, so distinct and sorted
        for (size_t q = c0; q < c1;) {
          size_t q1 = q;
          std::vector<int32_t> ev;
          while (q1 < c1 && evs[q1].match == evs[q].match) ev.push_back(evs[q1++].event);
          full.EncodeEvents(evs[q].match, ev, critic_rows(in, static_cast<int64_t>(at)));
          for (size_t z = q; z < q1; ++z, ++at) {
            tm[at] = evs[z].team;
            tg[at] = evs[z].target;
            mk[at] = 1;
          }
          q = q1;
        }
        if (at < static_cast<size_t>(C)) copy_rows(in, prog, "raw/", C, 0, static_cast<int64_t>(at), C);
        *in.as<float>("denom") = static_cast<float>(m_k);
        Flight f = Flight::launch(rt_, prog, in, w);
        f.land();
        if (!finite_grads(f.host, dev.grad_paramc)) {
          drop("critic", in, prog, critic);
          continue;
        }
        for (size_t o = 0; o < f.host.size(); ++o) {
          const int pi = dev.grad_paramc[o];
          if (pi < 0) continue;
          const float* g = reinterpret_cast<const float*>(f.host[o].data());
          std::vector<float>& acc = cgrads[d].g[pi];
          for (size_t z = 0; z < acc.size(); ++z) acc[z] += g[z];
        }
        closs[d] += *reinterpret_cast<const float*>(f.host[prog.output("loss")].data());
        ++ccalls[d];
      }
      ctime[d] = seconds_since(tc);
    };
    if (devices_[0]->grad_param32.empty())
      for (auto& dv : devices_) {
        dv->grad_param32 = Device::grad_map(*dv->actor32, actor);
        dv->grad_param512 = Device::grad_map(*dv->actor512, actor);
        dv->grad_paramc = Device::grad_map(*dv->critic, critic);
      }
    std::vector<std::future<void>> jobs;
    for (int d = 0; d < D; ++d) {
      jobs.push_back(std::async(std::launch::async, actor_work, d));
      jobs.push_back(std::async(std::launch::async, critic_work, d));
    }
    for (auto& j : jobs) j.get();
    double amax = 0, cmax = 0;
    for (int d = 0; d < D; ++d) {
      amax = std::max(amax, atime[d]);
      cmax = std::max(cmax, ctime[d]);
      st.actor_encode_s += aenc[d] / D;
      st.actor_gpu_s += agpu[d] / D;
      st.actor_calls += acalls[d];
      st.critic_calls += ccalls[d];
      st.clipped += clipped[d];
      st.h_phys += hp[d];
      st.h_sonar += hs[d];
      st.value_mse += closs[d] * m_k;
    }
    st.actor_s += amax;                              // the minibatch's slowest device
    st.critic_s += cmax;
    // actor step, KL guard
    if (actor_on && n_k > 0) {
      for (int d = 1; d < D; ++d) agrads[0].add(agrads[d]);
      double kl = 0;
      int64_t nk = 0;
      for (int d = 0; d < D; ++d)
        for (const int32_t i : groups[d][k])
          for (int64_t q = start[i]; q < start[i + 1]; ++q) {
            const double x = static_cast<double>(logp_new[q]) - logp_old[q];
            if (!std::isfinite(x)) continue;
            kl += std::exp(x) - 1.0 - x;
            ++nk;
          }
      kl = nk ? kl / nk : 0.0;
      st.kl_max = std::max(st.kl_max, kl);
      if (!std::isfinite(kl) || !agrads[0].finite()) throw std::runtime_error("nonfinite actor update");
      if (kl > config_.kl_limit) {
        actor_on = false;
        st.stopped = true;
      } else {
        st.actor_norm = agrads[0].clip(config_.grad_norm);
        actor_opt.step(actor, agrads[0]);
        ++st.actor_steps;
      }
    }
    if (m_k > 0) {
      for (int d = 1; d < D; ++d) cgrads[0].add(cgrads[d]);
      if (!cgrads[0].finite()) throw std::runtime_error("nonfinite critic gradient");
      st.critic_norm = cgrads[0].clip(config_.grad_norm);
      critic_opt.step(critic, cgrads[0]);
      ++st.critic_steps;
    }
  }
  st.value_mse /= std::max<int64_t>(1, st.critic_events);
  st.h_phys /= std::max<int64_t>(1, N);
  st.h_sonar /= std::max<int64_t>(1, N);
  st.clipped /= std::max<int64_t>(1, N);
  return st;
}

}  // namespace rt
