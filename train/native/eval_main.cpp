// Evaluation on the CPU with the plain-C++ policy (the submission's forward pass):
// every candidate against every opponent on every map, both sides, several seeds;
// one match per thread at a time.
//
//   dragons_eval --maps <file> [--maps-limit N] [--seeds 4] [--threads 28]
//                --candidate name=actor.bin ... --opponent greedy|random|name=actor.bin ...
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "collector.hpp"
#include "policy.hpp"

namespace {

struct Player {
  std::string name;
  int32_t slot;                   // index into the policies, or rl::kGreedy / rl::kRandom
};

std::vector<std::string> read_maps(const std::string& path, size_t limit) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::vector<std::string> maps;
  size_t n;
  while (maps.size() < limit && f >> n) {
    f.get();
    std::string text(n, '\0');
    f.read(text.data(), static_cast<std::streamsize>(n));
    maps.push_back(std::move(text));
  }
  if (maps.empty()) throw std::runtime_error("no maps in " + path);
  return maps;
}

// One decision's buffers (a one-row ActorBatch).
struct Row {
  std::vector<int32_t> win, wdest, grid, entity, segment, status, prev_dirs, echo_len;
  std::vector<uint8_t> prev_words, inbox, echo;
  std::vector<float> h;
  rl::ActorBatch batch;
  Row() : win(rl::kWinTiles * rl::kTileFields), wdest(rl::kWinTiles * 4), grid(rl::kGridCells * rl::kGridFields),
          entity(rl::kMaxEntities * rl::kEntityFields), segment(rl::kWindowTiles * rl::kSegmentFields),
          status(rl::kStatusFields), prev_dirs(rl::kPrevDirsFields), echo_len(5), prev_words(32),
          inbox(rl::kMaxInbox * 8), echo(5 * rl::kMaxEcho), h(rl::kHidden) {
    batch.k_pad = rl::kMaxInbox;
    batch.e_pad = rl::kMaxEcho;
    batch.win = win.data();
    batch.wdest = wdest.data();
    batch.grid = grid.data();
    batch.entity = entity.data();
    batch.segment = segment.data();
    batch.status = status.data();
    batch.prev_dirs = prev_dirs.data();
    batch.prev_words = prev_words.data();
    batch.inbox = inbox.data();
    batch.echo = echo.data();
    batch.echo_len = echo_len.data();
    batch.h = h.data();
  }
  rl::PolicyInput input() const {
    return {win.data(), wdest.data(), grid.data(), entity.data(), segment.data(), status.data(), prev_dirs.data(),
            prev_words.data(), inbox.data(), rl::kMaxInbox, echo.data(), rl::kMaxEcho, echo_len.data()};
  }
};

struct Result {
  int8_t winner = -1;
  int32_t rounds = 0;
  int64_t decisions = 0;
  double seconds = 0;
};

// Plays one match to its end.
Result play(const rl::MatchSetup& setup, const std::vector<std::unique_ptr<rl::Policy>>& policies, uint64_t seed) {
  const auto t0 = std::chrono::steady_clock::now();
  rl::Collector col({setup}, 1, false);
  Row row;
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> uni(0.0f, 1.0f);
  const std::vector<int32_t> only{0};
  Result r;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  while (!col.finished(0)) {
    const std::vector<rl::Pending> pend = col.Prepare();
    if (pend.empty()) break;
    col.EncodeActors(only, row.batch);
    float u[rl::kPolicyUniforms];
    for (float& x : u) x = uni(rng);
    const rl::PolicyOutput o = policies.at(pend[0].policy)->Act(row.input(), row.h.data(), u);
    rl::Command c;
    c.kind = static_cast<uint8_t>(o.kind);
    c.nsteps = static_cast<uint8_t>(o.kind == 0 ? o.nsteps : 0);
    for (int k = 0; k < rl::kMaxMoveTokens; ++k) c.dirs[k] = static_cast<uint8_t>(o.dirs[k]);
    c.split = o.split;
    c.sonar_mask = 0;
    for (int d = 0; d < 4; ++d)
      if (o.sonar[d] > 0) {
        c.sonar_mask |= 1 << d;
        c.words[d] = static_cast<uint64_t>(o.sonar[d] - 1);
      }
    col.Apply(only, {c}, &o.logp, o.h.data(), &nan);
    ++r.decisions;
  }
  const rl::MatchRecord& m = col.match_records()[0];
  r.winner = m.winner;
  r.rounds = m.rounds;
  r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  std::string maps_path;
  size_t maps_limit = 1u << 30;
  int seeds = 4, threads = static_cast<int>(std::thread::hardware_concurrency());
  std::vector<std::pair<std::string, std::string>> cand_args, opp_args;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i], v = argv[i + 1];
    auto named = [&](const std::string& s) {
      const size_t eq = s.find('=');
      return eq == std::string::npos ? std::make_pair(s, s) : std::make_pair(s.substr(0, eq), s.substr(eq + 1));
    };
    if (k == "--maps") maps_path = v;
    else if (k == "--maps-limit") maps_limit = std::stoul(v);
    else if (k == "--seeds") seeds = std::stoi(v);
    else if (k == "--threads") threads = std::stoi(v);
    else if (k == "--candidate") cand_args.push_back(named(v));
    else if (k == "--opponent") opp_args.push_back(named(v));
    else {
      std::fprintf(stderr, "unknown option %s\n", k.c_str());
      return 2;
    }
  }
  if (maps_path.empty() || cand_args.empty() || opp_args.empty()) {
    std::fprintf(stderr, "usage: %s --maps F --candidate name=actor.bin ... --opponent greedy|random|name=actor.bin ...\n",
                 argv[0]);
    return 2;
  }
  const std::vector<std::string> maps = read_maps(maps_path, maps_limit);
  std::vector<std::unique_ptr<rl::Policy>> policies;
  std::vector<Player> cands, opps;
  for (const auto& [name, path] : cand_args) {
    cands.push_back({name, static_cast<int32_t>(policies.size())});
    policies.push_back(std::make_unique<rl::Policy>(path));
  }
  for (const auto& [name, path] : opp_args) {
    if (path == "greedy") opps.push_back({name, rl::kGreedy});
    else if (path == "random") opps.push_back({name, rl::kRandom});
    else {
      opps.push_back({name, static_cast<int32_t>(policies.size())});
      policies.push_back(std::make_unique<rl::Policy>(path));
    }
  }
  struct Job {
    size_t cand, opp;
    int side;
    rl::MatchSetup setup;
    uint64_t seed;
  };
  std::vector<Job> jobs;
  for (size_t c = 0; c < cands.size(); ++c)
    for (size_t o = 0; o < opps.size(); ++o)
      for (size_t m = 0; m < maps.size(); ++m)
        for (int s = 0; s < seeds; ++s)
          for (int side = 0; side < 2; ++side) {
            Job j{c, o, side, {}, 0};
            j.setup.map_text = maps[m];
            j.setup.policy[side] = cands[c].slot;
            j.setup.policy[1 - side] = opps[o].slot;
            j.setup.record[0] = j.setup.record[1] = false;
            j.seed = 0x9e3779b97f4a7c15ULL * (1 + m) ^ (static_cast<uint64_t>(s) << 32 | static_cast<uint64_t>(side));
            j.setup.seed = j.seed;
            jobs.push_back(std::move(j));
          }
  std::printf("%zu matches: %zu candidates x %zu opponents x %zu maps x %d seeds x 2 sides, %d threads\n", jobs.size(),
              cands.size(), opps.size(), maps.size(), seeds, threads);
  std::fflush(stdout);
  std::vector<Result> results(jobs.size());
  std::atomic<size_t> next{0}, done{0};
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t)
    pool.emplace_back([&] {
      for (size_t i; (i = next++) < jobs.size();) {
        results[i] = play(jobs[i].setup, policies, jobs[i].seed);
        const size_t d = ++done;
        if (d % std::max<size_t>(1, jobs.size() / 10) == 0) {
          std::printf("  %zu/%zu matches (%.0fs)\n", d, jobs.size(),
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
          std::fflush(stdout);
        }
      }
    });
  for (auto& t : pool) t.join();
  const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  // per candidate x opponent: W/D/L and score (win + half a draw)
  int64_t decisions = 0;
  double match_s = 0;
  for (size_t c = 0; c < cands.size(); ++c) {
    std::printf("%s:", cands[c].name.c_str());
    double total = 0;
    int n_total = 0;
    for (size_t o = 0; o < opps.size(); ++o) {
      int w = 0, d = 0, l = 0;
      double rounds = 0;
      for (size_t i = 0; i < jobs.size(); ++i) {
        if (jobs[i].cand != c || jobs[i].opp != o) continue;
        const Result& r = results[i];
        if (r.winner < 0) ++d;
        else if (r.winner == jobs[i].side) ++w;
        else ++l;
        rounds += r.rounds;
      }
      const int n = w + d + l;
      const double score = (w + 0.5 * d) / std::max(n, 1);
      total += score * n;
      n_total += n;
      std::printf("  vs %s %.3f (W/D/L %d/%d/%d, %.0f rounds)", opps[o].name.c_str(), score, w, d, l, rounds / std::max(n, 1));
    }
    std::printf("  | overall %.3f\n", total / std::max(n_total, 1));
  }
  for (const Result& r : results) {
    decisions += r.decisions;
    match_s += r.seconds;
  }
  std::printf("%.0fs wall; %.1fs per match on one core, %.3f ms per decision (policy + game + features)\n", wall,
              match_s / results.size(), 1e3 * match_s / std::max<int64_t>(decisions, 1));
  return 0;
}
