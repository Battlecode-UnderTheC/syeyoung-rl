// The trainer: collection workers and the PPO learner in one process, the
// model only as programs exported from JAX (dragons_rl/export.py). No Python.
//
//   dragons_train --plugin <xla_cuda_plugin.so> --programs <dir> --run <dir> [--maps <file>]
//                 [--gpus 4] [--workers 2] [--matches 896] [--threads 48] [--batch 65536] [--updates N]
#include <sys/stat.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <ctime>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "learner.hpp"
#include "maps.hpp"
#include "params.hpp"
#include "pjrt.hpp"
#include "worker.hpp"

namespace {

std::string now() {
  const std::time_t t = std::time(nullptr);
  char b[16];
  std::strftime(b, sizeof b, "%H:%M:%S", std::localtime(&t));
  return b;
}

#define LOG(...)                                   \
  do {                                             \
    std::printf("%s ", now().c_str());             \
    std::printf(__VA_ARGS__);                      \
    std::printf("\n");                             \
    std::fflush(stdout);                           \
  } while (0)

bool exists(const std::string& p) {
  struct stat s;
  return stat(p.c_str(), &s) == 0;
}

struct Options {
  std::string plugin, programs, run, maps;
  int gpus = 4, workers = 2, matches = 896, threads = 48, batch = 65536, snapshot_every = 20, rebuild_threads = 16;
  long updates = 1L << 40;          // to run in this process
  double collect_memory = 0.3, learner_memory = 0.45;   // GPU shares of the two clients
  double pending = 4.0;             // batches harvested ahead of the learner before workers pause
};

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i], v = argv[i + 1];
    if (k == "--plugin") o.plugin = v;
    else if (k == "--programs") o.programs = v;
    else if (k == "--run") o.run = v;
    else if (k == "--maps") o.maps = v;
    else if (k == "--gpus") o.gpus = std::stoi(v);
    else if (k == "--workers") o.workers = std::stoi(v);
    else if (k == "--matches") o.matches = std::stoi(v);
    else if (k == "--threads") o.threads = std::stoi(v);
    else if (k == "--batch") o.batch = std::stoi(v);
    else if (k == "--rebuild-threads") o.rebuild_threads = std::stoi(v);
    else if (k == "--collect-memory") o.collect_memory = std::stod(v);
    else if (k == "--pending") o.pending = std::stod(v);
    else if (k == "--learner-memory") o.learner_memory = std::stod(v);
    else if (k == "--updates") o.updates = std::stol(v);
    else throw std::invalid_argument("unknown option " + k);
  }
  if (o.maps.empty()) o.maps = o.programs + "/maps.txt";
  if (o.plugin.empty() || o.programs.empty() || o.run.empty()) throw std::invalid_argument("--plugin, --programs, --run");
  return o;
}

// ---- the run's state on disk ------------------------------------------------------------------------

struct State {
  long update = 0;
  int64_t decisions = 0;
  std::vector<std::string> history;               // league snapshots (actor weights), oldest first
};

void save_state(const std::string& run, const State& s, const rt::Params& actor, const rt::Params& critic,
                const rt::Adam& ao, const rt::Adam& co) {
  const std::string c = run + "/ckpt";
  actor.save(c + "/actor.bin");
  critic.save(c + "/critic.bin");
  ao.m.save(c + "/actor_m.bin");
  ao.v.save(c + "/actor_v.bin");
  co.m.save(c + "/critic_m.bin");
  co.v.save(c + "/critic_v.bin");
  std::ofstream f(c + "/state.tmp");
  f << "update " << s.update << "\ndecisions " << s.decisions << "\nactor_t " << ao.t << "\ncritic_t " << co.t << "\n";
  for (const auto& h : s.history) f << "history " << h << "\n";
  f.close();
  std::rename((c + "/state.tmp").c_str(), (c + "/state.txt").c_str());
}

// a checkpoint directory: this trainer's <run>/ckpt, or <programs>/start (export.py --from)
bool load_state(const std::string& c, State& s, rt::Params& actor, rt::Params& critic, rt::Adam& ao, rt::Adam& co) {
  if (!exists(c + "/state.txt")) return false;
  actor = rt::Params::load(c + "/actor.bin");
  critic = rt::Params::load(c + "/critic.bin");
  ao.m = rt::Params::load(c + "/actor_m.bin");
  ao.v = rt::Params::load(c + "/actor_v.bin");
  co.m = rt::Params::load(c + "/critic_m.bin");
  co.v = rt::Params::load(c + "/critic_v.bin");
  std::ifstream f(c + "/state.txt");
  std::string k;
  while (f >> k) {
    if (k == "update") f >> s.update;
    else if (k == "decisions") f >> s.decisions;
    else if (k == "actor_t") f >> ao.t;
    else if (k == "critic_t") f >> co.t;
    else if (k == "history") {
      std::string p;
      f >> p;
      s.history.push_back(p);
    }
  }
  return true;
}

// ---- harvested matches ----------------------------------------------------------------------------------

// Harvested matches: raw from the workers, then rebuilt (replayed with observations) by
// the rebuild threads, then taken by the learner.
struct Harvest {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<std::pair<std::shared_ptr<rl::Pack>, int64_t>> raw;   // pack, its decisions
  std::vector<std::shared_ptr<rl::Pack>> packs;                    // rebuilt
  int64_t raw_decisions = 0, decisions = 0;
};

int64_t pack_decisions(const rl::Pack& p) {
  int64_t n = 0;
  for (const auto& m : p.matches)
    for (const auto& l : m.lives) n += static_cast<int64_t>(l.logp.size());
  return n;
}

}  // namespace

int main(int argc, char** argv) {
  const Options opt = parse(argc, argv);
  mkdir(opt.run.c_str(), 0755);
  mkdir((opt.run + "/ckpt").c_str(), 0755);
  mkdir((opt.run + "/history").c_str(), 0755);
  std::FILE* metrics = std::fopen((opt.run + "/metrics.jsonl").c_str(), "a");

  State st;
  rt::Params actor, critic;
  rt::Adam aopt, copt;
  if (load_state(opt.run + "/ckpt", st, actor, critic, aopt, copt)) {
    LOG("resumed %s at update %ld", opt.run.c_str(), st.update);
  } else if (load_state(opt.programs + "/start", st, actor, critic, aopt, copt)) {
    save_state(opt.run, st, actor, critic, aopt, copt);
    LOG("created %s from the exported checkpoint (update %ld, %zu snapshots)", opt.run.c_str(), st.update,
        st.history.size());
  } else {
    actor = rt::Params::load(opt.programs + "/actor.bin");
    critic = rt::Params::load(opt.programs + "/critic.bin");
    aopt.init(actor);
    copt.init(critic);
    const std::string s0 = opt.run + "/history/S0.bin";
    actor.save(s0);
    st.history.push_back(s0);
    save_state(opt.run, st, actor, critic, aopt, copt);
    LOG("created %s from %s (%lld actor, %lld critic parameters)", opt.run.c_str(), opt.programs.c_str(),
        static_cast<long long>(actor.total()), static_cast<long long>(critic.total()));
  }

  rt::MapPool maps(opt.maps);
  // two clients: each has its own streams, so the workers' steps and the learner's calls
  // run concurrently on a GPU (with one, every program on a device queued on one stream)
  rt::Runtime rt(opt.plugin, rt::ClientOptions{opt.collect_memory, true});
  rt::Runtime lrt(opt.plugin, rt::ClientOptions{opt.learner_memory, true});
  if (rt.num_devices() < opt.gpus) throw std::runtime_error("fewer devices than --gpus");
  std::vector<std::unique_ptr<rt::Program>> acts(opt.gpus);
  {
    std::vector<std::thread> compile;
    for (int g = 0; g < opt.gpus; ++g)
      compile.emplace_back([&, g] { acts[g] = std::make_unique<rt::Program>(rt, opt.programs, "act", g); });
    for (auto& t : compile) t.join();
  }
  std::vector<int> devs(opt.gpus);
  for (int g = 0; g < opt.gpus; ++g) devs[g] = g;
  rt::LearnerConfig lc;
  lc.threads = opt.threads;
  lc.dump_dir = opt.run;
  rt::Learner learner(lrt, opt.programs, devs, lc);
  LOG("compiled the programs on %d devices", opt.gpus);

  // league: this update's opponent per worker (a recent or an older snapshot, evenly)
  std::map<std::string, std::shared_ptr<rt::Params>> league_cache;
  std::mt19937_64 rng(std::random_device{}());
  auto opponent = [&]() -> std::shared_ptr<rt::Params> {
    const size_t n = st.history.size(), recent = std::min<size_t>(8, n);
    const bool old = n > recent && (rng() % 2);
    const std::string& p = old ? st.history[rng() % (n - recent)] : st.history[n - recent + rng() % recent];
    auto it = league_cache.find(p);
    if (it == league_cache.end()) it = league_cache.emplace(p, std::make_shared<rt::Params>(rt::Params::load(p))).first;
    return it->second;
  };

  // workers
  const int threads_per_worker = std::max(1, opt.threads / (opt.gpus * opt.workers));
  std::vector<std::unique_ptr<rt::Worker>> workers;
  for (int g = 0; g < opt.gpus; ++g)
    for (int w = 0; w < opt.workers; ++w) {
      workers.push_back(std::make_unique<rt::Worker>(rt, *acts[g], opt.matches, threads_per_worker,
                                                      [&maps] { return maps.next(); }, rng()));
      workers.back()->set_weights(actor.arrays, opponent()->arrays, critic.arrays);
    }
  Harvest harvest;
  std::atomic<bool> stop{false};
  std::atomic<int64_t> paused_us{0};                // workers held back by the pending bound
  // harvested, not yet learnt from. Generous: one long match is ~50k decisions, so harvests
  // are lumpy, and pausing the workers stalls the matches the learner waits for next
  const int64_t max_pending = static_cast<int64_t>(opt.pending * opt.batch);
  std::vector<std::thread> loops;
  for (auto& w : workers)
    loops.emplace_back([&, wp = w.get()] {
      try {
        int64_t n = 0;
        while (!stop) {
          {
            const auto tp = std::chrono::steady_clock::now();
            std::unique_lock<std::mutex> lock(harvest.mu);
            harvest.cv.wait(lock, [&] { return stop || harvest.raw_decisions + harvest.decisions < max_pending; });
            paused_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tp).count();
          }
          wp->step();
          if (++n % 16 == 0) {
            auto pack = wp->harvest();
            if (!pack->matches.empty()) {
              const int64_t d = pack_decisions(*pack);
              std::lock_guard<std::mutex> lock(harvest.mu);
              harvest.raw.emplace_back(std::move(pack), d);
              harvest.raw_decisions += d;
              harvest.cv.notify_all();
            }
          }
        }
      } catch (const std::exception& e) {
        LOG("worker died: %s", e.what());
        std::abort();
      }
    });
  std::atomic<int64_t> rebuild_us{0};
  for (int r = 0; r < opt.rebuild_threads; ++r)
    loops.emplace_back([&] {
      try {
        while (true) {
          std::pair<std::shared_ptr<rl::Pack>, int64_t> job;
          {
            std::unique_lock<std::mutex> lock(harvest.mu);
            harvest.cv.wait(lock, [&] { return stop || !harvest.raw.empty(); });
            if (stop) return;
            job = std::move(harvest.raw.front());
            harvest.raw.pop_front();
          }
          const auto t0 = std::chrono::steady_clock::now();
          rl::RebuildPack(*job.first);                 // off the workers' and the learner's paths
          rebuild_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
          std::lock_guard<std::mutex> lock(harvest.mu);
          harvest.packs.push_back(std::move(job.first));
          harvest.raw_decisions -= job.second;
          harvest.decisions += job.second;
          harvest.cv.notify_all();
        }
      } catch (const std::exception& e) {
        LOG("rebuild died: %s", e.what());
        std::abort();
      }
    });
  LOG("update %ld: %d devices x %d workers x %d matches, batch %d", st.update, opt.gpus, opt.workers, opt.matches,
      opt.batch);

  auto t_run = std::chrono::steady_clock::now();
  struct Totals {
    int64_t steps = 0, decisions = 0, encode = 0, gpu = 0, apply = 0, paused = 0, rebuild = 0;
  };
  auto totals = [&] {
    Totals t;
    for (auto& wk : workers) {
      auto& ws = wk->stats();
      t.steps += ws.steps;
      t.decisions += ws.decisions;
      t.encode += ws.encode_us;
      t.gpu += ws.gpu_us;
      t.apply += ws.apply_us;
    }
    t.paused = paused_us;
    t.rebuild = rebuild_us;
    return t;
  };
  Totals before = totals();
  auto t_before = std::chrono::steady_clock::now();
  int64_t steps_seen = 0;
  const long last_update = st.update + opt.updates;   // --updates: how many this process runs
  while (st.update < last_update) {
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::shared_ptr<rl::Pack>> packs;
    int64_t have = 0;
    {
      std::unique_lock<std::mutex> lock(harvest.mu);
      harvest.cv.wait(lock, [&] { return harvest.decisions >= opt.batch; });
      packs.swap(harvest.packs);
      have = harvest.decisions;
      harvest.decisions = 0;
      harvest.cv.notify_all();
    }
    const double t_collect = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    auto t1 = std::chrono::steady_clock::now();
    // results before the learner rebuilds the packs (their W/D/L, rounds, deaths)
    int64_t w = 0, d = 0, l = 0, matches = 0, rounds = 0;
    std::array<int64_t, 5> deaths{};
    int64_t rammed = 0;                                      // enemies that died ramming a learner dragon
    for (const auto& p : packs)
      for (const auto& m : p->matches) {
        ++matches;
        const bool r0 = m.setup.record[0], r1 = m.setup.record[1];
        for (int c = 0; c < 5; ++c) deaths[c] += (r0 ? m.deaths[c] : 0) + (r1 ? m.deaths[5 + c] : 0);
        for (const auto& l : m.lives)
          if (m.setup.record[l.team]) rammed += l.rammed;
        if (r0 == r1) continue;                                  // self-play: even by symmetry
        const int side = r0 ? 0 : 1;
        w += m.winner == side;
        d += m.winner < 0;
        l += m.winner >= 0 && m.winner != side;
        rounds += m.rounds;
      }
    const rt::UpdateStats us = learner.update(std::move(packs), actor, critic, aopt, copt);
    const double t_train = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    ++st.update;
    st.decisions += us.decisions;
    if (st.update % opt.snapshot_every == 0) {
      const std::string p = opt.run + "/history/S" + std::to_string(st.update) + ".bin";
      actor.save(p);
      st.history.push_back(p);
    }
    for (auto& wk : workers) wk->set_weights(actor.arrays, opponent()->arrays, critic.arrays);
    save_state(opt.run, st, actor, critic, aopt, copt);
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    int64_t steps = 0, wdec = 0;
    for (auto& wk : workers) {
      steps += wk->stats().steps;
      wdec += wk->stats().decisions;
    }
    const double run_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_run).count();
    LOG("update %ld: %lld decisions from %lld matches, W/D/L %lld/%lld/%lld, rounds %.1f, kl max %.4f, %d steps%s, "
        "H %.3f/%.3f, vmse %.3f, %.0f learner decisions/s (collect %.1fs, train %.1fs: rebuild %.1f batch %.1f "
        "actor %.1f (encode %.2f gpu %.2f) critic %.1f; %d+%d calls); workers %.0f decisions/s",
        st.update, static_cast<long long>(us.decisions), static_cast<long long>(matches), static_cast<long long>(w),
        static_cast<long long>(d), static_cast<long long>(l), (w + d + l) ? double(rounds) / (w + d + l) : 0.0,
        us.kl_max, us.actor_steps, us.stopped ? " (kl stop)" : "", us.h_phys, us.h_sonar, us.value_mse,
        us.decisions / wall, t_collect, t_train, us.rebuild_s, us.batch_s, us.actor_s, us.actor_encode_s,
        us.actor_gpu_s, us.critic_s, us.actor_calls,
        us.critic_calls, wdec / run_s);
    LOG("update %ld: actor per device: mean %.2fs of max %.2fs; encode %.2f launch %.2f weights %.2f wait %.2f; "
        "calls %.0f%% filled; launch: upload %.2f execute %.2f download %.2f", st.update, us.actor_mean_s, us.actor_s,
        us.actor_encode_s, us.actor_launch_s,
        us.actor_weights_s, us.actor_gpu_s, us.actor_slots ? 100.0 * us.actor_fill / us.actor_slots : 0.0,
        us.launch_parts[0], us.launch_parts[1], us.launch_parts[2]);
    if (us.dropped_calls) LOG("update %ld: %d calls dropped: %s", st.update, us.dropped_calls, us.note.c_str());
    {
      const Totals now_t = totals();
      const double span = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_before).count();
      const double worker_s = span * 1e6 * static_cast<double>(workers.size());
      long rss_kb = 0;
      if (std::FILE* f = std::fopen("/proc/self/statm", "r")) {
        long pages = 0, resident = 0;
        if (std::fscanf(f, "%ld %ld", &pages, &resident) == 2) rss_kb = resident * 4;
        std::fclose(f);
      }
      LOG("update %ld: workers %.0f decisions/s, %.1f steps/s each; their time: encode %.2f gpu %.2f apply %.2f paused %.2f; "
          "rebuild threads %.0f%% busy; host memory %.1f GB",
          st.update, (now_t.decisions - before.decisions) / span, (now_t.steps - before.steps) / span / workers.size(),
          (now_t.encode - before.encode) / worker_s, (now_t.gpu - before.gpu) / worker_s,
          (now_t.apply - before.apply) / worker_s, (now_t.paused - before.paused) / worker_s,
          100.0 * (now_t.rebuild - before.rebuild) / (span * 1e6 * std::max(1, opt.rebuild_threads)), rss_kb / 1e6);
      before = now_t;
      t_before = std::chrono::steady_clock::now();
    }
    LOG("update %ld: deaths (learner) wall %lld self %lld other_body %lld head_to_head %lld no_valid_action %lld; "
        "rammed by enemies %lld",
        st.update, static_cast<long long>(deaths[0]), static_cast<long long>(deaths[1]),
        static_cast<long long>(deaths[2]), static_cast<long long>(deaths[3]), static_cast<long long>(deaths[4]),
        static_cast<long long>(rammed));
    std::fprintf(metrics,
                 "{\"update\": %ld, \"decisions\": %lld, \"total_decisions\": %lld, \"wall_s\": %.3f, \"collect_s\": %.3f, "
                 "\"train_s\": %.3f, \"kl_max\": %.6f, \"actor_steps\": %d, \"value_mse\": %.5f, \"h_phys\": %.5f, "
                 "\"h_sonar\": %.5f, \"clip_fraction\": %.5f, \"wins\": %lld, \"draws\": %lld, \"losses\": %lld, "
                 "\"worker_decisions_per_s\": %.1f, \"learner_decisions_per_s\": %.1f}\n",
                 st.update, static_cast<long long>(us.decisions), static_cast<long long>(st.decisions), wall, t_collect,
                 t_train, us.kl_max, us.actor_steps, us.value_mse, us.h_phys, us.h_sonar, us.clipped,
                 static_cast<long long>(w), static_cast<long long>(d), static_cast<long long>(l), wdec / run_s,
                 us.decisions / wall);
    std::fflush(metrics);
    (void)have;
    (void)steps_seen;
    (void)steps;
  }
  stop = true;
  harvest.cv.notify_all();
  for (auto& t : loops) t.join();
  return 0;
}
