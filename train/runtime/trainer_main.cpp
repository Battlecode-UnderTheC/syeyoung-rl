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
  int gpus = 4, workers = 2, matches = 896, threads = 48, batch = 65536, snapshot_every = 20;
  long updates = 1L << 40;          // to run in this process
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

struct Harvest {
  std::mutex mu;
  std::condition_variable cv;
  std::vector<std::shared_ptr<rl::Pack>> packs;
  int64_t decisions = 0;
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
  rt::Runtime rt(opt.plugin);
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
  rt::Learner learner(rt, opt.programs, devs, lc);
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
  const int64_t max_pending = 3LL * opt.batch;
  std::vector<std::thread> loops;
  for (auto& w : workers)
    loops.emplace_back([&, wp = w.get()] {
      try {
        int64_t n = 0;
        while (!stop) {
          {
            std::unique_lock<std::mutex> lock(harvest.mu);
            harvest.cv.wait(lock, [&] { return stop || harvest.decisions < max_pending; });
          }
          wp->step();
          if (++n % 16 == 0) {
            auto pack = wp->harvest();
            if (!pack->matches.empty()) {
              const int64_t d = pack_decisions(*pack);
              std::lock_guard<std::mutex> lock(harvest.mu);
              harvest.packs.push_back(std::move(pack));
              harvest.decisions += d;
              harvest.cv.notify_all();
            }
          }
        }
      } catch (const std::exception& e) {
        LOG("worker died: %s", e.what());
        std::abort();
      }
    });
  LOG("update %ld: %d devices x %d workers x %d matches, batch %d", st.update, opt.gpus, opt.workers, opt.matches,
      opt.batch);

  auto t_run = std::chrono::steady_clock::now();
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
    for (const auto& p : packs)
      for (const auto& m : p->matches) {
        ++matches;
        const bool r0 = m.setup.record[0], r1 = m.setup.record[1];
        for (int c = 0; c < 5; ++c) deaths[c] += (r0 ? m.deaths[c] : 0) + (r1 ? m.deaths[5 + c] : 0);
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
    LOG("update %ld: deaths (learner) wall %lld self %lld other_body %lld head_to_head %lld no_valid_action %lld",
        st.update, static_cast<long long>(deaths[0]), static_cast<long long>(deaths[1]),
        static_cast<long long>(deaths[2]), static_cast<long long>(deaths[3]), static_cast<long long>(deaths[4]));
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
