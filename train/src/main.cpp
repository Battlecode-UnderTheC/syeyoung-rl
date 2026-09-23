// The trainer: rollouts on one GPU, PPO updates on another, forever (or for
// --iterations). Self-play: every dragon of both teams is the learner.
//
//   trainer --artifacts artifacts/v1 --plugin <xla_cuda_plugin.so> --maps DIR[,DIR...] --out runs/NAME
//
// Every iteration prints one line and appends one JSON object to <out>/metrics.jsonl;
// checkpoints are <out>/ckpt/<iteration>.bin (parameters and Adam state in the
// artifacts' state.txt layout), and --resume <file> continues from one.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "artifacts.hpp"
#include "dragons/core.hpp"
#include "pjrt.hpp"
#include "rollout.hpp"
#include "update.hpp"

namespace fs = std::filesystem;
using namespace train;

namespace {

// --name value flags; every flag must be known and every value must parse.
class Flags {
 public:
  Flags(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a.rfind("--", 0) != 0 || i + 1 >= argc) throw std::runtime_error("expected --name value, got " + a);
      values_[a.substr(2)] = argv[++i];
    }
  }
  template <typename T>
  T Get(const std::string& name, T fallback) {
    known_.push_back(name);
    auto it = values_.find(name);
    if (it == values_.end()) return fallback;
    std::istringstream in(it->second);
    T v;
    if (!(in >> v) || !in.eof()) throw std::runtime_error("bad value for --" + name + ": " + it->second);
    return v;
  }
  std::string Str(const std::string& name, const std::string& fallback) {
    known_.push_back(name);
    auto it = values_.find(name);
    return it == values_.end() ? fallback : it->second;
  }
  void CheckAllKnown() const {
    for (const auto& [k, v] : values_)
      if (std::find(known_.begin(), known_.end(), k) == known_.end()) throw std::runtime_error("unknown flag --" + k);
  }

 private:
  std::map<std::string, std::string> values_;
  std::vector<std::string> known_;
};

// every *.map in the listed directories, keeping boards the model's memory covers
std::vector<std::string> LoadMaps(const std::string& dirs, int max_side) {
  std::vector<std::string> out;
  std::istringstream list(dirs);
  for (std::string dir; std::getline(list, dir, ',');) {
    for (const auto& e : fs::directory_iterator(dir)) {
      if (e.path().extension() != ".map") continue;
      std::string text = pjrt::ReadFile(e.path().string());
      std::istringstream head(text);
      std::string key;
      int w = 0, h = 0;
      head >> key >> w >> h;
      if (w <= max_side && h <= max_side) out.push_back(std::move(text));
    }
  }
  if (out.empty()) throw std::runtime_error("no usable maps in " + dirs);
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Flags flags(argc, argv);
    const std::string art_dir = flags.Str("artifacts", "artifacts/v1");
    const std::string plugin = flags.Str("plugin", "");
    const std::string map_dirs = flags.Str("maps", "../dragons-env/dragons_env/maps");
    const std::string out_dir = flags.Str("out", "runs/default");
    const std::string resume = flags.Str("resume", "");
    const int rollout_device = flags.Get("rollout-device", 0);
    const std::string learner_list = flags.Str("learner-devices", "0,1,2,3");
    const int64_t iterations = flags.Get<int64_t>("iterations", 1000000);
    const int save_every = flags.Get("save-every", 20);
    RolloutOptions ro;
    ro.steps = flags.Get("steps", ro.steps);
    ro.gamma = flags.Get("gamma", ro.gamma);
    ro.lambda = flags.Get("lambda", ro.lambda);
    ro.threads = flags.Get("threads", 16);
    ro.seed = flags.Get<uint64_t>("seed", 0);
    UpdateOptions uo;
    uo.epochs = flags.Get("epochs", uo.epochs);
    uo.lr = flags.Get("lr", uo.lr);
    uo.clip = flags.Get("clip", uo.clip);
    uo.ent_coef = flags.Get("ent-coef", uo.ent_coef);
    uo.vf_coef = flags.Get("vf-coef", uo.vf_coef);
    uo.max_grad_norm = flags.Get("max-grad-norm", uo.max_grad_norm);
    uo.target_kl = flags.Get("target-kl", uo.target_kl);
    flags.CheckAllKnown();
    if (plugin.empty()) throw std::runtime_error("--plugin is required (jax_plugins/xla_cuda13/xla_cuda_plugin.so)");

    std::vector<int> learners;
    {
      std::istringstream in(learner_list);
      for (std::string tok; std::getline(in, tok, ',');) learners.push_back(std::stoi(tok));
    }
    if (learners.empty()) throw std::runtime_error("--learner-devices is empty");
    fs::create_directories(out_dir + "/ckpt");
    const Artifacts art = Artifacts::Load(art_dir);
    pjrt::Runtime rt(plugin);
    std::printf("pjrt: %d devices; rollout on %d, learners on %s\n", rt.num_devices(), rollout_device, learner_list.c_str());
    const std::vector<std::string> maps = LoadMaps(map_dirs, 64);
    std::printf("%zu maps, %lld envs, %d steps per rollout\n", maps.size(),
                static_cast<long long>(art.Get("num_envs")), ro.steps);

    int64_t start = 0;
    TrainState state = TrainState::Load(rt, art, resume.empty() ? art_dir + "/state.bin" : resume, learners[0]);
    if (!resume.empty()) start = std::stoll(fs::path(resume).stem().string()) + 1;
    Rollout rollout(rt, art, maps, rollout_device, ro);
    Updater updater(rt, art, learners, ro.seed + 1);
    std::vector<pjrt::Buffer> actor_params = state.CopyParams(rollout_device);
    Storage storage;
    std::ofstream metrics(out_dir + "/metrics.jsonl", std::ios::app);
    int64_t total_turns = 0;
    const auto t_run = std::chrono::steady_clock::now();

    for (int64_t it = start; it < iterations; ++it) {
      const RolloutStats rs = rollout.Collect(actor_params, storage);
      const UpdateStats us = updater.Update(state, storage, uo);
      actor_params = state.CopyParams(rollout_device);
      total_turns += rs.transitions;
      const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_run).count();
      const double games = std::max<int64_t>(rs.games, 1);
      std::printf("it %lld | %lld turns in %.1fs (act %.1f env %.1f gather %.1f) + update %.1fs (%lld rounds: grads %.1f apply %.1f) "
                  "| games %lld A %.2f B %.2f draw %.2f len %.0f elim %.2f | policy %.4f value %.4f entropy %.3f kl %.4f "
                  "clip %.3f ev %.3f | replay kl %.1e clip %.1e | slots %lld overflow %lld | %.0f turns/s\n",
                  static_cast<long long>(it), static_cast<long long>(rs.transitions), rs.seconds, rs.act_seconds,
                  rs.env_seconds, rs.gather_seconds, us.seconds, static_cast<long long>(us.rounds), us.round_seconds,
                  us.apply_seconds,
                  static_cast<long long>(rs.games), rs.wins[0] / games, rs.wins[1] / games, rs.draws / games,
                  rs.rounds / games, rs.eliminations / games, us.metrics[0], us.metrics[1], us.metrics[2], us.metrics[3],
                  us.metrics[4], us.explained_variance, us.replay_kl, us.replay_clip, static_cast<long long>(rs.max_slots),
                  static_cast<long long>(rs.overflow), total_turns / elapsed);
      std::fflush(stdout);
      metrics << "{\"iteration\":" << it << ",\"turns\":" << total_turns << ",\"elapsed\":" << elapsed
              << ",\"rollout_seconds\":" << rs.seconds << ",\"update_seconds\":" << us.seconds
              << ",\"games\":" << rs.games << ",\"win_a\":" << rs.wins[0] / games << ",\"win_b\":" << rs.wins[1] / games
              << ",\"draw\":" << rs.draws / games << ",\"game_rounds\":" << rs.rounds / games
              << ",\"eliminations\":" << rs.eliminations / games << ",\"splits\":" << rs.splits
              << ",\"sprints\":" << rs.sprints << ",\"death_wall\":" << rs.deaths[0] << ",\"death_self\":" << rs.deaths[1]
              << ",\"death_body\":" << rs.deaths[2] << ",\"death_head\":" << rs.deaths[3]
              << ",\"death_action\":" << rs.deaths[4] << ",\"policy_loss\":" << us.metrics[0]
              << ",\"value_loss\":" << us.metrics[1] << ",\"entropy\":" << us.metrics[2] << ",\"approx_kl\":" << us.metrics[3]
              << ",\"clip_fraction\":" << us.metrics[4] << ",\"actor_grad_norm\":" << us.metrics[5]
              << ",\"critic_grad_norm\":" << us.metrics[6] << ",\"explained_variance\":" << us.explained_variance
              << ",\"replay_kl\":" << us.replay_kl << ",\"replay_clip\":" << us.replay_clip
              << ",\"epochs\":" << us.epochs << ",\"max_slots\":" << rs.max_slots << ",\"overflow\":" << rs.overflow
              << "}\n";
      metrics.flush();
      if ((it + 1) % save_every == 0) state.Save(art, out_dir + "/ckpt/" + std::to_string(it) + ".bin");
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
