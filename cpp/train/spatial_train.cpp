// Native trainer for dragons-spatial-v1: collection, event GAE and the
// recurrent joint-command PPO update, all in C++ on libtorch (spec 8-9, 14).
//
// Shape of an iteration:
//   * every worker thread owns an ArenaGroup (one board size, thousands of
//     arenas) and a replica of the policy on its GPU; it plays `segment`
//     rounds, recording the first `record_arenas` arenas in full;
//   * the recorded arenas' event timelines become team advantages with the
//     event GAE, bootstrapped by the critic where the segment cut a match;
//   * the master takes PPO epochs over per-lifetime chunks, replaying each
//     from its snapshot on the GPU (sparse updates + on-device materialise),
//     burn-in without gradient, scored steps with one ratio per command and a
//     fixed team-round denominator;
//   * the updated parameters are copied to every replica.
//
// Unrecorded arenas still act with the policy: they keep the GPU batches
// large, which is what makes collection cheap per decision.

#include <ATen/Parallel.h>
#include <torch/cuda.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "train/arena.hpp"
#include "train/gpu_state.hpp"
#include "train/model.hpp"

using namespace dragons;
using namespace dragons::train;
namespace sp = dragons::spatial;

namespace {

struct Options {
  std::string model = "native";
  std::string maps = "native/maps.txt";
  std::string out = "runs/native";
  int threads = 4;
  int arenas_per_thread = 1024;
  int record_arenas = 64;
  int max_slots = 16;
  int segment = 16;             // rounds per iteration
  int iterations = 1000;
  int epochs = 3;
  int sequence_length = 64;
  int burn_in = 16;
  int microbatch = 64;          // chunks per microbatch
  int snapshot_every = 16;
  int max_steps = 8;
  bool sonar = false;
  double lr = 3e-4;
  double clip = 0.1;
  double value_coef = 0.5;
  double entropy_coef = 0.005;
  double sonar_entropy_coef = 0.005;
  double grad_norm = 0.5;
  double target_kl = 0.02;
  double gamma_round = 1.0;
  double lambda_round = 0.99;
  int save_every = 10;
  uint64_t seed = 0;
};

template <typename T>
T Arg(int argc, char** argv, const std::string& name, T fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (name == argv[i]) {
      std::istringstream in(argv[i + 1]);
      T value;
      in >> value;
      return value;
    }
  }
  return fallback;
}

bool Flag(int argc, char** argv, const std::string& name) {
  for (int i = 1; i < argc; ++i) if (name == argv[i]) return true;
  return false;
}

// maps.txt: map texts separated by lines of "---"
std::map<std::pair<int, int>, std::vector<std::string>> LoadMaps(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read " + path);
  std::map<std::pair<int, int>, std::vector<std::string>> groups;
  std::string line, current;
  auto flush = [&] {
    if (current.empty()) return;
    std::istringstream head(current);
    std::string key;
    int w, h;
    head >> key >> w >> h;
    groups[{w, h}].push_back(current);
    current.clear();
  };
  while (std::getline(in, line)) {
    if (line == "---") { flush(); continue; }
    current += line + "\n";
  }
  flush();
  return groups;
}

struct Chunk {
  const ArenaRecord* arena;
  const LifetimeRecord* lifetime;
  int snapshot;      // index into lifetime->snapshots
  int start, stop;   // turn indices [start, stop) scored, replay begins at the snapshot's turn
  int dims_w, dims_h;
  int team;
  std::vector<float> advantage;   // per scored turn
};

// Event GAE per arena per team (spec 8).
void Advantages(const ArenaRecord& rec, const std::array<float, 2>& bootstrap, double gamma, double lambda,
                std::vector<std::array<float, 2>>& adv, std::vector<std::array<float, 2>>& ret) {
  const int E = static_cast<int>(rec.events.size());
  adv.assign(E, {0.0f, 0.0f});
  ret.assign(E, {0.0f, 0.0f});
  for (int team = 0; team < 2; ++team) {
    double running = 0.0;
    for (int e = E - 1; e >= 0; --e) {
      const EventRecord& ev = rec.events[e];
      const double g = std::pow(gamma, ev.delta_rounds), l = std::pow(lambda, ev.delta_rounds);
      const double cont = ev.terminal ? 0.0 : 1.0;
      const double next = e + 1 < E ? rec.events[e + 1].value[team] : bootstrap[team];
      const double delta = ev.reward[team] + cont * g * next - ev.value[team];
      running = delta + cont * g * l * running;
      adv[e][team] = static_cast<float>(running);
      ret[e][team] = static_cast<float>(running + ev.value[team]);
    }
  }
}

struct Worker {
  std::unique_ptr<Model> model;
  std::unique_ptr<ArenaGroup> group;
  torch::Device device{torch::kCPU};
  std::vector<ArenaRecord> records;
  std::vector<std::array<float, 2>> bootstrap;
  int64_t decisions = 0;
};

}  // namespace

int main(int argc, char** argv) {
  Options o;
  o.model = Arg<std::string>(argc, argv, "--model", o.model);
  o.maps = Arg<std::string>(argc, argv, "--maps", o.maps);
  o.out = Arg<std::string>(argc, argv, "--out", o.out);
  o.threads = Arg(argc, argv, "--threads", o.threads);
  o.arenas_per_thread = Arg(argc, argv, "--arenas-per-thread", o.arenas_per_thread);
  o.record_arenas = Arg(argc, argv, "--record-arenas", o.record_arenas);
  o.max_slots = Arg(argc, argv, "--max-slots", o.max_slots);
  o.segment = Arg(argc, argv, "--segment", o.segment);
  o.iterations = Arg(argc, argv, "--iterations", o.iterations);
  o.epochs = Arg(argc, argv, "--epochs", o.epochs);
  o.sequence_length = Arg(argc, argv, "--sequence-length", o.sequence_length);
  o.burn_in = Arg(argc, argv, "--burn-in", o.burn_in);
  o.microbatch = Arg(argc, argv, "--microbatch", o.microbatch);
  o.snapshot_every = Arg(argc, argv, "--snapshot-every", o.snapshot_every);
  o.max_steps = Arg(argc, argv, "--max-steps", o.max_steps);
  o.sonar = Flag(argc, argv, "--sonar");
  o.lr = Arg(argc, argv, "--lr", o.lr);
  o.clip = Arg(argc, argv, "--clip", o.clip);
  o.entropy_coef = Arg(argc, argv, "--entropy-coef", o.entropy_coef);
  o.target_kl = Arg(argc, argv, "--target-kl", o.target_kl);
  o.save_every = Arg(argc, argv, "--save-every", o.save_every);
  o.seed = Arg<uint64_t>(argc, argv, "--seed", o.seed);
  const bool collect_only = Flag(argc, argv, "--collect-only");

  const int gpus = torch::cuda::is_available() ? static_cast<int>(torch::cuda::device_count()) : 0;
  const torch::Device master_device = gpus ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);
  std::printf("gpus %d, master on %s\n", gpus, master_device.str().c_str());
  at::set_num_threads(1);
  std::filesystem::create_directories(o.out);

  auto groups = LoadMaps(o.maps);
  std::vector<std::pair<std::pair<int, int>, std::vector<std::string>>> dims(groups.begin(), groups.end());
  Model master(o.model, master_device, o.sonar, o.max_steps);
  std::vector<Worker> workers(o.threads);
  for (int t = 0; t < o.threads; ++t) {
    Worker& w = workers[t];
    w.device = gpus ? torch::Device(torch::kCUDA, t % gpus) : torch::Device(torch::kCPU);
    w.model = std::make_unique<Model>(o.model, w.device, o.sonar, o.max_steps);
    const auto& [wh, maps] = dims[t % dims.size()];
    GroupOptions g;
    g.arenas = o.arenas_per_thread;
    g.max_slots = o.max_slots;
    g.record_arenas = o.record_arenas;
    g.snapshot_every = o.snapshot_every;
    g.max_steps = o.max_steps;
    g.sonar = o.sonar;
    g.seed = o.seed * 1000 + t;
    w.group = std::make_unique<ArenaGroup>(wh.first, wh.second, maps, g, *w.model, w.device);
    std::printf("thread %d: %dx%d, %d arenas on %s\n", t, wh.first, wh.second, g.arenas, w.device.str().c_str());
  }

  std::vector<torch::Tensor> params = master.ActorParameters();
  for (auto p : master.CriticParameters()) params.push_back(p);
  for (auto& p : params) p.set_requires_grad(true);
  torch::optim::Adam optimizer(params, torch::optim::AdamOptions(o.lr));
  std::mt19937_64 rng(o.seed);

  for (int iteration = 0; iteration < o.iterations; ++iteration) {
    // ---- collect ---------------------------------------------------------
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    for (Worker& w : workers) {
      threads.emplace_back([&w, &o] {
        const int64_t before = w.group->decisions();
        for (int r = 0; r < o.segment; ++r) w.group->PlayRound();
        w.bootstrap = w.group->BootstrapValues();
        w.records = w.group->TakeRecords();
        w.group->ResetFinished();
        w.decisions = w.group->decisions() - before;
      });
    }
    for (auto& th : threads) th.join();
    const double collect_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    int64_t decisions = 0;
    for (const Worker& w : workers) decisions += w.decisions;
    if (collect_only) {
      std::printf("iter %d: %lld decisions in %.2fs = %.0f decisions/s\n", iteration, static_cast<long long>(decisions),
                  collect_s, decisions / collect_s);
      continue;
    }

    // ---- advantages and chunks -------------------------------------------
    const auto t1 = std::chrono::steady_clock::now();
    std::vector<Chunk> chunks;
    std::vector<std::vector<std::vector<std::array<float, 2>>>> returns(workers.size());
    int64_t scored_total = 0;
    std::set<std::tuple<int, int, int, int>> team_rounds;   // (worker, arena, team, round)
    for (size_t wi = 0; wi < workers.size(); ++wi) {
      Worker& w = workers[wi];
      returns[wi].resize(w.records.size());
      for (size_t ai = 0; ai < w.records.size(); ++ai) {
        const ArenaRecord& rec = w.records[ai];
        std::vector<std::array<float, 2>> adv;
        Advantages(rec, w.bootstrap[ai], o.gamma_round, o.lambda_round, adv, returns[wi][ai]);
        for (const auto& [id, lr] : rec.lifetimes) {
          if (!lr.learner || lr.turns.empty() || lr.snapshots.empty()) continue;
          const int n = static_cast<int>(lr.turns.size());
          for (int start = 0; start < n; start += o.sequence_length) {
            const int stop = std::min(n, start + o.sequence_length);
            const int want = std::max(lr.turns[start].turn_index - o.burn_in, lr.snapshots.front().turn_index);
            int snap = 0;
            for (size_t s = 0; s < lr.snapshots.size(); ++s) if (lr.snapshots[s].turn_index <= want) snap = static_cast<int>(s);
            Chunk c{&rec, &lr, snap, start, stop, w.group->width(), w.group->height(), lr.team, {}};
            for (int t = start; t < stop; ++t) {
              c.advantage.push_back(adv[lr.turns[t].event][lr.team]);
              team_rounds.insert({static_cast<int>(wi), rec.arena, lr.team, lr.turns[t].round});
            }
            scored_total += stop - start;
            chunks.push_back(std::move(c));
          }
        }
      }
    }
    if (chunks.empty()) { std::printf("iter %d: nothing recorded\n", iteration); continue; }
    double mean = 0, var = 0;
    for (const Chunk& c : chunks) for (float a : c.advantage) mean += a;
    mean /= std::max<int64_t>(scored_total, 1);
    for (const Chunk& c : chunks) for (float a : c.advantage) var += (a - mean) * (a - mean);
    const double std_ = std::sqrt(var / std::max<int64_t>(scored_total, 1)) + 1e-8;
    for (Chunk& c : chunks) for (float& a : c.advantage) a = static_cast<float>((a - mean) / std_);
    const double denominator = static_cast<double>(std::max<size_t>(team_rounds.size(), 1));

    // ---- PPO -------------------------------------------------------------
    double kl_sum = 0, clip_sum = 0, policy_loss_sum = 0, value_loss_sum = 0, entropy_sum = 0;
    double replay_parity = -1.0;    // max |replayed - stored| log prob before the first step
    int epochs_done = 0;
    for (int epoch = 0; epoch < o.epochs; ++epoch) {
      std::vector<size_t> order(chunks.size());
      std::iota(order.begin(), order.end(), 0);
      std::shuffle(order.begin(), order.end(), rng);
      // microbatches must share a board size
      std::map<std::pair<int, int>, std::vector<size_t>> by_dims;
      for (size_t i : order) by_dims[{chunks[i].dims_w, chunks[i].dims_h}].push_back(i);
      double epoch_kl = 0, epoch_clip = 0;
      for (auto& [wh, members] : by_dims) {
        const int W = wh.first, H = wh.second;
        for (size_t mb = 0; mb < members.size(); mb += o.microbatch) {
          std::vector<const Chunk*> batch;
          for (size_t i = mb; i < std::min(members.size(), mb + static_cast<size_t>(o.microbatch)); ++i) batch.push_back(&chunks[members[i]]);
          const int64_t B = static_cast<int64_t>(batch.size());
          const auto fopt = torch::TensorOptions().dtype(torch::kFloat).device(master_device);
          torch::Tensor raw = torch::zeros({B, kRaw, H, W}, fopt);
          torch::Tensor M = torch::zeros({B, kLatent, H, W}, fopt);
          torch::Tensor h = torch::zeros({B, kHidden}, fopt);
          int T = 0;
          std::vector<int> first_turn(B), scored_from(B);
          for (int64_t b = 0; b < B; ++b) {
            const Chunk& c = *batch[b];
            const Snapshot& s = c.lifetime->snapshots[c.snapshot];
            raw[b] = s.raw.to(master_device);
            M[b] = s.M.to(master_device);
            h[b] = s.h.to(master_device);
            // replay from the first recorded turn at or after the snapshot's turn index
            int t0i = 0;
            while (t0i < c.start && c.lifetime->turns[t0i].turn_index < s.turn_index) ++t0i;
            first_turn[b] = t0i;
            scored_from[b] = c.start;
            T = std::max(T, c.stop - t0i);
          }
          std::vector<torch::Tensor> logps, ents, sents, advs, olds;
          for (int t = 0; t < T; ++t) {
            std::vector<int64_t> rows;
            std::vector<const TurnRecord*> turns;
            std::vector<bool> scored;
            for (int64_t b = 0; b < B; ++b) {
              const int ti = first_turn[b] + t;
              if (ti >= batch[b]->stop) continue;
              rows.push_back(b);
              turns.push_back(&batch[b]->lifetime->turns[ti]);
              scored.push_back(ti >= scored_from[b]);
            }
            if (rows.empty()) break;
            const int64_t R = static_cast<int64_t>(rows.size());
            torch::Tensor rows_t = torch::tensor(rows, torch::kLong).to(master_device);
            // small tensors
            torch::Tensor status = torch::zeros({R, sp::kStatus}, torch::kFloat);
            torch::Tensor history = torch::zeros({R, sp::kHistoryTurns, sp::kHistoryFeatures}, torch::kFloat);
            torch::Tensor history_mask = torch::zeros({R, sp::kHistoryTurns}, torch::kBool);
            torch::Tensor entities = torch::zeros({R, sp::kMaxEntities, sp::kEntityFeatures}, torch::kFloat);
            torch::Tensor entity_mask = torch::zeros({R, sp::kMaxEntities}, torch::kBool);
            torch::Tensor xy = torch::zeros({R, sp::kTiles, 2}, torch::kLong);
            torch::Tensor head_xy = torch::zeros({R, 2}, torch::kLong);
            torch::Tensor rounds = torch::zeros({R}, torch::kLong);
            int64_t max_pairs = 0, max_loose = 0, max_inbox = 0, max_split = 0;
            std::vector<RawUpdate> updates;
            for (int64_t r = 0; r < R; ++r) {
              const TurnRecord& tr = *turns[r];
              std::copy(tr.status.begin(), tr.status.end(), status[r].data_ptr<float>());
              std::copy(tr.history.begin(), tr.history.end(), history[r].data_ptr<float>());
              for (int i = 0; i < sp::kHistoryTurns; ++i) history_mask[r][i] = tr.history_mask[i] != 0;
              std::copy(tr.entities.begin(), tr.entities.end(), entities[r].data_ptr<float>());
              for (int i = 0; i < sp::kMaxEntities; ++i) entity_mask[r][i] = tr.entity_mask[i] != 0;
              std::copy(tr.visible_xy.begin(), tr.visible_xy.end(), xy[r].data_ptr<int64_t>());
              head_xy[r][0] = tr.head_x; head_xy[r][1] = tr.head_y;
              rounds[r] = tr.round;
              max_pairs = std::max<int64_t>(max_pairs, tr.portal_pairs.size());
              max_loose = std::max<int64_t>(max_loose, tr.portal_unpaired.size());
              max_inbox = std::max<int64_t>(max_inbox, tr.inbox.size());
              max_split = std::max<int64_t>(max_split, tr.split_sizes.size());
              for (RawUpdate u : tr.updates) { u.slot = static_cast<int32_t>(rows[r]); updates.push_back(u); }
            }
            torch::Tensor pairs = torch::zeros({R, max_pairs, 2, 2}, torch::kLong);
            torch::Tensor pair_mask = torch::zeros({R, max_pairs}, torch::kBool);
            torch::Tensor axis = torch::zeros({R, max_pairs}, torch::kLong);
            torch::Tensor loose = torch::zeros({R, max_loose, 2}, torch::kLong);
            torch::Tensor loose_mask = torch::zeros({R, max_loose}, torch::kBool);
            torch::Tensor inbox = torch::zeros({R, max_inbox, 4}, torch::kLong);
            torch::Tensor inbox_mask = torch::zeros({R, max_inbox}, torch::kBool);
            StoredCommand sc;
            const int K = o.max_steps;
            torch::Tensor top = torch::zeros({R}, torch::kLong), split_index = torch::zeros({R}, torch::kLong), sonar = torch::zeros({R}, torch::kLong);
            torch::Tensor tokens = torch::full({R, K}, sp::kStop, torch::kLong);
            torch::Tensor active = torch::zeros({R, K}, torch::kBool), masks = torch::zeros({R, K, sp::kTokens}, torch::kBool);
            torch::Tensor prefix = torch::zeros({R, K, sp::kPrefixFeatures}, torch::kFloat);
            torch::Tensor split_sizes = torch::zeros({R, max_split}, torch::kLong), split_feats = torch::zeros({R, max_split, 4}, torch::kFloat);
            torch::Tensor split_mask = torch::zeros({R, max_split}, torch::kBool), split_legal = torch::zeros({R}, torch::kBool);
            torch::Tensor old = torch::zeros({R}, torch::kFloat), adv = torch::zeros({R}, torch::kFloat), scored_t = torch::zeros({R}, torch::kBool);
            for (int64_t r = 0; r < R; ++r) {
              const TurnRecord& tr = *turns[r];
              for (size_t i = 0; i < tr.portal_pairs.size(); ++i) { for (int j = 0; j < 4; ++j) pairs[r][i][j / 2][j % 2] = tr.portal_pairs[i][j]; pair_mask[r][i] = true; axis[r][i] = tr.portal_axis[i]; }
              for (size_t i = 0; i < tr.portal_unpaired.size(); ++i) { loose[r][i][0] = tr.portal_unpaired[i][0]; loose[r][i][1] = tr.portal_unpaired[i][1]; loose_mask[r][i] = true; }
              for (size_t i = 0; i < tr.inbox.size(); ++i) { for (int b = 0; b < 4; ++b) inbox[r][i][b] = static_cast<int64_t>((tr.inbox[i] >> (8 * b)) & 0xFF); inbox_mask[r][i] = true; }
              top[r] = tr.top; split_index[r] = tr.split_index; sonar[r] = tr.sonar;
              std::copy(tr.tokens.begin(), tr.tokens.end(), tokens[r].data_ptr<int64_t>());
              for (int j = 0; j < K; ++j) active[r][j] = tr.active[j] != 0;
              for (int j = 0; j < K * sp::kTokens; ++j) masks[r].data_ptr<bool>()[j] = tr.masks[j] != 0;
              std::copy(tr.prefix.begin(), tr.prefix.end(), prefix[r].data_ptr<float>());
              for (size_t i = 0; i < tr.split_sizes.size(); ++i) { split_sizes[r][i] = tr.split_sizes[i]; split_mask[r][i] = true; for (int j = 0; j < 4; ++j) split_feats[r][i][j] = tr.split_feats[4 * i + j]; }
              split_legal[r] = tr.split_legal;
              old[r] = tr.log_prob;
              scored_t[r] = static_cast<bool>(scored[r]);
              const Chunk& c = *batch[rows[r]];
              const int ti = first_turn[rows[r]] + t;
              adv[r] = scored[r] ? c.advantage[ti - c.start] : 0.0f;
            }
            // GPU: pre-write, update, materialise, encode
            torch::Tensor rounds_d = rounds.to(master_device), xy_d = xy.to(master_device), head_d = head_xy.to(master_device);
            torch::Tensor raw_rows = raw.index_select(0, rows_t);
            torch::Tensor pre = Materialize(raw_rows, rounds_d);
            torch::Tensor visible_prev = GatherCells(pre, xy_d);
            torch::Tensor last = GatherCells(raw_rows.select(1, R_LAST_SEEN).unsqueeze(1), xy_d).squeeze(-1);
            torch::Tensor visible_age = torch::where(last >= 0, torch::clamp(rounds_d.to(torch::kFloat).unsqueeze(1) - last, 0, 500) / 500.0, torch::zeros_like(last));
            ApplyUpdates(raw, updates);
            torch::Tensor facts = Materialize(raw.index_select(0, rows_t), rounds_d);
            EncoderBatch x;
            x.local = CropWindows(facts, head_d); x.facts = facts;
            x.board_valid = torch::ones({R, H, W}, torch::TensorOptions().dtype(torch::kBool).device(master_device));
            x.visible_xy = xy_d; x.visible_prev = visible_prev; x.visible_age = visible_age;
            x.status = status.to(master_device); x.history = history.to(master_device); x.history_mask = history_mask.to(master_device);
            x.entities = entities.to(master_device); x.entity_mask = entity_mask.to(master_device);
            x.portal_pairs = pairs.to(master_device); x.pair_mask = pair_mask.to(master_device); x.portal_axis = axis.to(master_device);
            x.portal_unpaired = loose.to(master_device); x.unpaired_mask = loose_mask.to(master_device);
            x.inbox = inbox.to(master_device); x.inbox_mask = inbox_mask.to(master_device); x.head_xy = head_d;
            torch::Tensor scored_d = scored_t.to(master_device);
            const bool any_scored = scored_t.any().item<bool>();
            torch::Tensor h_rows = h.index_select(0, rows_t), M_rows = M.index_select(0, rows_t);
            torch::Tensor h_new, M_new, fused;
            if (any_scored) {
              std::tie(h_new, M_new, fused) = master.Encode(x, M_rows, h_rows);
            } else {
              torch::NoGradGuard ng;
              std::tie(h_new, M_new, fused) = master.Encode(x, M_rows, h_rows);
            }
            // burn-in rows are detached: no gradient into the scored unroll from them
            torch::Tensor keep = scored_d.to(torch::kFloat);
            torch::Tensor h_keep = h_new * keep.unsqueeze(1) + h_new.detach() * (1 - keep).unsqueeze(1);
            torch::Tensor M_keep = M_new * keep.view({R, 1, 1, 1}) + M_new.detach() * (1 - keep).view({R, 1, 1, 1});
            h = h.index_copy(0, rows_t, h_keep);
            M = M.index_copy(0, rows_t, M_keep);
            if (any_scored) {
              torch::Tensor sel = torch::nonzero(scored_d).squeeze(1);
              sc.top = top.to(master_device).index_select(0, sel); sc.tokens = tokens.to(master_device).index_select(0, sel);
              sc.active = active.to(master_device).index_select(0, sel); sc.masks = masks.to(master_device).index_select(0, sel);
              sc.prefix = prefix.to(master_device).index_select(0, sel); sc.split_index = split_index.to(master_device).index_select(0, sel);
              sc.sonar = sonar.to(master_device).index_select(0, sel); sc.split_feats = split_feats.to(master_device).index_select(0, sel);
              sc.split_mask = split_mask.to(master_device).index_select(0, sel); sc.split_sizes = split_sizes.to(master_device).index_select(0, sel);
              sc.split_legal = split_legal.to(master_device).index_select(0, sel);
              auto [lp, ent, sent] = master.Evaluate(h_new.index_select(0, sel), sc);
              logps.push_back(lp); ents.push_back(ent); sents.push_back(sent);
              advs.push_back(adv.to(master_device).index_select(0, sel)); olds.push_back(old.to(master_device).index_select(0, sel));
            }
          }
          if (logps.empty()) continue;
          torch::Tensor lp = torch::cat(logps), ent = torch::cat(ents), sent = torch::cat(sents);
          torch::Tensor advantage = torch::cat(advs), old_lp = torch::cat(olds);
          if (replay_parity < 0) replay_parity = (lp.detach() - old_lp).abs().max().item<double>();
          torch::Tensor ratio = torch::exp(lp - old_lp);
          torch::Tensor surrogate = torch::min(ratio * advantage, torch::clamp(ratio, 1 - o.clip, 1 + o.clip) * advantage);
          torch::Tensor policy_loss = -surrogate.sum() / denominator;
          torch::Tensor entropy_term = (o.entropy_coef * ent.sum() + o.sonar_entropy_coef * sent.sum()) / denominator;
          // value loss over a sample of unique team-events of this microbatch's arenas
          std::vector<const EventRecord*> evs;
          std::vector<float> targets;
          std::vector<int64_t> teams;
          for (const Chunk* c : batch) {
            const ArenaRecord& rec = *c->arena;
            size_t wi = 0;
            for (; wi < workers.size(); ++wi) { bool found = false; for (const auto& r : workers[wi].records) if (&r == &rec) { found = true; break; } if (found) break; }
            size_t ai = 0;
            for (; ai < workers[wi].records.size(); ++ai) if (&workers[wi].records[ai] == &rec) break;
            for (int t = c->start; t < c->stop; ++t) {
              const int e = c->lifetime->turns[t].event;
              for (int team = 0; team < 2; ++team) { evs.push_back(&rec.events[e]); targets.push_back(returns[wi][ai][e][team]); teams.push_back(team); }
            }
          }
          torch::Tensor value_loss = torch::zeros({}, fopt);
          if (!evs.empty()) {
            const int64_t E = std::min<int64_t>(static_cast<int64_t>(evs.size()), 512);
            std::vector<int64_t> pick(evs.size());
            std::iota(pick.begin(), pick.end(), 0);
            std::shuffle(pick.begin(), pick.end(), rng);
            pick.resize(E);
            int64_t N = 1;
            for (int64_t i : pick) N = std::max<int64_t>(N, evs[i]->dragons.size(0));
            torch::Tensor grid = torch::zeros({E, 20, H, W}, torch::kFloat), dragons = torch::zeros({E, N, 16}, torch::kFloat);
            torch::Tensor team = torch::zeros({E, N}, torch::kLong), mask = torch::zeros({E, N}, torch::kBool);
            torch::Tensor tfeat = torch::zeros({E, 2, 4}, torch::kFloat), phase = torch::zeros({E, 3}, torch::kFloat);
            torch::Tensor target = torch::zeros({E}, torch::kFloat), which = torch::zeros({E}, torch::kLong);
            for (int64_t j = 0; j < E; ++j) {
              const EventRecord& ev = *evs[pick[j]];
              grid[j] = ev.map_grid.to(torch::kFloat);
              const int64_t n = ev.dragons.size(0);
              dragons[j].slice(0, 0, n) = ev.dragons; team[j].slice(0, 0, n) = ev.dragon_team; mask[j].slice(0, 0, n) = true;
              tfeat[j] = ev.team_feats; phase[j] = ev.phase; target[j] = targets[pick[j]]; which[j] = teams[pick[j]];
            }
            torch::Tensor v = master.CriticValues(grid.to(master_device), dragons.to(master_device), team.to(master_device), mask.to(master_device), tfeat.to(master_device), phase.to(master_device));
            torch::Tensor vt = v.gather(1, which.to(master_device).unsqueeze(1)).squeeze(1);
            value_loss = ((vt - target.to(master_device)).pow(2)).mean();
          }
          torch::Tensor loss = policy_loss - entropy_term + o.value_coef * value_loss;
          optimizer.zero_grad();
          loss.backward();
          torch::nn::utils::clip_grad_norm_(params, o.grad_norm);
          optimizer.step();
          {
            torch::NoGradGuard ng;
            epoch_kl += ((ratio - 1) - torch::log(ratio)).sum().item<double>();
            epoch_clip += ((ratio - 1).abs() > o.clip).to(torch::kFloat).sum().item<double>();
            policy_loss_sum += policy_loss.item<double>();
            value_loss_sum += value_loss.item<double>();
            entropy_sum += ent.sum().item<double>();
          }
        }
      }
      epochs_done = epoch + 1;
      kl_sum = epoch_kl / std::max<int64_t>(scored_total, 1);
      clip_sum = epoch_clip / std::max<int64_t>(scored_total, 1);
      if (kl_sum > o.target_kl) break;
    }
    // ---- sync replicas ---------------------------------------------------
    for (Worker& w : workers) {
      w.model->CopyActorFrom(master);
      torch::NoGradGuard ng;
      auto mine = w.model->CriticParameters();
      auto theirs = master.CriticParameters();
      for (size_t i = 0; i < mine.size(); ++i) mine[i].copy_(theirs[i].to(mine[i].device()));
    }
    const double learn_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    std::printf("iter %d: %lld decisions, collect %.2fs (%.0f/s), learn %.2fs (%lld scored, %zu chunks, %.0f team-rounds), "
                "kl %.4f clip %.3f policy %.4f value %.4f entropy %.3f epochs %d replay-parity %.2e\n",
                iteration, static_cast<long long>(decisions), collect_s, decisions / collect_s, learn_s,
                static_cast<long long>(scored_total), chunks.size(), denominator, kl_sum, clip_sum, policy_loss_sum,
                value_loss_sum, entropy_sum / std::max<int64_t>(scored_total, 1), epochs_done, replay_parity);
    std::fflush(stdout);
    if (iteration % o.save_every == 0) master.Save(o.out);
  }
  return 0;
}
