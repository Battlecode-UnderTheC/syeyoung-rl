// Native PPO trainer: rollout, GAE and the update all in C++.
//
// Shape of an iteration:
//   * every worker thread runs its own VecGame for `segment` rounds, acting
//     with a per-device replica of the policy and recording whole per-dragon
//     sequences from the first `record_arenas` of its arenas;
//   * sequences are turned into advantages with GAE, bootstrapped by the value
//     head where the segment cut a match short;
//   * the master module takes K PPO epochs over those sequences, replaying each
//     one from its stored start state so the GRU is trained through time;
//   * the updated parameters are copied back to every replica.
//
// Recording only a subset of arenas is deliberate: at a million arenas the
// observations alone would be tens of gigabytes per iteration, while the
// unrecorded arenas still do useful work keeping the GPU batches large.
//
// Phase 2 note: with one dragon per team and splitting off, a dragon dying *is*
// the match ending, so the terminal reward lands on its final step. That stops
// being true once splitting is enabled and the team reward will have to be
// shared across a team's dragons the way algo/gae.py does it.

#include <ATen/Parallel.h>
#include <c10/core/DeviceGuard.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <memory>
#include <torch/cuda.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "agent/encoder.hpp"

namespace {

using dragons::kChannels;
using dragons::kScalars;
using dragons::kTiles;

struct Options {
  std::string model = "native/actor.ts";
  std::string maps = "native/maps.txt";
  std::string space = "native/space.txt";
  std::string out = "runs/native";
  int threads = 28;
  int arenas_per_thread = 32768;
  int record_arenas = 2048;      // per thread
  int segment = 8;               // rounds per iteration
  int iterations = 100000;
  int gpus = 4;
  int max_slots = 8;
  int epochs = 1;
  int sequences_per_batch = 4096;
  float learning_rate = 3e-4f;
  float clip = 0.2f;
  float value_coef = 0.5f;
  float entropy_coef = 0.01f;
  float grad_clip = 0.5f;
  float gamma = 0.997f;
  float lambda = 0.95f;
  int save_every = 50;
  // The last `eval_arenas` of each thread pit the live policy (team A) against
  // a frozen snapshot (team B). Self-play win rate against *itself* is 0.5 by
  // construction and says nothing; beating an older self is the actual signal.
  int eval_arenas = 512;
  int freeze_every = 20;
  // Potential-based shaping on the dragon's own length (spec section 2):
  //   r += shaping * (gamma * L(next) - L(now)),  L = 0 once it is dead.
  // That is +shaping for every pearl and -shaping*L for dying, which is the
  // immediate signal terminal-only reward cannot give. The terminal potential
  // is zero, so the optimal policy is unchanged -- only the credit path is.
  float shaping = 0.05f;
};

int Arg(int argc, char** argv, const std::string& flag, int fallback) {
  for (int i = 1; i + 1 < argc; ++i) if (flag == argv[i]) return std::atoi(argv[i + 1]);
  return fallback;
}
float ArgF(int argc, char** argv, const std::string& flag, float fallback) {
  for (int i = 1; i + 1 < argc; ++i) if (flag == argv[i]) return std::atof(argv[i + 1]);
  return fallback;
}
std::string ArgS(int argc, char** argv, const std::string& flag, const std::string& fb) {
  for (int i = 1; i + 1 < argc; ++i) if (flag == argv[i]) return argv[i + 1];
  return fb;
}

std::vector<std::string> ReadMaps(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot read maps: " + path);
  std::vector<std::string> maps;
  std::string line, current;
  while (std::getline(input, line)) {
    if (line == "---") { if (!current.empty()) maps.push_back(current); current.clear(); }
    else { current += line; current += '\n'; }
  }
  if (!current.empty()) maps.push_back(current);
  return maps;
}

struct SpaceConfig {
  int max_sprint = 3;
  std::vector<int> split_sizes;
  int hidden = 256;
  int actions = 0;
};

SpaceConfig ReadSpace(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot read space: " + path);
  SpaceConfig config;
  std::string key;
  while (input >> key) {
    if (key == "max_sprint") input >> config.max_sprint;
    else if (key == "hidden") input >> config.hidden;
    else if (key == "actions") input >> config.actions;
    else if (key == "split_sizes") {
      std::string rest, number; std::getline(input, rest);
      for (char c : rest) {
        if (c == ',' || c == ' ') { if (!number.empty()) { config.split_sizes.push_back(std::stoi(number)); number.clear(); } }
        else number += c;
      }
      if (!number.empty()) config.split_sizes.push_back(std::stoi(number));
    }
  }
  return config;
}

// One recorded dragon-turn. Observations are quantised to bytes: every channel
// is a flag except the pearl timer, which loses nothing that matters at 1/255.
struct Step {
  std::vector<uint8_t> grid;
  std::vector<float> scalars;
  std::vector<uint8_t> mask;
  int32_t action = 0;
  float log_prob = 0.0f;
  float value = 0.0f;
  float reward = 0.0f;
  int32_t length = 0;          // for the shaping potential
  uint8_t done = 0;
};

struct Sequence {
  std::vector<float> start_hidden;
  std::vector<Step> steps;
  // Length after the final recorded step: 0 if the dragon died, its surviving
  // length if the match ended around it, -1 while the segment is still open
  // (then the last step simply gets no shaping).
  int32_t tail_length = -1;
};

struct ThreadResult {
  std::vector<Sequence> sequences;
  int64_t turns = 0;
  int64_t finished_matches = 0;
  int64_t wins_a = 0, wins_b = 0, draws = 0;
  int64_t eval_matches = 0, eval_wins = 0, eval_draws = 0;
  double total_length = 0.0;
  int64_t length_samples = 0;
  std::array<int64_t, 5> deaths{};        // W S O H A
};

int DeathSlot(char reason) {
  switch (reason) {
    case 'W': return 0; case 'S': return 1; case 'O': return 2;
    case 'H': return 3; default: return 4;
  }
}

}  // namespace

int main(int argc, char** argv) {
  using namespace dragons;
  Options options;
  options.model = ArgS(argc, argv, "--model", options.model);
  options.maps = ArgS(argc, argv, "--maps", options.maps);
  options.space = ArgS(argc, argv, "--space", options.space);
  options.out = ArgS(argc, argv, "--out", options.out);
  options.threads = Arg(argc, argv, "--threads", options.threads);
  options.arenas_per_thread = Arg(argc, argv, "--arenas-per-thread", options.arenas_per_thread);
  options.record_arenas = Arg(argc, argv, "--record-arenas", options.record_arenas);
  options.segment = Arg(argc, argv, "--segment", options.segment);
  options.iterations = Arg(argc, argv, "--iterations", options.iterations);
  options.gpus = Arg(argc, argv, "--gpus", options.gpus);
  options.max_slots = Arg(argc, argv, "--max-slots", options.max_slots);
  options.epochs = Arg(argc, argv, "--epochs", options.epochs);
  options.sequences_per_batch = Arg(argc, argv, "--sequences-per-batch", options.sequences_per_batch);
  options.learning_rate = ArgF(argc, argv, "--lr", options.learning_rate);
  options.entropy_coef = ArgF(argc, argv, "--entropy", options.entropy_coef);
  options.gamma = ArgF(argc, argv, "--gamma", options.gamma);
  options.save_every = Arg(argc, argv, "--save-every", options.save_every);
  options.eval_arenas = Arg(argc, argv, "--eval-arenas", options.eval_arenas);
  options.freeze_every = Arg(argc, argv, "--freeze-every", options.freeze_every);
  options.shaping = ArgF(argc, argv, "--shaping", options.shaping);

  const SpaceConfig config = ReadSpace(options.space);
  const std::vector<std::string> pool = ReadMaps(options.maps);
  if (pool.empty()) throw std::runtime_error("no maps");
  at::set_num_threads(1);

  const bool cuda = torch::cuda::is_available() && options.gpus > 0;
  const torch::Device master_device(cuda ? torch::kCUDA : torch::kCPU, 0);
  torch::jit::script::Module master = torch::jit::load(options.model, master_device);
  master.train();
  std::vector<torch::Tensor> parameters;
  for (const auto& parameter : master.parameters()) parameters.push_back(parameter);
  torch::optim::Adam optimizer(parameters, torch::optim::AdamOptions(options.learning_rate));

  // one inference replica per device; threads share the replica of their device
  std::vector<torch::jit::script::Module> replicas;
  std::vector<torch::jit::script::Module> learners;   // data-parallel shards
  std::vector<torch::jit::script::Module> frozen;
  std::vector<torch::Device> devices;
  for (int g = 0; g < std::max(options.gpus, 1); ++g) {
    devices.emplace_back(cuda ? torch::Device(torch::kCUDA, g) : torch::Device(torch::kCPU));
    replicas.push_back(torch::jit::load(options.model, devices.back()));
    replicas.back().eval();
    learners.push_back(torch::jit::load(options.model, devices.back()));
    learners.back().train();
    frozen.push_back(torch::jit::load(options.model, devices.back()));
    frozen.back().eval();
  }

  // parameter handles, materialised once: the jit iterators are not
  // random-access so they cannot be indexed or advanced cheaply
  std::vector<std::vector<torch::Tensor>> shard_parameters(learners.size());
  for (size_t shard = 0; shard < learners.size(); ++shard) {
    for (const auto& parameter : learners[shard].parameters()) {
      shard_parameters[shard].push_back(parameter);
    }
  }

  // the sims live for the whole run, so matches continue across iterations
  ActionSpace space(config.max_sprint, config.split_sizes, true);
  std::vector<std::unique_ptr<VecGame>> sims;
  std::vector<std::vector<std::string>> thread_maps(options.threads);
  for (int t = 0; t < options.threads; ++t) {
    thread_maps[t].resize(options.arenas_per_thread);
    for (int i = 0; i < options.arenas_per_thread; ++i) {
      thread_maps[t][i] = pool[(static_cast<size_t>(t) * options.arenas_per_thread + i) % pool.size()];
    }
    sims.push_back(std::make_unique<VecGame>(thread_maps[t],
                                             static_cast<uint64_t>(t) * 7919, space, true));
  }
  const int hidden_size = config.hidden;
  const int n_actions = space.size();
  std::vector<torch::Tensor> thread_hidden;
  for (int t = 0; t < options.threads; ++t) {
    thread_hidden.push_back(torch::zeros(
        {options.arenas_per_thread * options.max_slots, hidden_size},
        torch::TensorOptions().device(devices[t % devices.size()])));
  }

  std::filesystem::create_directories(options.out);
  std::ofstream metrics(options.out + "/metrics.jsonl", std::ios::app);

  auto rollout = [&](int t, ThreadResult* result) {
    torch::NoGradGuard no_grad;
    VecGame& vec = *sims[t];
    const torch::Device device = devices[t % devices.size()];
    torch::jit::script::Module& policy = replicas[t % replicas.size()];
    torch::Tensor& hidden = thread_hidden[t];
    const int arenas = options.arenas_per_thread;

    auto cpu = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor grid = torch::empty({arenas, kChannels, kVision, kVision}, cpu);
    torch::Tensor scalars = torch::empty({arenas, kScalars}, cpu);
    torch::Tensor mask = torch::empty({arenas, n_actions}, cpu.dtype(torch::kBool));
    torch::Tensor rows_index = torch::empty({arenas}, cpu.dtype(torch::kLong));
    if (cuda) { grid = grid.pin_memory(); scalars = scalars.pin_memory(); mask = mask.pin_memory(); }
    auto on_device = torch::TensorOptions().device(device);
    torch::Tensor messages = torch::zeros({arenas, 8, 4}, on_device.dtype(torch::kLong));
    torch::Tensor message_mask = torch::zeros({arenas, 8}, on_device.dtype(torch::kBool));
    std::vector<int32_t> arena_of(arenas), dragon_of(arenas), chosen(arenas);

    // sequence per recorded (arena, slot), reset each iteration
    const int recorded = std::min(options.record_arenas, arenas);
    std::vector<int> sequence_of(static_cast<size_t>(recorded) * options.max_slots, -1);
    torch::jit::script::Module& opponent = frozen[t % frozen.size()];
    const int eval_from = std::max(arenas - options.eval_arenas, recorded);
    std::vector<int64_t> live_rows, frozen_rows;

    for (int round = 0; round < options.segment; ++round) {
      vec.BeginRound();
      for (int slot = 0;; ++slot) {
        const int rows = vec.PrepareSlot(slot, grid.data_ptr<float>(), scalars.data_ptr<float>(),
                                         mask.data_ptr<bool>(), arena_of.data(), dragon_of.data());
        if (rows == 0) break;
        auto index = rows_index.accessor<int64_t, 1>();
        for (int row = 0; row < rows; ++row) {
          index[row] = static_cast<int64_t>(arena_of[row]) * options.max_slots + slot;
        }
        torch::Tensor rows_on_device = rows_index.narrow(0, 0, rows).to(device, true);
        torch::Tensor state = hidden.index_select(0, rows_on_device);
        torch::Tensor grid_d = grid.narrow(0, 0, rows).to(device, true);
        torch::Tensor scalars_d = scalars.narrow(0, 0, rows).to(device, true);
        torch::Tensor mask_d = mask.narrow(0, 0, rows).to(device, true);

        // In an eval arena, team B is played by the frozen snapshot.
        live_rows.clear();
        frozen_rows.clear();
        for (int row = 0; row < rows; ++row) {
          const Game& game = vec.game(arena_of[row]);
          const bool is_b = slot < static_cast<int>(game.dragons().size()) &&
                            game.dragons()[slot].team == 1;
          if (arena_of[row] >= eval_from && is_b) frozen_rows.push_back(row);
          else live_rows.push_back(row);
        }

        auto run = [&](torch::jit::script::Module& module, const std::vector<int64_t>& which,
                       torch::Tensor& logits_out, torch::Tensor& state_out,
                       torch::Tensor& values_out) {
          torch::Tensor pick = torch::from_blob(
              const_cast<int64_t*>(which.data()), {static_cast<int64_t>(which.size())},
              torch::TensorOptions().dtype(torch::kLong)).to(device);
          std::vector<torch::jit::IValue> inputs{
              grid_d.index_select(0, pick), scalars_d.index_select(0, pick),
              messages.narrow(0, 0, static_cast<int64_t>(which.size())),
              message_mask.narrow(0, 0, static_cast<int64_t>(which.size())),
              mask_d.index_select(0, pick), state.index_select(0, pick)};
          auto out = module.forward(inputs).toTuple();
          logits_out = out->elements()[0].toTensor();
          state_out = out->elements()[3].toTensor();
          values_out = out->elements()[4].toTensor();
          return pick;
        };

        torch::Tensor logits = torch::zeros({rows, n_actions},
            torch::TensorOptions().device(device));
        torch::Tensor new_state = state.clone();
        torch::Tensor values = torch::zeros({rows}, torch::TensorOptions().device(device));
        if (!live_rows.empty()) {
          torch::Tensor l, s2, v;
          torch::Tensor pick = run(policy, live_rows, l, s2, v);
          logits.index_copy_(0, pick, l);
          new_state.index_copy_(0, pick, s2);
          values.index_copy_(0, pick, v);
        }
        if (!frozen_rows.empty()) {
          torch::Tensor l, s2, v;
          torch::Tensor pick = run(opponent, frozen_rows, l, s2, v);
          logits.index_copy_(0, pick, l);
          new_state.index_copy_(0, pick, s2);
          values.index_copy_(0, pick, v);
        }
        hidden.index_copy_(0, rows_on_device, new_state);

        torch::Tensor noise = torch::rand_like(logits).clamp_min(1e-20);
        torch::Tensor actions = (logits - torch::log(-torch::log(noise))).argmax(-1);
        torch::Tensor log_probs =
            torch::log_softmax(logits, -1).gather(-1, actions.unsqueeze(-1)).squeeze(-1);

        torch::Tensor actions_cpu = actions.to(torch::kInt32).to(torch::kCPU);
        torch::Tensor logp_cpu = log_probs.to(torch::kCPU);
        torch::Tensor value_cpu = values.to(torch::kCPU);
        torch::Tensor state_cpu = state.to(torch::kCPU);

        const int32_t* picked = actions_cpu.data_ptr<int32_t>();
        const float* logp = logp_cpu.data_ptr<float>();
        const float* value = value_cpu.data_ptr<float>();
        const float* start_state = state_cpu.data_ptr<float>();
        const float* grid_data = grid.data_ptr<float>();
        const float* scalar_data = scalars.data_ptr<float>();
        const bool* mask_data = mask.data_ptr<bool>();

        for (int row = 0; row < rows; ++row) {
          chosen[row] = picked[row];
          if (arena_of[row] >= recorded) continue;
          const size_t key = static_cast<size_t>(arena_of[row]) * options.max_slots + slot;
          if (sequence_of[key] < 0) {
            sequence_of[key] = static_cast<int>(result->sequences.size());
            Sequence sequence;
            sequence.start_hidden.assign(start_state + static_cast<size_t>(row) * hidden_size,
                                         start_state + static_cast<size_t>(row + 1) * hidden_size);
            result->sequences.push_back(std::move(sequence));
          }
          Step step;
          step.grid.resize(kChannels * kTiles);
          const float* row_grid = grid_data + static_cast<size_t>(row) * kChannels * kTiles;
          for (int i = 0; i < kChannels * kTiles; ++i) {
            step.grid[i] = static_cast<uint8_t>(std::lround(std::clamp(row_grid[i], 0.0f, 1.0f) * 255.0f));
          }
          step.scalars.assign(scalar_data + static_cast<size_t>(row) * kScalars,
                              scalar_data + static_cast<size_t>(row + 1) * kScalars);
          step.mask.resize(n_actions);
          const bool* row_mask = mask_data + static_cast<size_t>(row) * n_actions;
          for (int a = 0; a < n_actions; ++a) step.mask[a] = row_mask[a] ? 1 : 0;
          step.action = picked[row];
          step.log_prob = logp[row];
          step.value = value[row];
          step.length = vec.game(arena_of[row]).dragons()[slot].length();
          result->sequences[sequence_of[key]].steps.push_back(std::move(step));
        }
        vec.SubmitSlot(chosen.data(), nullptr, nullptr);
        result->turns += rows;
      }
      vec.EndRound();

      // settle finished matches: terminal reward on the last recorded step
      for (int a = 0; a < arenas; ++a) {
        Game& game = vec.game(a);
        if (!game.Finished()) continue;
        const Outcome outcome = game.Result();
        if (a < recorded) {
          for (int slot = 0; slot < options.max_slots; ++slot) {
            const size_t key = static_cast<size_t>(a) * options.max_slots + slot;
            const int sequence = sequence_of[key];
            if (sequence < 0 || result->sequences[sequence].steps.empty()) continue;
            const int team = slot < static_cast<int>(game.dragons().size())
                                 ? game.dragons()[slot].team : 0;
            float reward = 0.0f;
            if (outcome == Outcome::TeamA) reward = team == 0 ? 1.0f : -1.0f;
            else if (outcome == Outcome::TeamB) reward = team == 1 ? 1.0f : -1.0f;
            Step& last = result->sequences[sequence].steps.back();
            last.reward = reward;
            last.done = 1;
            const bool survived = slot < static_cast<int>(game.dragons().size()) &&
                                  game.dragons()[slot].alive;
            result->sequences[sequence].tail_length =
                survived ? game.dragons()[slot].length() : 0;
            sequence_of[key] = -1;      // a new match starts a new sequence
          }
        }
        for (const DeathEvent& death : game.deaths()) {
          result->deaths[DeathSlot(static_cast<char>(death.reason))]++;
        }
        result->finished_matches++;
        if (outcome == Outcome::TeamA) result->wins_a++;
        else if (outcome == Outcome::TeamB) result->wins_b++;
        else result->draws++;
        if (a >= eval_from) {           // live policy is team A here
          result->eval_matches++;
          if (outcome == Outcome::TeamA) result->eval_wins++;
          else if (outcome == Outcome::Draw) result->eval_draws++;
        }
        for (int team = 0; team < 2; ++team) {
          result->total_length += game.TeamScore(static_cast<uint8_t>(team)).max_length;
          result->length_samples++;
        }
        // restart the arena so the sim never idles
        game = Game::FromMapText(thread_maps[t][a], static_cast<uint64_t>(a) * 31 + result->finished_matches);
        hidden.index_fill_(0, torch::arange(static_cast<int64_t>(a) * options.max_slots,
                                            static_cast<int64_t>(a + 1) * options.max_slots,
                                            torch::TensorOptions().dtype(torch::kLong).device(hidden.device())),
                           0);
      }
    }
  };

  std::mt19937 shuffle_rng(1234);

  // Rollout and update overlap: while the learner works on the segment that
  // just finished, the workers are already collecting the next one. The
  // workers therefore act with weights one iteration behind, which PPO's
  // importance ratio is there to account for.
  //
  // The inference replicas are refreshed at the one moment no rollout is in
  // flight -- right after the join -- so nothing reads a tensor while it is
  // being written.
  std::array<std::vector<ThreadResult>, 2> buffers;
  std::vector<std::thread> in_flight;
  int slot_index = 0;
  auto launch = [&](int which) {
    buffers[which].assign(options.threads, ThreadResult{});
    in_flight.clear();
    for (int t = 0; t < options.threads; ++t) {
      in_flight.emplace_back(rollout, t, &buffers[which][t]);
    }
  };
  launch(slot_index);

  for (int iteration = 0; iteration < options.iterations; ++iteration) {
    const auto started = std::chrono::steady_clock::now();
    for (auto& worker : in_flight) worker.join();
    const double rollout_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

    // safe point: no rollout running, so the replicas can take the new weights
    {
      torch::NoGradGuard no_grad;
      for (size_t r = 0; r < replicas.size(); ++r) {
        auto target_it = replicas[r].parameters().begin();
        for (const auto& source : master.parameters()) {
          (*target_it).copy_(source.to((*target_it).device()));
          ++target_it;
        }
      }
    }
    std::vector<ThreadResult>& results = buffers[slot_index];
    const int next_slot = 1 - slot_index;
    launch(next_slot);
    slot_index = next_slot;

    // flatten sequences and compute GAE within each
    std::vector<Sequence*> sequences;
    int64_t turns = 0, matches = 0, wins_a = 0, wins_b = 0, draws = 0, samples = 0;
    int64_t eval_matches = 0, eval_wins = 0, eval_draws = 0;
    double reward_total = 0.0; int64_t reward_steps = 0;
    double length_total = 0.0;
    std::array<int64_t, 5> deaths{};
    for (ThreadResult& result : results) {
      turns += result.turns; matches += result.finished_matches;
      wins_a += result.wins_a; wins_b += result.wins_b; draws += result.draws;
      length_total += result.total_length; samples += result.length_samples;
      eval_matches += result.eval_matches; eval_wins += result.eval_wins;
      eval_draws += result.eval_draws;
      for (int i = 0; i < 5; ++i) deaths[i] += result.deaths[i];
      for (Sequence& sequence : result.sequences) {
        if (!sequence.steps.empty()) sequences.push_back(&sequence);
      }
    }

    std::vector<std::vector<float>> advantages(sequences.size());
    std::vector<std::vector<float>> returns(sequences.size());
    for (size_t s = 0; s < sequences.size(); ++s) {
      const std::vector<Step>& steps = sequences[s]->steps;
      advantages[s].assign(steps.size(), 0.0f);
      returns[s].assign(steps.size(), 0.0f);

      // shaped reward per step, on top of the terminal win/lose
      std::vector<float> reward(steps.size());
      for (size_t i = 0; i < steps.size(); ++i) {
        reward[i] = steps[i].reward;
        if (options.shaping == 0.0f) continue;
        float next_length;
        if (i + 1 < steps.size()) {
          next_length = static_cast<float>(steps[i + 1].length);
        } else if (sequences[s]->tail_length >= 0) {
          next_length = static_cast<float>(sequences[s]->tail_length);
        } else {
          continue;                    // segment still open; no shaping here
        }
        reward[i] += options.shaping *
                     (options.gamma * next_length - static_cast<float>(steps[i].length));
      }

      float running = 0.0f;
      // the tail of a cut-short segment is bootstrapped by the value head
      float next_value = steps.back().done ? 0.0f : steps.back().value;
      for (int i = static_cast<int>(steps.size()) - 1; i >= 0; --i) {
        const float mask_done = steps[i].done ? 0.0f : 1.0f;
        const float delta = reward[i] + options.gamma * next_value * mask_done - steps[i].value;
        running = delta + options.gamma * options.lambda * mask_done * running;
        advantages[s][i] = running;
        returns[s][i] = running + steps[i].value;
        next_value = steps[i].value;
      }
    }

    for (size_t s = 0; s < sequences.size(); ++s) {
      for (const Step& step : sequences[s]->steps) {
        reward_total += step.reward;
        ++reward_steps;
      }
    }

    // normalise advantages over all recorded steps
    double sum = 0.0, sum_sq = 0.0; int64_t count = 0;
    for (const auto& row : advantages) for (float a : row) { sum += a; sum_sq += a * a; ++count; }
    const float mean = count ? static_cast<float>(sum / count) : 0.0f;
    const float variance = count ? static_cast<float>(sum_sq / count - mean * mean) : 1.0f;
    const float inv_std = 1.0f / (std::sqrt(std::max(variance, 0.0f)) + 1e-8f);

    const auto learn_started = std::chrono::steady_clock::now();
    std::vector<size_t> order(sequences.size());
    std::iota(order.begin(), order.end(), 0);
    double last_policy_loss = 0.0, last_value_loss = 0.0, last_entropy = 0.0, last_kl = 0.0;
    double assemble_seconds = 0.0, gpu_seconds = 0.0;
    int updates = 0;

    for (int epoch = 0; epoch < options.epochs; ++epoch) {
      std::shuffle(order.begin(), order.end(), shuffle_rng);
      for (size_t start = 0; start < order.size(); start += options.sequences_per_batch) {
        const size_t stop = std::min(start + static_cast<size_t>(options.sequences_per_batch), order.size());
        const int batch = static_cast<int>(stop - start);
        if (batch == 0) continue;
        int longest = 0;
        for (size_t i = start; i < stop; ++i) {
          longest = std::max<int>(longest, sequences[order[i]]->steps.size());
        }

        const auto assemble_started = std::chrono::steady_clock::now();
        auto cpu_float = torch::TensorOptions().dtype(torch::kFloat32);
        auto cpu_byte = torch::TensorOptions().dtype(torch::kUInt8);
        // Observations stay bytes until they are on the device: that is a
        // quarter of the PCIe traffic and no per-scalar conversion on the CPU.
        torch::Tensor grid_bytes = torch::zeros({batch, longest, kChannels * kTiles}, cpu_byte);
        torch::Tensor mask_bytes = torch::zeros({batch, longest, n_actions}, cpu_byte);
        torch::Tensor scalars = torch::zeros({batch, longest, kScalars}, cpu_float);
        torch::Tensor actions = torch::zeros({batch, longest}, cpu_float.dtype(torch::kLong));
        torch::Tensor old_logp = torch::zeros({batch, longest}, cpu_float);
        torch::Tensor target = torch::zeros({batch, longest}, cpu_float);
        torch::Tensor advantage = torch::zeros({batch, longest}, cpu_float);
        torch::Tensor valid = torch::zeros({batch, longest}, cpu_float);
        torch::Tensor start_hidden = torch::zeros({batch, hidden_size}, cpu_float);

        uint8_t* grid_ptr = grid_bytes.data_ptr<uint8_t>();
        uint8_t* mask_ptr = mask_bytes.data_ptr<uint8_t>();
        float* scalar_ptr = scalars.data_ptr<float>();
        int64_t* action_ptr = actions.data_ptr<int64_t>();
        float* logp_ptr = old_logp.data_ptr<float>();
        float* target_ptr = target.data_ptr<float>();
        float* adv_ptr = advantage.data_ptr<float>();
        float* valid_ptr = valid.data_ptr<float>();
        float* hidden_ptr = start_hidden.data_ptr<float>();

        // Filling a few hundred megabytes single-threaded was costing more than
        // the backward pass, so the rows are split across the same cores the
        // sim just finished using.
        const int fill_threads = std::min<int>(options.threads, batch);
        std::vector<std::thread> fillers;
        for (int worker = 0; worker < fill_threads; ++worker) {
          fillers.emplace_back([&, worker] {
            for (int row = worker; row < batch; row += fill_threads) {
              const size_t s_index = order[start + row];
              const Sequence& sequence = *sequences[s_index];
              std::memcpy(hidden_ptr + static_cast<size_t>(row) * hidden_size,
                          sequence.start_hidden.data(), sizeof(float) * hidden_size);
              for (size_t step = 0; step < sequence.steps.size(); ++step) {
                const Step& entry = sequence.steps[step];
                const size_t flat = static_cast<size_t>(row) * longest + step;
                std::memcpy(grid_ptr + flat * kChannels * kTiles, entry.grid.data(),
                            kChannels * kTiles);
                std::memcpy(mask_ptr + flat * n_actions, entry.mask.data(), n_actions);
                std::memcpy(scalar_ptr + flat * kScalars, entry.scalars.data(),
                            sizeof(float) * kScalars);
                action_ptr[flat] = entry.action;
                logp_ptr[flat] = entry.log_prob;
                target_ptr[flat] = returns[s_index][step];
                adv_ptr[flat] = (advantages[s_index][step] - mean) * inv_std;
                valid_ptr[flat] = 1.0f;
              }
            }
          });
        }
        for (std::thread& filler : fillers) filler.join();

        assemble_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - assemble_started).count();
        const auto gpu_started = std::chrono::steady_clock::now();
        const int shards = static_cast<int>(learners.size());
        const int shard_rows = (batch + shards - 1) / shards;
        std::vector<double> shard_policy(shards, 0.0), shard_value(shards, 0.0);
        std::vector<double> shard_entropy(shards, 0.0), shard_kl(shards, 0.0);
        std::vector<char> shard_used(shards, 0);

        // Each GPU takes a slice of the minibatch, runs the whole BPTT chain on
        // it and leaves gradients in its own replica; the master then sums them
        // and takes one step, so this is exact data parallelism, not four
        // independent optimisers drifting apart.
        //
        // The shards run on their own CPU threads. Launching them in a plain
        // loop looks parallel because CUDA is asynchronous, but the losses are
        // read back per shard, and each read synchronises -- which serialises
        // the four GPUs and buys nothing.
        std::vector<std::thread> shard_threads;
        for (int shard = 0; shard < shards; ++shard) {
          shard_threads.emplace_back([&, shard] {
          const int from = shard * shard_rows;
          const int rows_here = std::min(shard_rows, batch - from);
          if (rows_here <= 0) return;
          const torch::Device shard_device = devices[shard];
          const c10::DeviceGuard device_guard(shard_device);

          torch::Tensor s_grid = grid_bytes.narrow(0, from, rows_here)
                                     .to(shard_device, true).to(torch::kFloat32)
                                     .div_(255.0f)
                                     .view({rows_here, longest, kChannels, kVision, kVision});
          torch::Tensor s_mask = mask_bytes.narrow(0, from, rows_here)
                                     .to(shard_device, true).to(torch::kBool);
          torch::Tensor s_scalars = scalars.narrow(0, from, rows_here).to(shard_device, true);
          torch::Tensor s_actions = actions.narrow(0, from, rows_here).to(shard_device, true);
          torch::Tensor s_old_logp = old_logp.narrow(0, from, rows_here).to(shard_device, true);
          torch::Tensor s_target = target.narrow(0, from, rows_here).to(shard_device, true);
          torch::Tensor s_adv = advantage.narrow(0, from, rows_here).to(shard_device, true);
          torch::Tensor s_valid = valid.narrow(0, from, rows_here).to(shard_device, true);
          torch::Tensor state = start_hidden.narrow(0, from, rows_here).to(shard_device, true);
          torch::Tensor s_messages = torch::zeros({rows_here, 8, 4},
              torch::TensorOptions().dtype(torch::kLong).device(shard_device));
          torch::Tensor s_message_mask = torch::zeros({rows_here, 8},
              torch::TensorOptions().dtype(torch::kBool).device(shard_device));

          std::vector<torch::Tensor> logp_steps, entropy_steps, value_steps;
          for (int step = 0; step < longest; ++step) {
            std::vector<torch::jit::IValue> inputs{
                s_grid.select(1, step), s_scalars.select(1, step), s_messages,
                s_message_mask, s_mask.select(1, step), state};
            auto out = learners[shard].forward(inputs).toTuple();
            torch::Tensor logits = out->elements()[0].toTensor();
            state = out->elements()[3].toTensor();
            value_steps.push_back(out->elements()[4].toTensor());
            torch::Tensor log_probs = torch::log_softmax(logits, -1);
            logp_steps.push_back(
                log_probs.gather(-1, s_actions.select(1, step).unsqueeze(-1)).squeeze(-1));
            entropy_steps.push_back(-(log_probs.exp() * log_probs).nan_to_num(0.0).sum(-1));
          }
          torch::Tensor new_logp = torch::stack(logp_steps, 1);
          torch::Tensor entropy = torch::stack(entropy_steps, 1);
          torch::Tensor values = torch::stack(value_steps, 1);

          torch::Tensor ratio = torch::exp(new_logp - s_old_logp);
          torch::Tensor unclipped = ratio * s_adv;
          torch::Tensor clipped =
              torch::clamp(ratio, 1.0f - options.clip, 1.0f + options.clip) * s_adv;
          // every shard normalises by the whole minibatch so the summed
          // gradient is the mean over it, not the sum of four means
          const torch::Tensor denominator =
              torch::full({}, static_cast<double>(batch) / rows_here, s_valid.options())
              * s_valid.sum().clamp_min(1.0) * shards;
          torch::Tensor policy_loss = -(torch::min(unclipped, clipped) * s_valid).sum() / denominator;
          torch::Tensor value_loss = (((values - s_target).pow(2)) * s_valid).sum() / denominator;
          torch::Tensor entropy_loss = (entropy * s_valid).sum() / denominator;
          torch::Tensor loss = policy_loss + options.value_coef * value_loss
                               - options.entropy_coef * entropy_loss;
          loss.backward();

          shard_policy[shard] = policy_loss.item<double>();
          shard_value[shard] = value_loss.item<double>();
          shard_entropy[shard] = entropy_loss.item<double>();
          shard_kl[shard] =
              (((ratio - 1.0) - torch::log(ratio)) * s_valid).sum().item<double>()
              / std::max(1.0, s_valid.sum().item<double>());
          shard_used[shard] = 1;
          });
        }
        for (std::thread& shard_thread : shard_threads) shard_thread.join();

        double policy_sum = 0.0, value_sum = 0.0, entropy_sum = 0.0, kl_sum = 0.0;
        int shard_count = 0;
        for (int shard = 0; shard < shards; ++shard) {
          if (!shard_used[shard]) continue;
          policy_sum += shard_policy[shard];
          value_sum += shard_value[shard];
          entropy_sum += shard_entropy[shard];
          kl_sum += shard_kl[shard];
          ++shard_count;
        }

        // sum the shard gradients onto the master and take one step
        for (size_t i = 0; i < parameters.size(); ++i) {
          torch::Tensor total;
          for (int shard = 0; shard < shards; ++shard) {
            const torch::Tensor& grad = shard_parameters[shard][i].grad();
            if (!grad.defined()) continue;
            torch::Tensor contribution = grad.to(master_device);
            total = total.defined() ? total + contribution : contribution;
          }
          parameters[i].mutable_grad() = total;
        }
        torch::nn::utils::clip_grad_norm_(parameters, options.grad_clip);
        optimizer.step();
        optimizer.zero_grad();
        {
          torch::NoGradGuard no_grad;
          for (int shard = 0; shard < shards; ++shard) {
            for (size_t i = 0; i < parameters.size(); ++i) {
              shard_parameters[shard][i].mutable_grad() = torch::Tensor();
              shard_parameters[shard][i].copy_(
                  parameters[i].to(shard_parameters[shard][i].device()));
            }
          }
        }

        gpu_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - gpu_started).count();
        const double denom_shards = std::max(1, shard_count);
        torch::Tensor policy_loss = torch::full({}, policy_sum);
        torch::Tensor value_loss = torch::full({}, value_sum);
        torch::Tensor entropy_loss = torch::full({}, entropy_sum);
        torch::Tensor ratio = torch::full({}, kl_sum / denom_shards);
        torch::Tensor valid_dummy = valid;
        last_policy_loss = policy_sum;
        last_value_loss = value_sum;
        last_entropy = entropy_sum;
        last_kl = kl_sum / denom_shards;
        ++updates;
      }
    }
    const double learn_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - learn_started).count();

    const double seconds = rollout_seconds + learn_seconds;
    const double win_rate = matches ? static_cast<double>(wins_a) / matches : 0.0;
    // the number that actually means something: live policy against a frozen self
    const double eval_win = eval_matches ? static_cast<double>(eval_wins) / eval_matches : 0.0;
    const double eval_draw = eval_matches ? static_cast<double>(eval_draws) / eval_matches : 0.0;
    char line[1024];
    std::snprintf(line, sizeof(line),
        "{\"iteration\":%d,\"turns\":%lld,\"turns_per_second\":%.0f,"
        "\"rollout_s\":%.2f,\"learn_s\":%.2f,\"sequences\":%zu,\"updates\":%d,"
        "\"matches\":%lld,\"win_a\":%.3f,\"draws\":%lld,\"mean_max_length\":%.2f,"
        "\"policy_loss\":%.4f,\"value_loss\":%.4f,\"entropy\":%.4f,\"approx_kl\":%.5f,"
        "\"death_W\":%lld,\"death_S\":%lld,\"death_O\":%lld,\"death_H\":%lld,\"death_A\":%lld,"
        "\"vs_frozen_win\":%.4f,\"vs_frozen_draw\":%.4f,\"vs_frozen_matches\":%lld,"
        "\"assemble_s\":%.2f,\"gpu_s\":%.2f,\"shaping\":%.4f,"
        "\"terminal_reward_per_step\":%.6f}",
        iteration, static_cast<long long>(turns), turns / seconds,
        rollout_seconds, learn_seconds, sequences.size(), updates,
        static_cast<long long>(matches), win_rate, static_cast<long long>(draws),
        samples ? length_total / samples : 0.0,
        last_policy_loss, last_value_loss, last_entropy, last_kl,
        static_cast<long long>(deaths[0]), static_cast<long long>(deaths[1]),
        static_cast<long long>(deaths[2]), static_cast<long long>(deaths[3]),
        static_cast<long long>(deaths[4]), eval_win, eval_draw,
        static_cast<long long>(eval_matches), assemble_seconds, gpu_seconds,
        options.shaping, reward_steps ? reward_total / reward_steps : 0.0);
    metrics << line << std::endl;
    std::printf("iter %5d | %8.0f turns/s | roll %5.2fs learn %5.2fs | seq %7zu | "
                "matches %7lld | len %5.2f | v %.4f | asm %5.2fs gpu %5.2fs | "
                "vs_frozen %.3f (%lld)\n",
                iteration, turns / seconds, rollout_seconds, learn_seconds,
                sequences.size(), static_cast<long long>(matches),
                samples ? length_total / samples : 0.0, last_value_loss,
                assemble_seconds, gpu_seconds, eval_win,
                static_cast<long long>(eval_matches));
    std::fflush(stdout);

    if (iteration % options.save_every == 0) {
      master.save(options.out + "/actor-latest.ts");
    }
    if (options.freeze_every > 0 && iteration > 0 && iteration % options.freeze_every == 0) {
      torch::NoGradGuard no_grad;      // the opponent becomes the current self
      for (size_t r = 0; r < frozen.size(); ++r) {
        auto target_it = frozen[r].parameters().begin();
        for (const auto& source : master.parameters()) {
          (*target_it).copy_(source.to((*target_it).device()));
          ++target_it;
        }
      }
    }
    if (iteration + 1 >= options.iterations) {
      for (auto& worker : in_flight) worker.join();
    }
  }
  return 0;
}
