// Multi-threaded native rollout.
//
// The sim is pure CPU and embarrassingly parallel across arenas, so each worker
// thread owns its own VecGame and drives it independently. Threads are pinned
// round-robin across the available GPUs; each carries its own copy of the
// policy (a couple of MB) and its own CUDA stream, so no thread waits on
// another's launch queue and there is no shared mutable state at all.
//
// One process, not one per core: a CUDA context costs hundreds of MB of device
// memory, so 28 processes would not fit on an 11 GB card.

#include <ATen/Parallel.h>
#include <torch/cuda.h>
#include <torch/script.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "agent/encoder.hpp"

#ifdef USE_CUDA_STREAMS
#include <c10/cuda/CUDAStream.h>
#endif

namespace {

std::vector<std::string> ReadMaps(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot read maps file: " + path);
  std::vector<std::string> maps;
  std::string line, current;
  while (std::getline(input, line)) {
    if (line == "---") {
      if (!current.empty()) maps.push_back(current);
      current.clear();
    } else {
      current += line;
      current += '\n';
    }
  }
  if (!current.empty()) maps.push_back(current);
  return maps;
}

int Arg(int argc, char** argv, const std::string& flag, int fallback) {
  for (int i = 1; i + 1 < argc; ++i) if (flag == argv[i]) return std::atoi(argv[i + 1]);
  return fallback;
}

std::string ArgStr(int argc, char** argv, const std::string& flag,
                   const std::string& fallback) {
  for (int i = 1; i + 1 < argc; ++i) if (flag == argv[i]) return argv[i + 1];
  return fallback;
}

struct SpaceConfig {
  int max_sprint = 3;
  std::vector<int> split_sizes;
  int hidden = 256;
  int actions = 0;
};

SpaceConfig ReadSpace(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot read space file: " + path);
  SpaceConfig config;
  std::string key;
  while (input >> key) {
    if (key == "max_sprint") input >> config.max_sprint;
    else if (key == "hidden") input >> config.hidden;
    else if (key == "actions") input >> config.actions;
    else if (key == "split_sizes") {
      std::string rest, number;
      std::getline(input, rest);
      for (char c : rest) {
        if (c == ',' || c == ' ') {
          if (!number.empty()) { config.split_sizes.push_back(std::stoi(number)); number.clear(); }
        } else number += c;
      }
      if (!number.empty()) config.split_sizes.push_back(std::stoi(number));
    }
  }
  return config;
}

struct Stats {
  int64_t turns = 0;
  double sim_seconds = 0.0;
  double policy_seconds = 0.0;
};

void RunWorker(int worker_index, const std::vector<std::string>& maps,
               const SpaceConfig& config, const std::string& model_path,
               torch::Device device, int rounds, int max_slots, Stats* stats) {
  torch::NoGradGuard no_grad;
  torch::jit::script::Module policy = torch::jit::load(model_path, device);
  policy.eval();

  dragons::ActionSpace space(config.max_sprint, config.split_sizes, true);
  dragons::VecGame vec(maps, static_cast<uint64_t>(worker_index) * 7919, space, true);
  const int arenas = static_cast<int>(maps.size());
  const int actions = space.size();

  auto cpu = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor grid = torch::empty({arenas, dragons::kChannels,
                                     dragons::kVision, dragons::kVision}, cpu);
  torch::Tensor scalars = torch::empty({arenas, dragons::kScalars}, cpu);
  torch::Tensor mask = torch::empty({arenas, actions}, cpu.dtype(torch::kBool));
  torch::Tensor rows_index = torch::empty({arenas}, cpu.dtype(torch::kLong));
  const bool cuda = device.is_cuda();
  if (cuda) { grid = grid.pin_memory(); scalars = scalars.pin_memory();
              mask = mask.pin_memory(); }

  auto on_device = torch::TensorOptions().device(device);
  torch::Tensor hidden = torch::zeros({arenas * max_slots, config.hidden}, on_device);
  torch::Tensor messages = torch::zeros({arenas, 8, 4},
                                        on_device.dtype(torch::kLong));
  torch::Tensor message_mask = torch::zeros({arenas, 8}, on_device.dtype(torch::kBool));
  std::vector<int32_t> arena_of(arenas), dragon_of(arenas), chosen(arenas);

  auto now = [] { return std::chrono::steady_clock::now(); };
  auto since = [](auto start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  };

  for (int round = 0; round < rounds; ++round) {
    vec.BeginRound();
    for (int slot = 0;; ++slot) {
      auto mark = now();
      const int rows = vec.PrepareSlot(slot, grid.data_ptr<float>(),
                                       scalars.data_ptr<float>(),
                                       mask.data_ptr<bool>(), arena_of.data(),
                                       dragon_of.data());
      stats->sim_seconds += since(mark);
      if (rows == 0) break;

      mark = now();
      auto index = rows_index.accessor<int64_t, 1>();
      for (int row = 0; row < rows; ++row) {
        index[row] = static_cast<int64_t>(arena_of[row]) * max_slots + slot;
      }
      torch::Tensor rows_on_device = rows_index.narrow(0, 0, rows).to(device, true);
      std::vector<torch::jit::IValue> inputs{
          grid.narrow(0, 0, rows).to(device, true),
          scalars.narrow(0, 0, rows).to(device, true),
          messages.narrow(0, 0, rows), message_mask.narrow(0, 0, rows),
          mask.narrow(0, 0, rows).to(device, true),
          hidden.index_select(0, rows_on_device)};
      auto output = policy.forward(inputs).toTuple();
      torch::Tensor logits = output->elements()[0].toTensor();
      hidden.index_copy_(0, rows_on_device, output->elements()[3].toTensor());
      torch::Tensor noise = torch::rand_like(logits).clamp_min(1e-20);
      torch::Tensor actions_out = (logits - torch::log(-torch::log(noise)))
                                      .argmax(-1).to(torch::kInt32).to(torch::kCPU);
      stats->policy_seconds += since(mark);

      mark = now();
      const int32_t* picked = actions_out.data_ptr<int32_t>();
      std::copy(picked, picked + rows, chosen.begin());
      vec.SubmitSlot(chosen.data(), nullptr, nullptr);
      stats->sim_seconds += since(mark);
      stats->turns += rows;
    }
    vec.EndRound();
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string model_path = ArgStr(argc, argv, "--model", "native/actor.ts");
  const std::string maps_path = ArgStr(argc, argv, "--maps", "native/maps.txt");
  const SpaceConfig config = ReadSpace(ArgStr(argc, argv, "--space", "native/space.txt"));
  const int rounds = Arg(argc, argv, "--rounds", 40);
  const int threads = Arg(argc, argv, "--threads", 1);
  const int per_thread = Arg(argc, argv, "--arenas-per-thread", 2048);
  const int gpus = Arg(argc, argv, "--gpus", 1);
  const int first_gpu = Arg(argc, argv, "--first-gpu", 0);
  const int max_slots = Arg(argc, argv, "--max-slots", 64);

  const std::vector<std::string> pool = ReadMaps(maps_path);
  if (pool.empty()) throw std::runtime_error("no maps");

  at::set_num_threads(1);             // the sim is the parallel part, not BLAS
  std::vector<Stats> stats(threads);
  std::vector<std::thread> workers;
  const auto started = std::chrono::steady_clock::now();
  for (int t = 0; t < threads; ++t) {
    std::vector<std::string> maps(per_thread);
    for (int i = 0; i < per_thread; ++i) {
      maps[i] = pool[(static_cast<size_t>(t) * per_thread + i) % pool.size()];
    }
    const torch::Device device = torch::cuda::is_available() && gpus > 0
        ? torch::Device(torch::kCUDA, static_cast<torch::DeviceIndex>(first_gpu + t % gpus))
        : torch::Device(torch::kCPU);
    workers.emplace_back(RunWorker, t, std::move(maps), std::cref(config),
                         std::cref(model_path), device, rounds, max_slots, &stats[t]);
  }
  for (std::thread& worker : workers) worker.join();
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

  int64_t turns = 0;
  double sim = 0.0, policy = 0.0;
  for (const Stats& s : stats) { turns += s.turns; sim += s.sim_seconds; policy += s.policy_seconds; }
  std::printf("threads %2d | gpus %d | arenas %6d | %9lld turns in %5.2fs | "
              "%10.0f turns/s | per-thread %8.0f/s | sim %4.1f%% policy %4.1f%%\n",
              threads, gpus, threads * per_thread, static_cast<long long>(turns),
              seconds, static_cast<double>(turns) / seconds,
              static_cast<double>(turns) / seconds / threads,
              100.0 * sim / (sim + policy), 100.0 * policy / (sim + policy));
  return 0;
}
