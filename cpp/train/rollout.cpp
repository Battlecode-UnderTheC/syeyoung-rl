// Native rollout: the sim, the observation encoding and the policy all run
// without returning to Python between dragon-turns.
//
// The policy is the TorchScript module exported by snake_rl.model.export, so
// there is exactly one definition of the network and the C++ side cannot drift
// from it.
//
// Recurrent state lives in one device tensor indexed by (arena, slot). Dragon
// slots are stable for a match -- the dragon vector only ever grows -- so a
// dragon's row never moves, and a newly split dragon simply starts at zero.

#include <torch/cuda.h>
#include <torch/script.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "agent/encoder.hpp"

namespace {

std::vector<std::string> ReadMaps(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot read maps file: " + path);
  std::vector<std::string> maps;
  std::string line, current;
  while (std::getline(input, line)) {
    if (line == "---") {           // maps are separated by a lone ---
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

// The action space and hidden size are written by tools/export_for_native.py
// alongside the model, so the two can never be configured apart.
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
    if (key == "max_sprint") {
      input >> config.max_sprint;
    } else if (key == "hidden") {
      input >> config.hidden;
    } else if (key == "actions") {
      input >> config.actions;
    } else if (key == "split_sizes") {
      std::string rest;
      std::getline(input, rest);
      std::string number;
      for (char c : rest) {
        if (c == ',' || c == ' ') {
          if (!number.empty()) { config.split_sizes.push_back(std::stoi(number)); number.clear(); }
        } else {
          number += c;
        }
      }
      if (!number.empty()) config.split_sizes.push_back(std::stoi(number));
    }
  }
  return config;
}

int Arg(int argc, char** argv, const std::string& flag, int fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (flag == argv[i]) return std::atoi(argv[i + 1]);
  }
  return fallback;
}

std::string ArgStr(int argc, char** argv, const std::string& flag,
                   const std::string& fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (flag == argv[i]) return argv[i + 1];
  }
  return fallback;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace dragons;

  const std::string model_path = ArgStr(argc, argv, "--model", "actor.ts");
  const std::string maps_path = ArgStr(argc, argv, "--maps", "maps.txt");
  const int rounds = Arg(argc, argv, "--rounds", 60);
  const int max_slots = Arg(argc, argv, "--max-slots", 64);
  const bool use_cuda = torch::cuda::is_available() &&
                        ArgStr(argc, argv, "--device", "cuda") != "cpu";
  const torch::Device device(use_cuda ? torch::kCUDA : torch::kCPU,
                             static_cast<torch::DeviceIndex>(Arg(argc, argv, "--gpu", 0)));

  std::vector<std::string> maps = ReadMaps(maps_path);
  const int arenas = Arg(argc, argv, "--arenas", static_cast<int>(maps.size()));
  if (maps.empty()) throw std::runtime_error("no maps");
  std::vector<std::string> chosen_maps(arenas);
  for (int i = 0; i < arenas; ++i) chosen_maps[i] = maps[i % maps.size()];
  maps.swap(chosen_maps);

  torch::jit::script::Module policy = torch::jit::load(model_path, device);
  policy.eval();
  torch::NoGradGuard no_grad;

  const SpaceConfig config = ReadSpace(ArgStr(argc, argv, "--space", "native/space.txt"));
  ActionSpace space(config.max_sprint, config.split_sizes, true);
  if (config.actions != 0 && config.actions != space.size()) {
    throw std::runtime_error("action space disagrees with the exported model");
  }
  VecGame vec(maps, 0, space, true);
  const int n_actions = space.size();
  const int hidden_size = config.hidden;

  // Staging buffers, filled by the sim and reused every slot.
  auto cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  torch::Tensor grid = torch::empty({arenas, kChannels, kVision, kVision}, cpu);
  torch::Tensor scalars = torch::empty({arenas, kScalars}, cpu);
  torch::Tensor mask = torch::empty({arenas, n_actions},
                                    cpu.dtype(torch::kBool));
  std::vector<int32_t> arena_of(arenas), dragon_of(arenas), chosen(arenas);
  if (use_cuda) {
    grid = grid.pin_memory();
    scalars = scalars.pin_memory();
    mask = mask.pin_memory();
  }

  torch::Tensor hidden = torch::zeros({arenas * max_slots, hidden_size},
                                      torch::TensorOptions().device(device));
  torch::Tensor messages = torch::zeros({arenas, 8, 4},
                                        torch::TensorOptions().dtype(torch::kLong).device(device));
  torch::Tensor message_mask = torch::zeros({arenas, 8},
                                            torch::TensorOptions().dtype(torch::kBool).device(device));
  torch::Tensor rows_index = torch::empty({arenas},
                                          torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));

  int64_t turns = 0;
  int64_t calls = 0;
  double sim_seconds = 0.0, copy_seconds = 0.0, policy_seconds = 0.0, back_seconds = 0.0;
  auto now = [] { return std::chrono::steady_clock::now(); };
  auto since = [](auto start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  };
  const auto started = std::chrono::steady_clock::now();
  for (int round = 0; round < rounds; ++round) {
    vec.BeginRound();
    for (int slot = 0;; ++slot) {
      auto mark = now();
      const int rows = vec.PrepareSlot(slot, grid.data_ptr<float>(),
                                       scalars.data_ptr<float>(),
                                       mask.data_ptr<bool>(), arena_of.data(),
                                       dragon_of.data());
      sim_seconds += since(mark);
      if (rows == 0) break;
      ++calls;

      auto index_accessor = rows_index.accessor<int64_t, 1>();
      for (int row = 0; row < rows; ++row) {
        index_accessor[row] = static_cast<int64_t>(arena_of[row]) * max_slots + slot;
      }
      mark = now();
      torch::Tensor rows_on_device =
          rows_index.narrow(0, 0, rows).to(device, /*non_blocking=*/true);
      torch::Tensor state = hidden.index_select(0, rows_on_device);

      std::vector<torch::jit::IValue> inputs{
          grid.narrow(0, 0, rows).to(device, true),
          scalars.narrow(0, 0, rows).to(device, true),
          messages.narrow(0, 0, rows),
          message_mask.narrow(0, 0, rows),
          mask.narrow(0, 0, rows).to(device, true),
          state};
      if (use_cuda) torch::cuda::synchronize();
      copy_seconds += since(mark);

      mark = now();
      auto output = policy.forward(inputs).toTuple();
      torch::Tensor logits = output->elements()[0].toTensor();
      torch::Tensor new_state = output->elements()[3].toTensor();
      if (use_cuda) torch::cuda::synchronize();
      policy_seconds += since(mark);
      mark = now();

      // Gumbel-max: exact categorical sampling without a distribution object.
      torch::Tensor noise = torch::rand_like(logits).clamp_min(1e-20);
      torch::Tensor actions = (logits - torch::log(-torch::log(noise)))
                                  .argmax(-1)
                                  .to(torch::kInt32)
                                  .to(torch::kCPU);
      hidden.index_copy_(0, rows_on_device, new_state);

      back_seconds += since(mark);
      mark = now();
      const int32_t* picked = actions.data_ptr<int32_t>();
      std::copy(picked, picked + rows, chosen.begin());
      vec.SubmitSlot(chosen.data(), nullptr, nullptr);
      sim_seconds += since(mark);
      turns += rows;
    }
    vec.EndRound();
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

  const double per = 1e6 / static_cast<double>(turns);
  std::printf("arenas %5d | %7lld turns in %5.2fs | %6.2f us/turn | %8.0f turns/s\n",
              arenas, static_cast<long long>(turns), seconds,
              seconds * per, static_cast<double>(turns) / seconds);
  std::printf("   sim %5.2f us/turn | h2d %5.2f | policy %5.2f | d2h %5.2f | "
              "%lld calls, %.0f rows/call, %.1f ms/call\n",
              sim_seconds * per, copy_seconds * per, policy_seconds * per,
              back_seconds * per, static_cast<long long>(calls),
              static_cast<double>(turns) / static_cast<double>(calls),
              (seconds / static_cast<double>(calls)) * 1e3);
  return 0;
}
