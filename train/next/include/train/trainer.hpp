// The training loop: collect on every collecting device, compute returns,
// update on the learner devices, evaluate now and then, checkpoint, log.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "pjrt/pjrt.hpp"
#include "train/collector.hpp"
#include "train/evaluation.hpp"
#include "train/learner.hpp"
#include "train/maps.hpp"
#include "train/returns.hpp"

namespace train {

// Which GPUs do what. Default: all of them do both, one after the other.
struct DevicePlan {
  std::vector<pjrt::Device> collect;
  std::vector<pjrt::Device> learn;
};

struct TrainerConfig {
  std::string artifacts;
  pjrt::RuntimeOptions runtime;
  std::string run_dir;                        // checkpoints, metrics.jsonl
  DevicePlan devices;
  std::vector<std::string> map_dirs;
  double generated_map_share = 0.7;
  GeneratorConfig generator;
  CollectorConfig collector;
  ReturnsConfig returns;
  UpdateConfig update;
  int32_t checkpoint_every = 10;
  int32_t evaluate_every = 50;
  EvaluationConfig evaluation;
  std::vector<std::string> opponents;         // "scripted:random-safe", "checkpoint:<path>"
  std::optional<int64_t> iterations;          // absent: run until stopped

  // From --name value flags; unknown flags and bad values throw.
  static TrainerConfig FromArgs(int argc, char** argv);
};

class Trainer {
 public:
  explicit Trainer(const TrainerConfig& config);   // resumes from the newest good checkpoint in run_dir
  void Run();
};

}  // namespace train
