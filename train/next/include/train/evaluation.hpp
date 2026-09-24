// Playing strength against fixed opponents, measured apart from training.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "train/maps.hpp"
#include "train/params.hpp"

namespace train {

// Someone to play against: a checkpoint, or a scripted bot.
class Opponent {
 public:
  virtual ~Opponent() = default;
  virtual std::string name() const = 0;
};
std::unique_ptr<Opponent> CheckpointOpponent(const std::string& checkpoint_path);
std::unique_ptr<Opponent> ScriptedOpponent(const std::string& which);   // "random-safe", "greedy-pearl"

struct MatchupResult {
  std::string opponent;
  int64_t games = 0, wins = 0, losses = 0, draws = 0;   // from the evaluated policy's side, both colours
};

struct EvaluationConfig {
  int64_t games_per_opponent = 1024;
  uint64_t seed = 0;
};

class Evaluator {
 public:
  Evaluator(const pjrt::Runtime& rt, const Artifacts& artifacts, pjrt::Device device, const MapSource& maps);
  std::vector<MatchupResult> Evaluate(const Params& policy, const std::vector<const Opponent*>& opponents,
                                      const EvaluationConfig& config);
};

}  // namespace train
