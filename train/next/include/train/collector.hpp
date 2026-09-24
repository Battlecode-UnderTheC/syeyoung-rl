// One device's share of the envs: steps them with a policy and returns the
// rollout as a Batch. Owns its envs, agents, slot pool and GRU pool; nothing
// else touches them. One collector per collecting device.
#pragma once

#include <cstdint>
#include <memory>

#include "pjrt/pjrt.hpp"
#include "train/batch.hpp"
#include "train/critic.hpp"
#include "train/maps.hpp"
#include "train/params.hpp"
#include "train/programs.hpp"

namespace train {

struct CollectorConfig {
  int32_t steps = 256;       // env steps per rollout (ModelConfig::rows turns each)
  int32_t threads = 8;       // CPU threads for env stepping, FactMaps and encoding
  uint64_t seed = 0;
};

struct CollectionStats {
  int64_t turns = 0, untrained_turns = 0;
  int64_t games = 0, team_a_wins = 0, team_b_wins = 0, draws = 0, eliminations = 0;
  int64_t game_rounds = 0;   // summed over finished games
  struct {
    int64_t wall = 0, self = 0, body = 0, head_to_head = 0, no_valid_action = 0;
  } deaths;
  int64_t splits = 0, sprints = 0;
  int32_t peak_slots = 0;
  struct {
    double total = 0, act = 0, env = 0, encode = 0, gather = 0;
  } seconds;
};

struct Rollout {
  Batch batch;
  CollectionStats stats;
};

class Collector {
 public:
  Collector(const pjrt::Runtime& rt, const Artifacts& artifacts, pjrt::Device device, const MapSource& maps,
            const CollectorConfig& config);
  ~Collector();

  // Plays config.steps steps with `params` (which must be on this collector's
  // device). Games continue across calls; a trajectory cut by the end of a
  // rollout is bootstrapped.
  Rollout Collect(const Params& params);
  pjrt::Device device() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace train
