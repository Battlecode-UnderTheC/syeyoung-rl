// The learner: PPO updates on harvested matches, gradients from the exported
// actor_grad_* / critic_grad programs on every device, Adam on the host.
// (The C++ counterpart of dragons_rl/ppo.py.)
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "collector.hpp"
#include "params.hpp"
#include "pjrt.hpp"

namespace rt {

struct LearnerConfig {
  int minibatches = 4;              // Adam steps per update (random lifetime groups; event groups for the critic)
  double death_penalty = 0.3;       // on the decisions of a lifetime that died by wall, self or no valid action
  double critic_fraction = 0.1;     // of learner events the critic trains on
  double kl_limit = 0.02;           // the actor stops stepping past this
  double grad_norm = 0.5;
  int threads = 16;                 // native threads (replays, rebuilding)
  std::string dump_dir;             // where the first calls with a nonfinite gradient are written (empty: not)
};

struct UpdateStats {
  int64_t decisions = 0, excluded = 0, critic_events = 0;
  int actor_steps = 0, critic_steps = 0;
  bool stopped = false;
  double kl_max = 0, value_mse = 0, clipped = 0, h_phys = 0, h_sonar = 0, actor_norm = 0, critic_norm = 0;
  double rebuild_s = 0, batch_s = 0, actor_s = 0, critic_s = 0;
  double actor_encode_s = 0, actor_gpu_s = 0;      // parts of actor_s (mean over devices)
  int actor_calls = 0, critic_calls = 0, dropped_calls = 0;   // dropped: a nonfinite gradient
  std::string note;                 // the first dropped call's diagnosis
};

class Learner {
 public:
  Learner(const Runtime& rt, const std::string& programs, const std::vector<int>& devices, LearnerConfig config);
  ~Learner();
  UpdateStats update(std::vector<std::shared_ptr<rl::Pack>> packs, Params& actor, Params& critic, Adam& actor_opt,
                     Adam& critic_opt);

 private:
  struct Device;
  const Runtime& rt_;
  LearnerConfig config_;
  std::vector<std::unique_ptr<Device>> devices_;
  int dumps_ = 0;
};

}  // namespace rt
