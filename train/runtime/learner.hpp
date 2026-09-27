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
  // the return: the match's end G (terminal_return below), plus rewards on the decisions
  // where they happen, discounted backwards over the dragon's earlier decisions
  double longest_weight = 0.5;      // G: + this x (our longest - theirs) / the larger
  double total_weight = 0.25;       // G: + this x (our total length - theirs) / their sum
  double pearl_reward = 0.05;       // per pearl eaten
  double death_penalty = 0.1;       // a death (none for a head-to-head) ...
  double length_cost = 0.05;        // ... and its length lost, per segment (head-to-heads too)
  double ram_reward = 0.1;          // per enemy that died ramming into the dragon's body
  double shaping_discount = 0.95;   // per decision, backwards from each event
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
  double actor_launch_s = 0, actor_weights_s = 0, actor_mean_s = 0;
  double actor_fill = 0, actor_slots = 0;
  double launch_parts[3] = {0, 0, 0};               // launch: upload enqueue, execute, download enqueue           // decisions / decision slots of the actor calls
  int actor_calls = 0, critic_calls = 0, dropped_calls = 0;   // dropped: a nonfinite gradient
  std::string note;                 // the first dropped call's diagnosis
};

// G's largest size (1 + longest_weight + total_weight): the critic (tanh) predicts G / this
inline constexpr double kReturnScale = 1.75;

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
  double terminal_return(const rl::MatchRecord& m, int32_t team) const;
  std::vector<std::unique_ptr<Device>> devices_;
  int dumps_ = 0;
};

}  // namespace rt
