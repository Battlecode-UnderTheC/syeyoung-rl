// PPO over one or more Batches, data-parallel across the learner devices.
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "pjrt/pjrt.hpp"
#include "train/batch.hpp"
#include "train/params.hpp"
#include "train/programs.hpp"
#include "train/returns.hpp"

namespace train {

struct UpdateConfig {
  int32_t epochs = 2;
  float learning_rate = 2.5e-4f;
  LossWeights weights{0.2f, 0.01f, 0.5f};
  float max_grad_norm = 1.0f;
  std::optional<float> target_kl;   // stop the epochs early past it
};

// The loss terms of a minibatch, as grad_step reports them.
struct MinibatchLosses {
  float policy, value, entropy, approx_kl, clip_fraction, samples;
};

struct UpdateReport {
  MinibatchLosses mean;              // over minibatches
  float actor_grad_norm, critic_grad_norm;
  int32_t skipped_steps;             // non-finite gradients
  int32_t steps, epochs;
  float explained_variance;
  // the first step runs before any parameter change: its approx KL measures how
  // exactly the update replays what the rollout sampled (~0)
  float replay_kl;
  struct {
    double total = 0, pack = 0, grads = 0, apply = 0;
  } seconds;
};

class Learner {
 public:
  // The training state lives on devices[0]; the others hold copies.
  Learner(const pjrt::Runtime& rt, const Artifacts& artifacts, std::vector<pjrt::Device> devices,
          uint64_t seed);
  ~Learner();

  // One PPO update on the batches (one per collector). Leaves the batches unchanged.
  UpdateReport Update(Checkpoint& state, std::span<const Batch* const> batches,
                      std::span<const Returns* const> returns, const UpdateConfig& config);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace train
