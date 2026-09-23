// The PPO update, data-parallel: epochs over the rollout's chunks, shuffled
// into minibatches of the exported grad_step's fixed shape and replayed from
// their saved start states. Each round every learner device takes one
// minibatch (packed and uploaded by its own thread); their gradients meet on
// the first learner, where `apply` averages them and steps Adam, and the new
// parameters go back to the others.
#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "artifacts.hpp"
#include "pjrt.hpp"
#include "rollout.hpp"

namespace train {

struct UpdateOptions {
  int epochs = 2;
  float lr = 2.5e-4f, clip = 0.2f, ent_coef = 0.01f, vf_coef = 0.5f, max_grad_norm = 1.0f;
  float target_kl = 0.0f;   // > 0: stop the epochs early once an epoch's mean approx KL passes it
};

struct UpdateStats {
  // means over minibatches: policy_loss, value_loss, entropy, approx_kl, clip_fraction;
  // then over rounds: actor_grad_norm, critic_grad_norm; then samples per minibatch
  double metrics[8] = {0};
  int64_t minibatches = 0, rounds = 0, epochs = 0;
  double seconds = 0, round_seconds = 0, apply_seconds = 0;
  double advantage_mean = 0, advantage_std = 0, explained_variance = 0;
  // the first round runs before any change to the parameters: its approx KL and clip
  // fraction measure how exactly the update replays what the rollout sampled (both ~0)
  double replay_kl = -1, replay_clip = -1;
};

class Updater {
 public:
  // `devices`: the learners; the first holds the parameters and the Adam state
  Updater(const pjrt::Runtime& rt, const Artifacts& art, const std::vector<int>& devices, uint64_t seed);
  // `state` must live on devices[0]
  UpdateStats Update(TrainState& state, Storage& data, const UpdateOptions& opts);

 private:
  struct Staging {
    std::vector<uint8_t> records, valid;
    std::vector<int32_t> action;
    std::vector<float> old_lp, adv, ret, h0, bins0;
    std::vector<uint16_t> m0;
  };
  void Pack(const Storage& data, const std::vector<int64_t>& chunks, Staging& s) const;

  const pjrt::Runtime& rt_;
  const Artifacts& art_;
  std::vector<int> devices_;
  pjrt::Program grad_step_, apply_;
  int64_t T_, B_, R_, H_, C_, M_, Bins_;
  std::vector<Staging> staging_;
  std::mt19937_64 rng_;
};

}  // namespace train
