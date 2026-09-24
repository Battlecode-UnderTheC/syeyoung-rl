// Typed wrappers over the exported programs. The only place that knows their
// argument names; everything else passes structs.
#pragma once

#include "pjrt/pjrt.hpp"
#include "train/artifacts.hpp"
#include "train/params.hpp"

namespace train {

// The GRU states of every dragon slot on one device.
struct GruPool {
  pjrt::Buffer h;            // f32 [slots, hidden]
};

struct RngKey {
  pjrt::Buffer key;          // u32 [2]
};

// A batch of turns as the programs take them.
struct TurnInputs {
  const pjrt::Buffer& features;   // u8 [rows, FeatureLayout::bytes()]
  const pjrt::Buffer& critic;     // u8 [rows, CriticLayout::bytes()]
  const pjrt::Buffer& slots;      // s32 [rows]
  const pjrt::Buffer& fresh;      // pred [rows]: a dragon's first turn (zero state)
};

// Samples an action for every row and advances the rows' GRU states.
class Act {
 public:
  explicit Act(const pjrt::Runtime& rt, const Artifacts& a);
  struct Out {
    pjrt::Buffer action;     // s32 [rows]
    pjrt::Buffer log_prob;   // f32 [rows]
    pjrt::Buffer value;      // f32 [rows]
    GruPool pool;            // the input pool, updated
    RngKey key;
  };
  Out Run(pjrt::Device d, const Params& params, const TurnInputs& turns, GruPool&& pool, RngKey&& key) const;
};

// The critic's value for every row; changes nothing (bootstraps).
class Value {
 public:
  explicit Value(const pjrt::Runtime& rt, const Artifacts& a);
  pjrt::Buffer Run(pjrt::Device d, const Params& params, const TurnInputs& turns, const GruPool& pool) const;
};

// The GRU states of some slots, as Act would read them (zero where fresh).
class GatherStates {
 public:
  explicit GatherStates(const pjrt::Runtime& rt, const Artifacts& a);
  pjrt::Buffer Run(pjrt::Device d, const GruPool& pool, const pjrt::Buffer& slots, const pjrt::Buffer& fresh) const;
};

// One minibatch of chunks, replayed from their start states.
struct MinibatchInputs {
  const pjrt::Buffer& features;   // u8 [T, B, feature bytes]
  const pjrt::Buffer& critic;     // u8 [T, B, critic bytes]
  const pjrt::Buffer& valid;      // pred [T, B]
  const pjrt::Buffer& action;     // s32 [T, B]
  const pjrt::Buffer& log_prob;   // f32 [T, B]: when sampled
  const pjrt::Buffer& advantage;  // f32 [T, B]: normalised
  const pjrt::Buffer& target;     // f32 [T, B]: the critic's target
  const pjrt::Buffer& start_h;    // f32 [B, hidden]
};

struct LossWeights {
  float clip, entropy, value;
};

struct GradOut {
  std::vector<pjrt::Buffer> grads;   // the parameters' structure
  pjrt::Buffer losses;               // f32: named in train/learner.hpp's MinibatchLosses
};

class GradStep {
 public:
  explicit GradStep(const pjrt::Runtime& rt, const Artifacts& a);
  GradOut Run(pjrt::Device d, const Params& params, const MinibatchInputs& batch, const LossWeights& w) const;
};

// Averages ModelConfig::learners gradients (all on the state's device), clips
// actor and critic separately, steps Adam; a step with a non-finite gradient
// is skipped and reported.
class Apply {
 public:
  explicit Apply(const pjrt::Runtime& rt, const Artifacts& a);
  struct Out {
    Params params;
    AdamState adam;
    pjrt::Buffer grad_norms;       // f32 [2]: actor, critic (-1 when skipped)
  };
  Out Run(Params&& params, AdamState&& adam, std::vector<GradOut>&& grads, float learning_rate,
          float max_grad_norm) const;
};

}  // namespace train
