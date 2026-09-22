// The scripted actor and critic, driven from C++ (spec 13).
//
// Three TorchScript modules exported by tools/export_spatial.py: the encoder
// (one turn: M_old, h_old -> h_new, M_new), the decoder (top-level, one
// teacher-forced move step, split scores, sonar) and the event critic. This
// wrapper owns the sampling loop -- the decoder step driven by the C++ prefix
// simulators -- and the teacher-forced evaluation PPO differentiates, sharing
// the same module calls and masks so the stored log probability is the one
// the update recomputes.
#pragma once

#include <array>
#include <functional>
#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

#include "agent/spatial/codec.hpp"
#include "agent/spatial/memory.hpp"

namespace dragons::train {

inline constexpr int kLatent = 16;
inline constexpr int kHidden = 128;
inline constexpr int kDecoder = 64;
inline constexpr int kSonarChoices = 17;

// Everything one encoder call needs for a batch of rows on one board size.
struct EncoderBatch {
  torch::Tensor local, facts, board_valid, visible_xy, visible_prev, visible_age, status,
      history, history_mask, entities, entity_mask, portal_pairs, pair_mask, portal_axis,
      portal_unpaired, unpaired_mask, inbox, inbox_mask, head_xy;
};

// The lossless record of one sampled command (spec 6.4 / 12.1).
struct StoredCommand {
  torch::Tensor top;           // [B] long
  torch::Tensor tokens;        // [B, K] long
  torch::Tensor active;        // [B, K] bool
  torch::Tensor masks;         // [B, K, 5] bool
  torch::Tensor prefix;        // [B, K, 49] float
  torch::Tensor split_index;   // [B] long
  torch::Tensor sonar;         // [B] long
  torch::Tensor split_feats;   // [B, C, 4] float
  torch::Tensor split_mask;    // [B, C] bool
  torch::Tensor split_sizes;   // [B, C] long
  torch::Tensor split_legal;   // [B] bool
};

struct SampledCommands {
  std::vector<bool> is_split;
  std::vector<std::string> dirs;
  std::vector<int> split_n;
  std::vector<int> sonar;      // 0 silent, j+1 emits code j
  torch::Tensor log_prob;      // [B]
  StoredCommand stored;
};

class Model {
 public:
  Model(const std::string& dir, torch::Device device, bool sonar, int max_steps);

  torch::Device device() const { return device_; }
  int max_steps() const { return K_; }
  bool sonar() const { return sonar_; }

  // encoder: returns (h_new, M_new, fused)
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Encode(const EncoderBatch& x,
                                                                  const torch::Tensor& M_old,
                                                                  const torch::Tensor& h_old);

  // one decoder step (dec, h, prev, prefix, mask) -> (logits, dec_new); the
  // collector supplies a CUDA-graph-backed one
  using StepFn = std::function<std::pair<torch::Tensor, torch::Tensor>(
      const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&)>;

  // rollout: sample one command per row, running the private prefix simulators
  SampledCommands Sample(const torch::Tensor& h, std::vector<spatial::LifetimeState*>& states,
                         bool deterministic = false, const StepFn& step = nullptr);
  // the same with a graph-backed sampling step: init(h) once, then
  // sample(prev, prefix, mask, R) -> (token, log prob, active) per step; the
  // callee owns the decoder state for all rows, so every step is on the full
  // row set (finished rows are masked to STOP)
  using SampleInitFn = std::function<void(const torch::Tensor&, int64_t)>;
  using SampleStepFn = std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>(
      const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, int64_t)>;
  SampledCommands SampleGraph(const torch::Tensor& h, std::vector<spatial::LifetimeState*>& states,
                              const SampleInitFn& init, const SampleStepFn& step);

  // PPO: teacher-forced joint log probability and entropy surrogate, [B] each
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Evaluate(const torch::Tensor& h,
                                                                    const StoredCommand& stored);

  torch::Tensor CriticValues(const torch::Tensor& map_grid, const torch::Tensor& dragons,
                             const torch::Tensor& team, const torch::Tensor& mask,
                             const torch::Tensor& team_feats, const torch::Tensor& phase);

  std::vector<torch::Tensor> ActorParameters();
  std::vector<torch::Tensor> CriticParameters();
  void CopyActorFrom(Model& other);   // replicas take the master's weights
  void Save(const std::string& dir);

  torch::jit::Module encoder, decoder, critic;

 private:
  torch::Tensor TopLogits(const torch::Tensor& h, const torch::Tensor& split_legal);
  std::pair<torch::Tensor, torch::Tensor> MoveStep(const torch::Tensor& dec, const torch::Tensor& h,
                                                    const torch::Tensor& prev, const torch::Tensor& prefix,
                                                    const torch::Tensor& mask);
  torch::Tensor SplitLogits(const torch::Tensor& h, const torch::Tensor& feats, const torch::Tensor& mask);
  torch::Tensor SonarLogits(const torch::Tensor& h, const StoredCommand& stored);
  torch::Tensor CommandFeatures(const StoredCommand& stored);
  torch::Device device_;
  bool sonar_;
  int K_;
};

// Split tables for a batch of rows, padded.
void SplitTables(const std::vector<spatial::LifetimeState*>& states, torch::Device device,
                 torch::Tensor& sizes, torch::Tensor& feats, torch::Tensor& mask,
                 torch::Tensor& legal);

}  // namespace dragons::train
