// CUDA-graph replay of the collector's GPU work at a fixed padded batch.
//
// Collection is launch-bound: an encoder call is a few hundred small kernels
// and a decoder step a few dozen, and each launch costs more than the work it
// does at these sizes. Capturing the whole materialise + encode pass, and one
// decoder step, as CUDA graphs at a fixed batch size P turns each into one
// launch. Batches shorter than P are padded with dummy rows whose outputs are
// ignored; the graphs' inputs and outputs are static tensors owned here.
//
// Without CUDA (or with graphs disabled) the same code runs eagerly.
#pragma once

#include <functional>
#include <memory>

#include <torch/torch.h>

#include "train/model.hpp"

namespace dragons::train {

struct GraphOptions {
  bool enabled = true;
  int64_t max_pairs = 8;     // portal pairs / unpaired portals / inbox words per row
  int64_t max_loose = 8;
  int64_t max_inbox = 8;
  int warmup = 3;
};

class CollectorGraphs {
 public:
  CollectorGraphs(Model& model, torch::Device device, int64_t P, int H, int W, const GraphOptions& opts);
  ~CollectorGraphs();

  int64_t capacity() const { return P_; }
  bool active() const { return use_graphs_; }

  // Static input buffers for the encode pass, [P, ...] on the device. The
  // caller fills the first R rows (the rest keep their last values) and calls
  // Encode(R); pre/post raw rows are the slot rows before and after the turn's
  // update. Outputs are views of static tensors: h_new [R,128], M_new [R,16,H,W].
  torch::Tensor raw_pre, raw_post, rounds, xy, head_xy, status, history, history_mask, entities,
      entity_mask, pairs, pair_mask, axis, loose, loose_mask, inbox, inbox_mask, M_in, h_in;
  struct EncodeOut { torch::Tensor h_new, M_new; };
  EncodeOut Encode(int64_t R);

  // One decoder step at the padded batch: fills the static buffers from the
  // given (R-row) tensors, replays, returns [R]-row views.
  std::pair<torch::Tensor, torch::Tensor> MoveStep(const torch::Tensor& dec, const torch::Tensor& h,
                                                   const torch::Tensor& prev, const torch::Tensor& prefix,
                                                   const torch::Tensor& mask);
  // The same step with the sampling inside the graph: returns (token, step
  // log prob, active) for the first R rows, and keeps the decoder state for
  // all rows in its own static buffer (initialised by StepInit from h).
  struct Sampled { torch::Tensor token, log_prob, active; };
  void StepInit(const torch::Tensor& h, int64_t R);
  Sampled SampleStep(const torch::Tensor& prev, const torch::Tensor& prefix, const torch::Tensor& mask, int64_t R);

 private:
  void RunEncodeEager();
  void RunStepEager();
  Model& model_;
  torch::Device device_;
  int64_t P_;
  int H_, W_;
  GraphOptions opts_;
  bool use_graphs_;
  // encode graph
  torch::Tensor enc_h_out_, enc_M_out_;
  int enc_runs_ = 0;
  struct Impl;
  std::unique_ptr<Impl> impl_;
  // step graph
  torch::Tensor step_dec_in_, step_h_, step_prev_, step_prefix_, step_mask_, step_logits_, step_dec_out_;
  int step_runs_ = 0;
  // sampling step graph
  void RunSampleEager();
  torch::Tensor samp_dec_, samp_tok_, samp_lp_, samp_active_;
  int samp_runs_ = 0;
};

}  // namespace dragons::train
