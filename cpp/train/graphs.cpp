#include "graphs.hpp"

#ifdef USE_CUDA
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#endif

#include <cstdlib>

#include "agent/spatial/codec.hpp"
#include "train/gpu_state.hpp"

namespace dragons::train {

namespace sp = spatial;

struct CollectorGraphs::Impl {
#ifdef USE_CUDA
  std::unique_ptr<at::cuda::CUDAGraph> encode, step;
#endif
  bool encode_captured = false, step_captured = false, samp_captured = false;
#ifdef USE_CUDA
  std::unique_ptr<at::cuda::CUDAGraph> samp;
#endif
};

CollectorGraphs::CollectorGraphs(Model& model, torch::Device device, int64_t P, int H, int W,
                                 const GraphOptions& opts)
    : model_(model), device_(device), P_(P), H_(H), W_(W), opts_(opts),
      use_graphs_(opts.enabled && device.is_cuda()), impl_(std::make_unique<Impl>()) {
  const auto f = torch::TensorOptions().dtype(torch::kFloat).device(device);
  const auto l = torch::TensorOptions().dtype(torch::kLong).device(device);
  const auto b = torch::TensorOptions().dtype(torch::kBool).device(device);
  raw_pre = torch::zeros({P, H, W, kRaw}, f);
  raw_post = torch::zeros({P, H, W, kRaw}, f);
  rounds = torch::zeros({P}, l);
  xy = torch::zeros({P, sp::kTiles, 2}, l);
  head_xy = torch::zeros({P, 2}, l);
  status = torch::zeros({P, sp::kStatus}, f);
  history = torch::zeros({P, sp::kHistoryTurns, sp::kHistoryFeatures}, f);
  history_mask = torch::zeros({P, sp::kHistoryTurns}, b);
  entities = torch::zeros({P, sp::kMaxEntities, sp::kEntityFeatures}, f);
  entity_mask = torch::zeros({P, sp::kMaxEntities}, b);
  pairs = torch::zeros({P, opts.max_pairs, 2, 2}, l);
  pair_mask = torch::zeros({P, opts.max_pairs}, b);
  axis = torch::zeros({P, opts.max_pairs}, l);
  loose = torch::zeros({P, opts.max_loose, 2}, l);
  loose_mask = torch::zeros({P, opts.max_loose}, b);
  inbox = torch::zeros({P, opts.max_inbox, 4}, l);
  inbox_mask = torch::zeros({P, opts.max_inbox}, b);
  M_in = torch::zeros({P, kLatent, H, W}, f);
  h_in = torch::zeros({P, kHidden}, f);
  step_dec_in_ = torch::zeros({P, kDecoder}, f);
  step_h_ = torch::zeros({P, kHidden}, f);
  step_prev_ = torch::full({P}, static_cast<int64_t>(sp::kTokens), l);
  step_prefix_ = torch::zeros({P, sp::kPrefixFeatures}, f);
  step_mask_ = torch::ones({P, sp::kTokens}, b);
  samp_dec_ = torch::zeros({P, kDecoder}, f);
}

CollectorGraphs::~CollectorGraphs() = default;

void CollectorGraphs::RunEncodeEager() {
  torch::NoGradGuard no_grad;
  torch::Tensor pre = Materialize(raw_pre, rounds);
  torch::Tensor visible_prev = GatherCells(pre, xy);
  torch::Tensor last = GatherCells(raw_pre.select(3, R_LAST_SEEN).unsqueeze(1), xy).squeeze(-1);
  torch::Tensor visible_age = torch::where(last >= 0, torch::clamp(rounds.to(torch::kFloat).unsqueeze(1) - last, 0, 500) / 500.0,
                                           torch::zeros_like(last));
  torch::Tensor facts = Materialize(raw_post, rounds);
  EncoderBatch x;
  x.local = CropWindows(facts, head_xy);
  x.facts = facts;
  x.board_valid = torch::ones({P_, H_, W_}, torch::TensorOptions().dtype(torch::kBool).device(device_));
  x.visible_xy = xy; x.visible_prev = visible_prev; x.visible_age = visible_age;
  x.status = status; x.history = history; x.history_mask = history_mask;
  x.entities = entities; x.entity_mask = entity_mask;
  x.portal_pairs = pairs; x.pair_mask = pair_mask; x.portal_axis = axis;
  x.portal_unpaired = loose; x.unpaired_mask = loose_mask;
  x.inbox = inbox; x.inbox_mask = inbox_mask; x.head_xy = head_xy;
  auto [h_new, M_new, fused] = model_.Encode(x, M_in, h_in);
  enc_h_out_ = h_new;
  enc_M_out_ = M_new;
}

CollectorGraphs::EncodeOut CollectorGraphs::Encode(int64_t R) {
#ifdef USE_CUDA
  if (use_graphs_ && std::getenv("SPATIAL_NO_ENC_GRAPH") == nullptr) {
    if (!impl_->encode_captured) {
      if (enc_runs_ < opts_.warmup) {
        RunEncodeEager();
        ++enc_runs_;
      } else {
        c10::cuda::getCurrentCUDAStream(device_.index()).synchronize();
        impl_->encode = std::make_unique<at::cuda::CUDAGraph>();
        impl_->encode->capture_begin({0, 0}, cudaStreamCaptureModeThreadLocal);
        RunEncodeEager();
        impl_->encode->capture_end();
        impl_->encode_captured = true;
        impl_->encode->replay();
      }
    } else {
      impl_->encode->replay();
    }
    return {enc_h_out_.slice(0, 0, R), enc_M_out_.slice(0, 0, R)};
  }
#endif
  RunEncodeEager();
  return {enc_h_out_.slice(0, 0, R), enc_M_out_.slice(0, 0, R)};
}

void CollectorGraphs::RunStepEager() {
  torch::NoGradGuard no_grad;
  auto out = model_.decoder.run_method("move_step", step_dec_in_, step_h_, step_prev_, step_prefix_, step_mask_).toTuple();
  step_logits_ = out->elements()[0].toTensor();
  step_dec_out_ = out->elements()[1].toTensor();
}

std::pair<torch::Tensor, torch::Tensor> CollectorGraphs::MoveStep(const torch::Tensor& dec, const torch::Tensor& h,
                                                                  const torch::Tensor& prev, const torch::Tensor& prefix,
                                                                  const torch::Tensor& mask) {
  const int64_t R = dec.size(0);
  step_dec_in_.slice(0, 0, R).copy_(dec);
  step_h_.slice(0, 0, R).copy_(h);
  step_prev_.slice(0, 0, R).copy_(prev);
  step_prefix_.slice(0, 0, R).copy_(prefix);
  step_mask_.slice(0, 0, R).copy_(mask);
#ifdef USE_CUDA
  if (use_graphs_ && std::getenv("SPATIAL_NO_STEP_GRAPH") == nullptr) {
    if (!impl_->step_captured) {
      if (step_runs_ < opts_.warmup) {
        RunStepEager();
        ++step_runs_;
      } else {
        c10::cuda::getCurrentCUDAStream(device_.index()).synchronize();
        impl_->step = std::make_unique<at::cuda::CUDAGraph>();
        impl_->step->capture_begin({0, 0}, cudaStreamCaptureModeThreadLocal);
        RunStepEager();
        impl_->step->capture_end();
        impl_->step_captured = true;
        impl_->step->replay();
      }
    } else {
      impl_->step->replay();
    }
    return {step_logits_.slice(0, 0, R).clone(), step_dec_out_.slice(0, 0, R).clone()};
  }
#endif
  RunStepEager();
  return {step_logits_.slice(0, 0, R).clone(), step_dec_out_.slice(0, 0, R).clone()};
}

}  // namespace dragons::train

namespace dragons::train {

void CollectorGraphs::StepInit(const torch::Tensor& h, int64_t R) {
  torch::NoGradGuard no_grad;
  step_h_.slice(0, 0, R).copy_(h);
  samp_dec_.copy_(model_.decoder.run_method("move_init", step_h_).toTensor());
}

void CollectorGraphs::RunSampleEager() {
  torch::NoGradGuard no_grad;
  auto out = model_.decoder.run_method("move_step", samp_dec_, step_h_, step_prev_, step_prefix_, step_mask_).toTuple();
  torch::Tensor logits = out->elements()[0].toTensor().to(torch::kFloat);
  torch::Tensor dec_new = out->elements()[1].toTensor();
  torch::Tensor noise = torch::rand_like(logits).clamp_min(1.0e-30f);
  torch::Tensor sampled = (logits - torch::log(-torch::log(noise))).argmax(-1);
  torch::Tensor n_allowed = step_mask_.to(torch::kLong).sum(1);
  torch::Tensor forced = step_mask_.to(torch::kLong).argmax(1);
  torch::Tensor tok = torch::where(n_allowed > 1, sampled, forced);
  torch::Tensor active = n_allowed > 1;
  torch::Tensor lp = torch::log_softmax(logits, -1).gather(1, tok.unsqueeze(1)).squeeze(1) * active.to(torch::kFloat);
  // static outputs: assign into persistent buffers so replay refreshes them
  if (!samp_tok_.defined()) {
    samp_tok_ = tok.clone();
    samp_lp_ = lp.clone();
    samp_active_ = active.clone();
  } else {
    samp_tok_.copy_(tok);
    samp_lp_.copy_(lp);
    samp_active_.copy_(active);
  }
  samp_dec_.copy_(dec_new);
}

CollectorGraphs::Sampled CollectorGraphs::SampleStep(const torch::Tensor& prev, const torch::Tensor& prefix,
                                                     const torch::Tensor& mask, int64_t R) {
  step_prev_.slice(0, 0, R).copy_(prev);
  step_prefix_.slice(0, 0, R).copy_(prefix);
  step_mask_.slice(0, 0, R).copy_(mask);
#ifdef USE_CUDA
  if (use_graphs_ && std::getenv("SPATIAL_NO_STEP_GRAPH") == nullptr) {
    if (!impl_->samp_captured) {
      if (samp_runs_ < opts_.warmup) {
        RunSampleEager();
        ++samp_runs_;
      } else {
        c10::cuda::getCurrentCUDAStream(device_.index()).synchronize();
        impl_->samp = std::make_unique<at::cuda::CUDAGraph>();
        impl_->samp->capture_begin({0, 0}, cudaStreamCaptureModeThreadLocal);
        RunSampleEager();
        impl_->samp->capture_end();
        impl_->samp_captured = true;
        impl_->samp->replay();
      }
    } else {
      impl_->samp->replay();
    }
    return {samp_tok_.slice(0, 0, R), samp_lp_.slice(0, 0, R), samp_active_.slice(0, 0, R)};
  }
#endif
  RunSampleEager();
  return {samp_tok_.slice(0, 0, R), samp_lp_.slice(0, 0, R), samp_active_.slice(0, 0, R)};
}

}  // namespace dragons::train
