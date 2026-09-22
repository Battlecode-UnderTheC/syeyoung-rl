#include "model.hpp"

#include <algorithm>
#include <stdexcept>

namespace dragons::train {

namespace sp = spatial;
using torch::indexing::Slice;

namespace {

constexpr float kNegInf = -3.0e4f;
constexpr int kBos = sp::kTokens;   // decoder start symbol

torch::Tensor GumbelArgmax(const torch::Tensor& logits_in) {
  // always in float32: in half the noise floor underflows to zero and the
  // Gumbel term becomes NaN, so a masked action could be picked
  torch::Tensor logits = logits_in.to(torch::kFloat);
  torch::Tensor noise = torch::rand_like(logits).clamp_min(1.0e-30f);
  return (logits - torch::log(-torch::log(noise))).argmax(-1);
}

torch::Tensor Pick(const torch::Tensor& logits, bool deterministic) {
  return deterministic ? logits.to(torch::kFloat).argmax(-1) : GumbelArgmax(logits);
}

torch::Tensor LogSoftmaxF(const torch::Tensor& logits) { return torch::log_softmax(logits.to(torch::kFloat), -1); }

torch::Tensor Gather1(const torch::Tensor& logp, const torch::Tensor& idx) {
  return logp.gather(1, idx.unsqueeze(1)).squeeze(1);
}

}  // namespace

Model::Model(const std::string& dir, torch::Device device, bool sonar, int max_steps)
    : device_(device), sonar_(sonar), K_(max_steps) {
  encoder = torch::jit::load(dir + "/spatial_encoder.ts", device);
  decoder = torch::jit::load(dir + "/spatial_decoder.ts", device);
  critic = torch::jit::load(dir + "/spatial_critic.ts", device);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Model::Encode(
    const EncoderBatch& x, const torch::Tensor& M_old, const torch::Tensor& h_old) {
  auto out = encoder.forward({x.local, x.facts, x.board_valid, x.visible_xy, x.visible_prev,
                              x.visible_age, x.status, x.history, x.history_mask, x.entities,
                              x.entity_mask, x.portal_pairs, x.pair_mask, x.portal_axis,
                              x.portal_unpaired, x.unpaired_mask, x.inbox, x.inbox_mask,
                              x.head_xy, M_old, h_old}).toTuple();
  return {out->elements()[0].toTensor(), out->elements()[1].toTensor(),
          out->elements()[2].toTensor()};
}

torch::Tensor Model::TopLogits(const torch::Tensor& h, const torch::Tensor& split_legal) {
  return decoder.run_method("top_logits", h, split_legal).toTensor();
}

std::pair<torch::Tensor, torch::Tensor> Model::MoveStep(const torch::Tensor& dec, const torch::Tensor& h,
                                                         const torch::Tensor& prev, const torch::Tensor& prefix,
                                                         const torch::Tensor& mask) {
  auto out = decoder.run_method("move_step", dec, h, prev, prefix, mask).toTuple();
  return {out->elements()[0].toTensor(), out->elements()[1].toTensor()};
}

torch::Tensor Model::SplitLogits(const torch::Tensor& h, const torch::Tensor& feats, const torch::Tensor& mask) {
  return decoder.run_method("split_logits", h, feats, mask).toTensor();
}

torch::Tensor Model::CommandFeatures(const StoredCommand& s) {
  // top one-hot(2), direction counts / K (4), steps / K, split fraction
  const int64_t B = s.top.size(0);
  torch::Tensor is_stop = (s.tokens == sp::kStop).to(torch::kLong);
  torch::Tensor real = (s.tokens != sp::kStop) & (torch::cumsum(is_stop, 1) == 0);
  torch::Tensor counts = torch::zeros({B, sp::kTokens}, s.prefix.options());
  counts.scatter_add_(1, s.tokens, real.to(s.prefix.dtype()));
  torch::Tensor steps = real.to(s.prefix.dtype()).sum(1, true) / static_cast<double>(K_);
  torch::Tensor top = torch::nn::functional::one_hot(s.top, 2).to(s.prefix.dtype());
  torch::Tensor frac;
  if (s.split_feats.size(1) > 0) {
    torch::Tensor idx = s.split_index.clamp_max(s.split_feats.size(1) - 1);
    frac = s.split_feats.index({torch::arange(B, s.top.options()), idx, Slice(0, 1)});
  } else {
    frac = torch::zeros_like(steps);
  }
  frac = frac * (s.top == 1).to(s.prefix.dtype()).unsqueeze(1);
  return torch::cat({top, counts.slice(1, 0, 4) / static_cast<double>(K_), steps, frac}, 1);
}

torch::Tensor Model::SonarLogits(const torch::Tensor& h, const StoredCommand& stored) {
  return decoder.run_method("sonar_logits", h, CommandFeatures(stored)).toTensor();
}

void SplitTables(const std::vector<sp::LifetimeState*>& states, torch::Device device,
                 torch::Tensor& sizes, torch::Tensor& feats, torch::Tensor& mask, torch::Tensor& legal) {
  const int64_t B = static_cast<int64_t>(states.size());
  std::vector<sp::SplitTable> tables;
  int64_t C = 0;
  for (auto* s : states) {
    tables.push_back(sp::MakeSplitTable(*s));
    C = std::max<int64_t>(C, static_cast<int64_t>(tables.back().sizes.size()));
  }
  sizes = torch::zeros({B, C}, torch::kLong);
  feats = torch::zeros({B, C, 4}, torch::kFloat);
  mask = torch::zeros({B, C}, torch::kBool);
  legal = torch::zeros({B}, torch::kBool);
  auto sz = sizes.accessor<int64_t, 2>();
  auto ft = feats.accessor<float, 3>();
  auto mk = mask.accessor<bool, 2>();
  auto lg = legal.accessor<bool, 1>();
  for (int64_t b = 0; b < B; ++b) {
    lg[b] = states[b]->split_legal;
    for (size_t i = 0; i < tables[b].sizes.size(); ++i) {
      sz[b][i] = tables[b].sizes[i];
      mk[b][i] = true;
      for (int j = 0; j < 4; ++j) ft[b][i][j] = tables[b].features[i][j];
    }
  }
  sizes = sizes.to(device);
  feats = feats.to(device);
  mask = mask.to(device);
  legal = legal.to(device);
}

SampledCommands Model::Sample(const torch::Tensor& h, std::vector<sp::LifetimeState*>& states,
                              bool deterministic, const StepFn& step) {
  torch::NoGradGuard no_grad;
  const int64_t B = h.size(0);
  SampledCommands out;
  StoredCommand& st = out.stored;
  SplitTables(states, device_, st.split_sizes, st.split_feats, st.split_mask, st.split_legal);

  torch::Tensor top_logits = TopLogits(h, st.split_legal);
  st.top = Pick(top_logits, deterministic);
  torch::Tensor lp = Gather1(LogSoftmaxF(top_logits), st.top);
  const torch::Tensor top_host = st.top.to(torch::kCPU);   // one copy; two .to() calls would be two buffers on CUDA
  std::vector<int64_t> top_cpu(top_host.data_ptr<int64_t>(), top_host.data_ptr<int64_t>() + B);

  const auto lopt = torch::TensorOptions().dtype(torch::kLong).device(device_);
  st.tokens = torch::full({B, K_}, sp::kStop, lopt);
  st.active = torch::zeros({B, K_}, torch::TensorOptions().dtype(torch::kBool).device(device_));
  st.masks = torch::zeros({B, K_, sp::kTokens}, torch::TensorOptions().dtype(torch::kBool).device(device_));
  st.prefix = torch::zeros({B, K_, sp::kPrefixFeatures}, h.options());

  std::vector<std::unique_ptr<sp::PrefixSimulator>> sims(B);
  std::vector<int64_t> running;
  for (int64_t b = 0; b < B; ++b) {
    if (top_cpu[b] == 0) {
      sims[b] = std::make_unique<sp::PrefixSimulator>(*states[b]);
      running.push_back(b);
    }
  }
  torch::Tensor dec = decoder.run_method("move_init", h).toTensor();
  torch::Tensor prev = torch::full({B}, kBos, lopt);
  out.dirs.assign(B, "");
  for (int j = 0; j < K_ && !running.empty(); ++j) {
    const int64_t R = static_cast<int64_t>(running.size());
    torch::Tensor rows = torch::tensor(running, torch::kLong);
    torch::Tensor masks_cpu = torch::zeros({R, sp::kTokens}, torch::kBool);
    torch::Tensor prefix_cpu = torch::zeros({R, sp::kPrefixFeatures}, torch::kFloat);
    auto mk = masks_cpu.accessor<bool, 2>();
    for (int64_t r = 0; r < R; ++r) {
      std::array<bool, sp::kTokens> m;
      sims[running[r]]->Mask(m);
      for (int t = 0; t < sp::kTokens; ++t) mk[r][t] = m[t];
      sims[running[r]]->Features(prefix_cpu.data_ptr<float>() + r * sp::kPrefixFeatures);
    }
    torch::Tensor rows_d = rows.to(device_);
    torch::Tensor masks_d = masks_cpu.to(device_);
    torch::Tensor prefix_d = prefix_cpu.to(device_);
    st.masks.index_put_({rows_d, j}, masks_d);
    st.prefix.index_put_({rows_d, j}, prefix_d);
    auto [logits, dec_rows] = step
        ? step(dec.index_select(0, rows_d), h.index_select(0, rows_d), prev.index_select(0, rows_d), prefix_d, masks_d)
        : MoveStep(dec.index_select(0, rows_d), h.index_select(0, rows_d), prev.index_select(0, rows_d), prefix_d, masks_d);
    dec.index_copy_(0, rows_d, dec_rows);
    torch::Tensor n_allowed = masks_d.to(torch::kLong).sum(1);
    torch::Tensor forced = masks_d.to(torch::kLong).argmax(1);
    torch::Tensor sampled = Pick(logits, deterministic);
    torch::Tensor tok = torch::where(n_allowed > 1, sampled, forced);
    torch::Tensor active = n_allowed > 1;
    st.tokens.index_put_({rows_d, j}, tok);
    st.active.index_put_({rows_d, j}, active);
    torch::Tensor step_lp = Gather1(LogSoftmaxF(logits), tok) * active.to(lp.dtype());
    lp.index_add_(0, rows_d, step_lp);
    prev.index_copy_(0, rows_d, tok);
    torch::Tensor tok_cpu = tok.to(torch::kCPU);
    std::vector<int64_t> still;
    for (int64_t r = 0; r < R; ++r) {
      const int t = static_cast<int>(tok_cpu[r].item<int64_t>());
      if (t != sp::kStop) {
        sims[running[r]]->Advance(t);
        out.dirs[running[r]] += "NESW"[t];
        still.push_back(running[r]);
      }
    }
    running = still;
  }

  torch::Tensor is_split = st.top == 1;
  torch::Tensor split_idx = torch::zeros({B}, lopt);
  if (st.split_feats.size(1) > 0) {       // nobody may split otherwise, and SPLIT is masked
    torch::Tensor split_logits = SplitLogits(h, st.split_feats, st.split_mask);
    split_idx = Pick(split_logits, deterministic);
    lp = lp + Gather1(LogSoftmaxF(split_logits), split_idx) * is_split.to(lp.dtype());
  }
  st.split_index = torch::where(is_split, split_idx, torch::zeros_like(split_idx));
  st.sonar = torch::zeros({B}, lopt);
  if (sonar_) {
    torch::Tensor s_logits = SonarLogits(h, st);
    st.sonar = Pick(s_logits, deterministic);
    lp = lp + Gather1(LogSoftmaxF(s_logits), st.sonar);
  }
  out.log_prob = lp;
  out.is_split.assign(B, false);
  out.split_n.assign(B, 0);
  out.sonar.assign(B, 0);
  torch::Tensor idx_cpu = st.split_index.to(torch::kCPU), sizes_cpu = st.split_sizes.to(torch::kCPU);
  torch::Tensor sonar_cpu = st.sonar.to(torch::kCPU);
  for (int64_t b = 0; b < B; ++b) {
    out.sonar[b] = static_cast<int>(sonar_cpu[b].item<int64_t>());
    if (top_cpu[b] == 1) {
      out.is_split[b] = true;
      out.split_n[b] = static_cast<int>(sizes_cpu[b][idx_cpu[b].item<int64_t>()].item<int64_t>());
    }
  }
  return out;
}

SampledCommands Model::SampleGraph(const torch::Tensor& h, std::vector<sp::LifetimeState*>& states,
                                   const SampleInitFn& init, const SampleStepFn& step) {
  torch::NoGradGuard no_grad;
  const int64_t B = h.size(0);
  SampledCommands out;
  StoredCommand& st = out.stored;
  SplitTables(states, device_, st.split_sizes, st.split_feats, st.split_mask, st.split_legal);

  torch::Tensor top_logits = TopLogits(h, st.split_legal);
  st.top = Pick(top_logits, false);
  torch::Tensor lp = Gather1(LogSoftmaxF(top_logits), st.top);
  const torch::Tensor top_host = st.top.to(torch::kCPU);
  const int64_t* top_p = top_host.data_ptr<int64_t>();

  const auto lopt = torch::TensorOptions().dtype(torch::kLong).device(device_);
  const auto bopt = torch::TensorOptions().dtype(torch::kBool).device(device_);
  st.tokens = torch::full({B, K_}, sp::kStop, lopt);
  st.active = torch::zeros({B, K_}, bopt);
  st.masks = torch::zeros({B, K_, sp::kTokens}, bopt);
  st.prefix = torch::zeros({B, K_, sp::kPrefixFeatures}, h.options());

  std::vector<std::unique_ptr<sp::PrefixSimulator>> sims(B);
  std::vector<uint8_t> running(B, 0);
  int64_t n_running = 0;
  for (int64_t b = 0; b < B; ++b) {
    if (top_p[b] == 0) {
      sims[b] = std::make_unique<sp::PrefixSimulator>(*states[b]);
      running[b] = 1;
      ++n_running;
    }
  }
  init(h, B);
  torch::Tensor prev_cpu = torch::full({B}, kBos, torch::kLong);
  out.dirs.assign(B, "");
  // every step runs on all B rows: finished rows get a STOP-only mask, which
  // makes them inactive (no sampled token, no log prob) exactly as in Sample
  torch::Tensor masks_cpu = torch::zeros({B, sp::kTokens}, torch::kBool);
  torch::Tensor prefix_cpu = torch::zeros({B, sp::kPrefixFeatures}, torch::kFloat);
  for (int j = 0; j < K_ && n_running > 0; ++j) {
    bool* mk = masks_cpu.data_ptr<bool>();
    float* pf = prefix_cpu.data_ptr<float>();
    std::fill(mk, mk + B * sp::kTokens, false);
    std::fill(pf, pf + B * sp::kPrefixFeatures, 0.0f);
    for (int64_t b = 0; b < B; ++b) {
      if (!running[b]) { mk[b * sp::kTokens + sp::kStop] = true; continue; }
      std::array<bool, sp::kTokens> m;
      sims[b]->Mask(m);
      for (int t = 0; t < sp::kTokens; ++t) mk[b * sp::kTokens + t] = m[t];
      sims[b]->Features(pf + b * sp::kPrefixFeatures);
    }
    auto [tok, step_lp, active] = step(prev_cpu.to(device_, true), prefix_cpu.to(device_, true), masks_cpu.to(device_, true), B);
    st.masks.select(1, j).copy_(masks_cpu.to(device_, true));
    st.prefix.select(1, j).copy_(prefix_cpu.to(device_, true));
    st.tokens.select(1, j).copy_(tok);
    st.active.select(1, j).copy_(active);
    lp = lp + step_lp;
    torch::Tensor tok_cpu = tok.to(torch::kCPU);
    const int64_t* tp = tok_cpu.data_ptr<int64_t>();
    int64_t* pp = prev_cpu.data_ptr<int64_t>();
    for (int64_t b = 0; b < B; ++b) {
      pp[b] = tp[b];
      if (!running[b]) continue;
      const int t = static_cast<int>(tp[b]);
      if (t == sp::kStop) { running[b] = 0; --n_running; continue; }
      sims[b]->Advance(t);
      out.dirs[b] += "NESW"[t];
    }
  }

  torch::Tensor is_split = st.top == 1;
  torch::Tensor split_idx = torch::zeros({B}, lopt);
  if (st.split_feats.size(1) > 0) {
    torch::Tensor split_logits = SplitLogits(h, st.split_feats, st.split_mask);
    split_idx = Pick(split_logits, false);
    lp = lp + Gather1(LogSoftmaxF(split_logits), split_idx) * is_split.to(lp.dtype());
  }
  st.split_index = torch::where(is_split, split_idx, torch::zeros_like(split_idx));
  st.sonar = torch::zeros({B}, lopt);
  if (sonar_) {
    torch::Tensor s_logits = SonarLogits(h, st);
    st.sonar = Pick(s_logits, false);
    lp = lp + Gather1(LogSoftmaxF(s_logits), st.sonar);
  }
  out.log_prob = lp;
  out.is_split.assign(B, false);
  out.split_n.assign(B, 0);
  out.sonar.assign(B, 0);
  torch::Tensor idx_cpu = st.split_index.to(torch::kCPU), sizes_cpu = st.split_sizes.to(torch::kCPU);
  torch::Tensor sonar_cpu = st.sonar.to(torch::kCPU);
  const int64_t C = sizes_cpu.size(1);
  for (int64_t b = 0; b < B; ++b) {
    out.sonar[b] = static_cast<int>(sonar_cpu.data_ptr<int64_t>()[b]);
    if (top_p[b] == 1) {
      out.is_split[b] = true;
      out.split_n[b] = static_cast<int>(sizes_cpu.data_ptr<int64_t>()[b * C + idx_cpu.data_ptr<int64_t>()[b]]);
    }
  }
  return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Model::Evaluate(const torch::Tensor& h,
                                                                         const StoredCommand& s) {
  const int64_t B = h.size(0);
  torch::Tensor top = torch::log_softmax(TopLogits(h, s.split_legal), -1);
  torch::Tensor lp = Gather1(top, s.top);
  torch::Tensor ent = -(top.exp() * top).sum(-1);
  torch::Tensor is_move = (s.top == 0).to(h.dtype());
  torch::Tensor dec = decoder.run_method("move_init", h).toTensor();
  torch::Tensor prev = torch::full({B}, kBos, torch::TensorOptions().dtype(torch::kLong).device(device_));
  torch::Tensor move_lp = torch::zeros({B}, h.options());
  torch::Tensor move_ent = torch::zeros({B}, h.options());
  for (int j = 0; j < K_; ++j) {
    auto [logits, dec_new] = MoveStep(dec, h, prev, s.prefix.select(1, j), s.masks.select(1, j));
    dec = dec_new;
    torch::Tensor logp = LogSoftmaxF(logits);
    torch::Tensor act = s.active.select(1, j).to(h.dtype());
    move_lp = move_lp + Gather1(logp, s.tokens.select(1, j)) * act;
    move_ent = move_ent - (logp.exp() * logp).sum(-1) * act;
    prev = s.tokens.select(1, j);
  }
  torch::Tensor split_lp = torch::zeros({B}, h.options());
  torch::Tensor split_ent = torch::zeros({B}, h.options());
  if (s.split_feats.size(1) > 0) {
    torch::Tensor split = torch::log_softmax(SplitLogits(h, s.split_feats, s.split_mask), -1);
    torch::Tensor idx = s.split_index.clamp_max(split.size(1) - 1);
    split_lp = Gather1(split, idx);
    split_ent = -(split.exp() * split).sum(-1);
  }
  lp = lp + is_move * move_lp + (1 - is_move) * split_lp;
  ent = ent + is_move * move_ent / static_cast<double>(K_) + (1 - is_move) * split_ent;
  torch::Tensor sonar_ent = torch::zeros({B}, h.options());
  if (sonar_) {
    torch::Tensor sl = torch::log_softmax(SonarLogits(h, s), -1);
    lp = lp + Gather1(sl, s.sonar);
    sonar_ent = -(sl.exp() * sl).sum(-1);
  }
  return {lp, ent, sonar_ent};
}

torch::Tensor Model::CriticValues(const torch::Tensor& map_grid, const torch::Tensor& dragons,
                                  const torch::Tensor& team, const torch::Tensor& mask,
                                  const torch::Tensor& team_feats, const torch::Tensor& phase) {
  return critic.forward({map_grid, dragons, team, mask, team_feats, phase}).toTensor();
}

std::vector<torch::Tensor> Model::ActorParameters() {
  std::vector<torch::Tensor> out;
  for (auto p : encoder.parameters()) out.push_back(p);
  for (auto p : decoder.parameters()) out.push_back(p);
  return out;
}

std::vector<torch::Tensor> Model::CriticParameters() {
  std::vector<torch::Tensor> out;
  for (auto p : critic.parameters()) out.push_back(p);
  return out;
}

void Model::CopyActorFrom(Model& other) {
  torch::NoGradGuard no_grad;
  auto mine = ActorParameters();
  auto theirs = other.ActorParameters();
  if (mine.size() != theirs.size()) throw std::runtime_error("parameter count mismatch");
  for (size_t i = 0; i < mine.size(); ++i) mine[i].copy_(theirs[i].to(mine[i].device()));
}

void Model::Save(const std::string& dir) {
  encoder.save(dir + "/spatial_encoder.ts");
  decoder.save(dir + "/spatial_decoder.ts");
  critic.save(dir + "/spatial_critic.ts");
}

}  // namespace dragons::train
