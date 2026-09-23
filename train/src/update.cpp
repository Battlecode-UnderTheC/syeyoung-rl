#include "update.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace train {

using Clock = std::chrono::steady_clock;

namespace {
double Since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
}  // namespace

Updater::Updater(const pjrt::Runtime& rt, const Artifacts& art, const std::vector<int>& devices, uint64_t seed)
    : rt_(rt),
      art_(art),
      devices_(devices),
      grad_step_(rt, art.dir, "grad_step"),
      apply_(rt, art.dir, "apply"),
      T_(art.Get("chunk")),
      B_(art.Get("minibatch")),
      R_(art.Get("record_bytes")),
      H_(art.Get("hidden")),
      C_(art.Get("map_cells")),
      M_(art.Get("map_channels")),
      Bins_(art.Get("bins")),
      rng_(seed) {
  if (static_cast<int64_t>(devices_.size()) != art.Get("learners"))
    throw std::runtime_error("the artifacts' apply averages " + std::to_string(art.Get("learners")) +
                             " learners, but " + std::to_string(devices_.size()) + " learner devices were given");
  staging_.resize(devices_.size());
  for (Staging& s : staging_) {
    s.records.resize(T_ * B_ * R_);
    s.valid.resize(T_ * B_);
    s.action.resize(T_ * B_);
    s.old_lp.resize(T_ * B_);
    s.adv.resize(T_ * B_);
    s.ret.resize(T_ * B_);
    s.h0.resize(B_ * H_);
    s.m0.resize(B_ * C_ * M_);
    s.bins0.resize(B_ * Bins_ * M_);
  }
}

// one minibatch, time-major [T, B]; slots past the chunks' ends stay invalid
void Updater::Pack(const Storage& data, const std::vector<int64_t>& chunks, Staging& s) const {
  std::fill(s.valid.begin(), s.valid.end(), 0);
  std::fill(s.action.begin(), s.action.end(), 0);
  std::fill(s.old_lp.begin(), s.old_lp.end(), 0.0f);
  std::fill(s.adv.begin(), s.adv.end(), 0.0f);
  std::fill(s.ret.begin(), s.ret.end(), 0.0f);
  std::fill(s.h0.begin(), s.h0.end(), 0.0f);
  std::fill(s.m0.begin(), s.m0.end(), 0);
  std::fill(s.bins0.begin(), s.bins0.end(), 0.0f);
  // invalid records must still decode to something finite: zero them
  for (int64_t t = 0; t < T_; ++t)
    for (int64_t b = 0; b < B_; ++b) {
      const bool used = b < static_cast<int64_t>(chunks.size()) &&
                        t < static_cast<int64_t>(data.chunks[chunks[b]].rows.size());
      if (!used) std::memset(&s.records[(t * B_ + b) * R_], 0, R_);
    }
  for (size_t b = 0; b < chunks.size(); ++b) {
    const Storage::Chunk& c = data.chunks[chunks[b]];
    for (size_t t = 0; t < c.rows.size() && static_cast<int64_t>(t) < T_; ++t) {
      const int64_t j = c.rows[t], k = static_cast<int64_t>(t) * B_ + static_cast<int64_t>(b);
      std::memcpy(&s.records[k * R_], &data.records[j * R_], R_);
      s.valid[k] = 1;
      s.action[k] = data.action[j];
      s.old_lp[k] = data.log_prob[j];
      s.adv[k] = data.advantage[j];
      s.ret[k] = data.returns[j];
    }
    std::memcpy(&s.h0[b * H_], &data.start_h[c.state * H_], H_ * 4);
    std::memcpy(&s.m0[b * C_ * M_], &data.start_map[c.state * C_ * M_], C_ * M_ * 2);
    std::memcpy(&s.bins0[b * Bins_ * M_], &data.start_bins[c.state * Bins_ * M_], Bins_ * M_ * 4);
  }
}

UpdateStats Updater::Update(TrainState& state, Storage& data, const UpdateOptions& opts) {
  const auto t_start = Clock::now();
  UpdateStats stats;
  const int64_t n = data.size();
  const size_t K = devices_.size();
  if (state.device != devices_[0]) throw std::runtime_error("the training state must live on the first learner device");

  // advantages normalised over the trained turns; explained variance of the critic
  {
    double count = 0, mean = 0, sq = 0, rm = 0;
    for (int64_t j = 0; j < n; ++j)
      if (data.end[j] != 3) {
        ++count;
        mean += data.advantage[j];
        rm += data.returns[j];
      }
    count = std::max(count, 1.0);
    mean /= count;
    rm /= count;
    double rv = 0, ev = 0;
    for (int64_t j = 0; j < n; ++j)
      if (data.end[j] != 3) {
        sq += (data.advantage[j] - mean) * (data.advantage[j] - mean);
        rv += (data.returns[j] - rm) * (data.returns[j] - rm);
        ev += (data.returns[j] - data.value[j]) * (data.returns[j] - data.value[j]);
      }
    const double sd = std::sqrt(sq / count) + 1e-8;
    stats.advantage_mean = mean;
    stats.advantage_std = sd;
    stats.explained_variance = rv > 0 ? 1.0 - ev / rv : 0.0;
    for (float& a : data.advantage) a = static_cast<float>((a - mean) / sd);
  }

  const float hyper_step[3] = {opts.clip, opts.ent_coef, opts.vf_coef};
  const float hyper_apply[2] = {opts.lr, opts.max_grad_norm};
  // the other learners' copies of the parameters
  std::vector<std::vector<pjrt::Buffer>> replicas(K);
  auto broadcast = [&] {
    for (size_t k = 1; k < K; ++k) replicas[k] = state.CopyParams(devices_[k]);
  };
  broadcast();

  std::vector<int64_t> order(data.chunks.size());
  std::iota(order.begin(), order.end(), 0);
  const int64_t per_round = static_cast<int64_t>(K) * B_;
  // full rounds only (each epoch reshuffles, so the few chunks left over differ), at least one
  const int64_t rounds = std::max<int64_t>(1, static_cast<int64_t>(order.size()) / per_round);

  for (int epoch = 0; epoch < opts.epochs; ++epoch) {
    std::shuffle(order.begin(), order.end(), rng_);
    double epoch_kl = 0;
    for (int64_t r = 0; r < rounds; ++r) {
      auto t0 = Clock::now();
      std::vector<std::vector<pjrt::Buffer>> grads(K);
      std::vector<float> metrics(K * 6, 0.0f);
      std::vector<std::exception_ptr> errors(K);
      std::vector<std::thread> threads;
      for (size_t k = 0; k < K; ++k) {
        threads.emplace_back([&, k] {
          try {
            std::vector<int64_t> mine;
            for (int64_t b = 0; b < B_; ++b) {
              const int64_t at = r * per_round + static_cast<int64_t>(k) * B_ + b;
              if (at < static_cast<int64_t>(order.size())) mine.push_back(order[at]);
            }
            Staging& s = staging_[k];
            Pack(data, mine, s);
            const int d = devices_[k];
            std::vector<pjrt::Buffer> in;
            in.push_back(rt_.Upload(d, s.records.data(), pjrt::DType::U8, {T_, B_, R_}));
            in.push_back(rt_.Upload(d, s.valid.data(), pjrt::DType::Pred, {T_, B_}));
            in.push_back(rt_.Upload(d, s.action.data(), pjrt::DType::S32, {T_, B_}));
            in.push_back(rt_.Upload(d, s.old_lp.data(), pjrt::DType::F32, {T_, B_}));
            in.push_back(rt_.Upload(d, s.adv.data(), pjrt::DType::F32, {T_, B_}));
            in.push_back(rt_.Upload(d, s.ret.data(), pjrt::DType::F32, {T_, B_}));
            in.push_back(rt_.Upload(d, s.h0.data(), pjrt::DType::F32, {B_, H_}));
            in.push_back(rt_.Upload(d, s.m0.data(), pjrt::DType::F16, {B_, C_, M_}));
            in.push_back(rt_.Upload(d, s.bins0.data(), pjrt::DType::F32, {B_, Bins_, M_}));
            for (float v : hyper_step) in.push_back(rt_.Upload(d, &v, pjrt::DType::F32, {}));
            const std::vector<pjrt::Buffer>& params = k == 0 ? state.params : replicas[k];
            std::vector<const pjrt::Buffer*> args;
            for (const auto& p : params) args.push_back(&p);
            for (const auto& b : in) args.push_back(&b);
            auto outs = grad_step_.Run(d, args);
            outs.back().ToHost(&metrics[k * 6], 6 * sizeof(float));
            outs.pop_back();
            grads[k] = k == 0 ? std::move(outs) : [&] {
              std::vector<pjrt::Buffer> moved;
              for (const auto& g : outs) moved.push_back(g.CopyTo(devices_[0]));
              return moved;
            }();
          } catch (...) {
            errors[k] = std::current_exception();
          }
        });
      }
      for (auto& t : threads) t.join();
      for (auto& e : errors)
        if (e) std::rethrow_exception(e);
      stats.round_seconds += Since(t0);

      t0 = Clock::now();
      std::vector<const pjrt::Buffer*> args;
      for (const auto& p : state.params) args.push_back(&p);
      for (const auto& o : state.opt) args.push_back(&o);
      for (size_t k = 0; k < K; ++k)
        for (const auto& g : grads[k]) args.push_back(&g);
      std::vector<pjrt::Buffer> hyper;
      for (float v : hyper_apply) hyper.push_back(rt_.Upload(devices_[0], &v, pjrt::DType::F32, {}));
      for (const auto& h : hyper) args.push_back(&h);
      auto outs = apply_.Run(devices_[0], args);
      const size_t np = state.params.size(), no = state.opt.size();
      for (size_t i = 0; i < np; ++i) state.params[i] = std::move(outs[i]);
      for (size_t i = 0; i < no; ++i) state.opt[i] = std::move(outs[np + i]);
      float norms[2];
      outs[np + no].ToHost(norms, sizeof norms);
      broadcast();
      stats.apply_seconds += Since(t0);

      double kl = 0, clip = 0;
      for (size_t k = 0; k < K; ++k) {
        for (int m = 0; m < 5; ++m) stats.metrics[m] += metrics[k * 6 + m];
        stats.metrics[7] += metrics[k * 6 + 5];
        kl += metrics[k * 6 + 3];
        clip += metrics[k * 6 + 4];
      }
      if (stats.rounds == 0) {
        stats.replay_kl = kl / K;
        stats.replay_clip = clip / K;
      }
      stats.metrics[5] += norms[0];
      stats.metrics[6] += norms[1];
      epoch_kl += kl / K;
      stats.minibatches += static_cast<int64_t>(K);
      ++stats.rounds;
    }
    stats.epochs = epoch + 1;
    if (opts.target_kl > 0 && epoch_kl / rounds > opts.target_kl) break;
  }
  for (int m : {0, 1, 2, 3, 4, 7}) stats.metrics[m] /= std::max<int64_t>(stats.minibatches, 1);
  for (int m : {5, 6}) stats.metrics[m] /= std::max<int64_t>(stats.rounds, 1);
  stats.seconds = Since(t_start);
  return stats;
}

}  // namespace train
