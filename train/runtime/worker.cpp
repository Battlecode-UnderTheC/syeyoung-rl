#include "worker.hpp"

#include <chrono>
#include <stdexcept>

namespace rt {

namespace {

bool has_prefix(const std::string& s, const std::string& p) { return s.compare(0, p.size(), p) == 0; }

int64_t us_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t).count();
}

rl::ActorBatch actor_batch(HostInputs& in, const std::string& raw, const std::string& h) {
  rl::ActorBatch b;
  b.k_pad = static_cast<int32_t>(in.spec(raw + "/inbox").dims[1]);
  b.e_pad = static_cast<int32_t>(in.spec(raw + "/echo").dims[2]);
  b.win = in.as<int32_t>(raw + "/win");
  b.wdest = in.as<int32_t>(raw + "/wdest");
  b.grid = in.as<int32_t>(raw + "/grid");
  b.entity = in.as<int32_t>(raw + "/entity");
  b.segment = in.as<int32_t>(raw + "/segment");
  b.status = in.as<int32_t>(raw + "/status");
  b.prev_dirs = in.as<int32_t>(raw + "/prev_dirs");
  b.prev_words = in.as<uint8_t>(raw + "/prev_words");
  b.inbox = in.as<uint8_t>(raw + "/inbox");
  b.echo = in.as<uint8_t>(raw + "/echo");
  b.echo_len = in.as<int32_t>(raw + "/echo_len");
  b.h = in.as<float>(h);
  return b;
}

rl::CriticBatch critic_batch(HostInputs& in, const std::string& raw) {
  rl::CriticBatch b;
  b.d_pad = static_cast<int32_t>(in.spec(raw + "/dragons").dims[1]);
  b.cells = in.as<int32_t>(raw + "/cells");
  b.dragons = in.as<int32_t>(raw + "/dragons");
  b.global = in.as<int32_t>(raw + "/global");
  b.summary = in.as<float>(raw + "/summary");
  return b;
}

}  // namespace

HostInputs::HostInputs(const Program& p, const std::vector<std::string>& weight_prefixes) : program_(p) {
  storage_.resize(p.inputs().size());
  for (size_t i = 0; i < p.inputs().size(); ++i) {
    const TensorSpec& s = p.inputs()[i];
    bool weight = false;
    for (const auto& w : weight_prefixes) weight |= has_prefix(s.name, w + "/");
    if (weight) continue;
    storage_[i].assign(s.bytes(), 0);
    indices_.push_back(static_cast<int>(i));
  }
}

void* HostInputs::data(const std::string& name) { return storage_.at(program_.input(name)).data(); }

const TensorSpec& HostInputs::spec(const std::string& name) const { return program_.inputs().at(program_.input(name)); }

void Weights::set(const Runtime& rt, const Program& p, const std::string& prefix, const std::vector<HostArray>& arrays) {
  for (const HostArray& a : arrays) {
    const int i = p.input(prefix + "/" + a.spec.name);
    if (p.inputs()[i].bytes() != a.data.size()) throw std::invalid_argument("weight shape mismatch: " + a.spec.name);
    by_input_[i] = std::make_shared<Buffer>(upload(rt, p.device(), p.inputs()[i], a.data.data()));
  }
}

PJRT_Buffer* Weights::get(int input) const { return by_input_.at(input)->get(); }

Worker::Worker(const Runtime& rt, const Program& act, int matches, int threads, std::function<std::string()> next_map,
               uint64_t seed)
    : rt_(rt), act_(act), next_map_(std::move(next_map)), in_(act, {"learner", "opponent", "critic"}), rng_(seed) {
  learner_rows_ = actor_batch(in_, "lraw", "lh");
  opponent_rows_ = actor_batch(in_, "oraw", "oh");
  critic_rows_ = critic_batch(in_, "craw");
  l_cap_ = static_cast<int32_t>(in_.spec("lh").dims[0]);
  o_cap_ = static_cast<int32_t>(in_.spec("oh").dims[0]);
  c_cap_ = static_cast<int32_t>(in_.spec("cteam").dims[0]);
  if (matches > l_cap_) throw std::invalid_argument("more matches than the program's learner rows");
  crow_.resize(c_cap_);
  std::vector<rl::MatchSetup> setups(matches);
  for (auto& s : setups) {
    s.map_text = next_map_();
    const int side = static_cast<int>(rng_() % 2);
    const bool self = (rng_() % 2) == 0;
    s.policy[side] = 0;
    s.policy[1 - side] = self ? 0 : 1;
    s.record[side] = true;
    s.record[1 - side] = self;
  }
  collector_ = std::make_unique<rl::Collector>(std::move(setups), threads, false);
  collector_->SetMatchmaking(0.5, rng_());
  out_host_.resize(act_.outputs().size());
  for (size_t i = 0; i < act_.outputs().size(); ++i) out_host_[i].resize(act_.outputs()[i].bytes());
}

void Worker::set_weights(const std::vector<HostArray>& learner, const std::vector<HostArray>& opponent,
                         const std::vector<HostArray>& critic) {
  Weights w;
  w.set(rt_, act_, "learner", learner);
  w.set(rt_, act_, "opponent", opponent);
  w.set(rt_, act_, "critic", critic);
  std::lock_guard<std::mutex> lock(weights_mu_);
  weights_ = std::move(w);
}

void Worker::step() {
  if (collector_->queued_maps() < static_cast<size_t>(collector_->num_matches() / 4)) {
    std::vector<std::string> maps;
    for (int i = 0; i < collector_->num_matches() / 2; ++i) maps.push_back(next_map_());
    collector_->PushMaps(std::move(maps));
  }
  auto t0 = std::chrono::steady_clock::now();
  int32_t counts[3];
  collector_->StepEncode(learner_rows_, l_cap_, opponent_rows_, o_cap_, critic_rows_, c_cap_, 8, crow_.data(),
                         in_.as<int32_t>("cteam"), counts);
  *in_.as<int32_t>("seed") = static_cast<int32_t>(rng_() & 0x7fffffff);
  stats_.encode_us += us_since(t0);
  auto t1 = std::chrono::steady_clock::now();
  std::vector<Buffer> uploaded;
  std::vector<PJRT_Buffer*> args(act_.inputs().size());
  {
    std::lock_guard<std::mutex> lock(weights_mu_);
    for (size_t i = 0; i < args.size(); ++i)
      if (weights_.has(static_cast<int>(i))) args[i] = weights_.get(static_cast<int>(i));
    for (const int i : in_.indices()) {
      uploaded.push_back(upload(rt_, act_.device(), act_.inputs()[i], in_.by_index(i)));
      args[i] = uploaded.back().get();
    }
    std::vector<Buffer> outs = act_.run(args);
    for (size_t i = 0; i < outs.size(); ++i) outs[i].download(out_host_[i].data(), out_host_[i].size());
  }
  stats_.gpu_us += us_since(t1);
  auto t2 = std::chrono::steady_clock::now();
  auto out = [&](const char* name) { return out_host_[act_.output(name)].data(); };
  collector_->StepApply(reinterpret_cast<const int32_t*>(out("command/kind")),
                        reinterpret_cast<const int32_t*>(out("command/nsteps")),
                        reinterpret_cast<const int32_t*>(out("command/dirs")),
                        reinterpret_cast<const int32_t*>(out("command/split")),
                        reinterpret_cast<const int32_t*>(out("command/sonar")),
                        reinterpret_cast<const float*>(out("logp")), reinterpret_cast<float*>(out("h")),
                        reinterpret_cast<const float*>(out("value")));
  stats_.apply_us += us_since(t2);
  stats_.steps += 1;
  stats_.decisions += counts[0] + counts[1];
}

}  // namespace rt
