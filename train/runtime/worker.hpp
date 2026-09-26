// A collection worker: one Collector's native step loop on one device, the
// policy an exported `act` program (see dragons_rl/export.py). No Python.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include "collector.hpp"
#include "pjrt.hpp"

namespace rt {

// Host storage for every non-weight input of a program, laid out by its manifest.
class HostInputs {
 public:
  explicit HostInputs(const Program& p, const std::vector<std::string>& weight_prefixes);
  void* data(const std::string& name);
  template <typename T> T* as(const std::string& name) { return reinterpret_cast<T*>(data(name)); }
  const TensorSpec& spec(const std::string& name) const;
  const std::vector<int>& indices() const { return indices_; }   // program input indices held here
  void* by_index(int i) { return storage_.at(i).data(); }

 private:
  const Program& program_;
  std::vector<int> indices_;
  std::vector<std::vector<uint8_t>> storage_;                     // per program input (empty for weights)
};

// Device-resident weights for a program's inputs named <prefix>/<name>.
class Weights {
 public:
  void set(const Runtime& rt, const Program& p, const std::string& prefix, const std::vector<HostArray>& arrays);
  PJRT_Buffer* get(int input) const;
  bool has(int input) const { return by_input_.count(input) > 0; }

 private:
  std::unordered_map<int, std::shared_ptr<Buffer>> by_input_;
};

struct WorkerStats {
  std::atomic<int64_t> steps{0}, decisions{0}, harvested{0};
  std::atomic<int64_t> encode_us{0}, gpu_us{0}, apply_us{0};
};

class Worker {
 public:
  // matches: concurrent matches; maps: a source of map texts (thread-safe)
  Worker(const Runtime& rt, const Program& act, int matches, int threads, std::function<std::string()> next_map,
         uint64_t seed);
  // weights for the learner / opponent / critic prefixes (copied to this worker's device)
  void set_weights(const std::vector<HostArray>& learner, const std::vector<HostArray>& opponent,
                   const std::vector<HostArray>& critic);
  void step();                                    // one native step (all matches)
  std::shared_ptr<rl::Pack> harvest() { return collector_->Harvest(); }
  WorkerStats& stats() { return stats_; }

 private:
  const Runtime& rt_;
  const Program& act_;
  std::function<std::string()> next_map_;
  std::unique_ptr<rl::Collector> collector_;
  HostInputs in_;
  rl::ActorBatch learner_rows_, opponent_rows_;
  rl::CriticBatch critic_rows_;
  int32_t l_cap_, o_cap_, c_cap_;
  std::vector<int32_t> crow_;
  std::mutex weights_mu_;
  Weights weights_;
  std::mt19937_64 rng_;
  std::vector<std::vector<uint8_t>> out_host_;
  WorkerStats stats_;
};

}  // namespace rt
