// The simple1 actor as plain C++ (standard library only): one decision's integer
// records in, the sampled command out. The same function as model.act(half=True)
// in dragons_rl (features.actor_inputs, encode, decode), so the submission and the
// C++ evaluator play exactly what was trained. Weights: the named-arrays file the
// trainer writes (actor.bin).
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace rl {

inline constexpr int kPolicyHidden = 192;
inline constexpr int kPolicyUniforms = 13;   // kind, up to eight tokens (or a split size), four sonar heads
inline constexpr int kSonarSymbols = 257;    // 0 silent, 1 + j sends the uint64 j
inline constexpr int kPolicyMaxSplit = 512;  // split sizes 2..513 are offered

// One decision's records, as EncodeObservation writes them (one row of an ActorBatch).
struct PolicyInput {
  const int32_t* win;        // [289, kTileFields]
  const int32_t* wdest;      // [289, 4]
  const int32_t* grid;       // [64, kGridFields]
  const int32_t* entity;     // [kMaxEntities, kEntityFields]
  const int32_t* segment;    // [kWindowTiles, kSegmentFields]
  const int32_t* status;     // [kStatusFields]
  const int32_t* prev_dirs;  // [8]
  const uint8_t* prev_words; // [32]
  const uint8_t* inbox;      // [k_pad, 8]
  int32_t k_pad;
  const uint8_t* echo;       // [5, e_pad]
  int32_t e_pad;
  const int32_t* echo_len;   // [5]
};

struct PolicyOutput {
  int32_t kind = 0, nsteps = 0, split = 0;
  std::array<int32_t, 8> dirs{};    // 0 past nsteps
  std::array<int32_t, 4> sonar{};   // per direction: 0 silent, 1 + j sends the word j
  float logp = 0;                   // joint log-probability of the command
  std::array<float, kPolicyHidden> h{};
  int32_t draws = 0;                // uniforms consumed
};

class Policy {
 public:
  explicit Policy(const std::string& weights_path);
  // One decision: h_old [192], uniforms [13] in [0, 1).
  PolicyOutput Act(const PolicyInput& in, const float* h_old, const float* uniforms) const;

 private:
  struct Array {
    std::vector<int64_t> dims;
    std::vector<float> data;
  };
  const float* W(const std::string& name, size_t expect) const;
  std::unordered_map<std::string, Array> arrays_;
};

}  // namespace rl
