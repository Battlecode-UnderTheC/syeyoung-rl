// What dragons_model/export.py produced: the programs, the configuration they
// were compiled for, the parameter spec, and a manifest tying them to code.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "agent/features.hpp"
#include "pjrt/pjrt.hpp"

namespace train {

// The shapes the programs were compiled for. Parsed once; nothing else reads
// config files.
struct ModelConfig {
  int32_t rows;              // dragons per act call (envs per collecting device)
  int32_t slots;             // GRU slots per collecting device
  int32_t gather_rows;
  int32_t chunk_turns;       // turns per training chunk
  int32_t minibatch;         // chunks per grad_step
  int32_t learners;          // gradients averaged per apply
  int32_t hidden;            // GRU width
  agent::MemoryViewSpec memory;
};

struct LeafSpec {
  std::string path;          // e.g. "actor/gru_x/w"
  pjrt::DType type;
  pjrt::Shape shape;
};

struct Manifest {
  std::string code_revision; // git revision the programs were exported from
  std::string feature_layout;// FeatureLayout::Describe() at export
};

class Artifacts {
 public:
  // Opens <dir> and checks it against this build: the feature layout must match
  // agent::FeatureLayout for the same spec, and every program's signature must
  // match what train/programs.hpp expects. Throws with the first mismatch.
  static Artifacts Open(const std::string& dir);

  const std::string& dir() const;
  const ModelConfig& config() const;
  const Manifest& manifest() const;
  const std::vector<LeafSpec>& params() const;
  const std::vector<LeafSpec>& adam() const;
  std::string initial_state_path() const;
};

}  // namespace train
