// The model's input for one turn, encoded into a fixed-layout byte record.
//
// This is the single definition of the model's inputs: the trainer writes it
// for every turn (and stores it for replay), the bot writes it before each
// forward pass, and the JAX side only casts its fields. FeatureLayout is
// exported next to the programs and checked at load.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "agent/actions.hpp"
#include "agent/fact_map.hpp"
#include "agent/observation.hpp"

namespace agent {

// [ASSUMED: option (a)] what the network sees of the FactMap.
struct MemoryViewSpec {
  int near_size = 13;        // tiles a side, full resolution, centred on the head
  int far_size = 13;         // cells a side
  int far_cell = 4;          // tiles per far cell
  // distance fields over the whole known board (breadth-first, respecting known
  // kelp, known portals and bodies last seen), sampled on the near view
  bool distances = true;
};

// One named field of the record: where it is, its element type and shape.
struct Field {
  std::string name;
  enum class Type : uint8_t { U8, I8, I32, F16, F32 } type;
  std::vector<int64_t> shape;
  int64_t offset;            // bytes from the record's start
};

class FeatureLayout {
 public:
  explicit FeatureLayout(const MemoryViewSpec& spec);
  int64_t bytes() const;                          // a multiple of 64
  const Field& Get(const std::string& name) const;
  std::span<const Field> fields() const;
  std::string Describe() const;                   // what export.py writes and the loader compares

 private:
  std::vector<Field> fields_;
  int64_t bytes_;
};

// Writes the turn's features (window channels, scalars, action mask, memory
// views) into `out` (exactly layout.bytes()). The caller has already applied
// FactMap::Observe for this observation.
void EncodeFeatures(const Observation& obs, const FactMap& memory, const FeatureLayout& layout,
                    std::span<uint8_t> out);

}  // namespace agent
