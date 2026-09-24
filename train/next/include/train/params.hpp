// Parameters and optimiser state on a device, and checkpoints on disk.
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "pjrt/pjrt.hpp"
#include "train/artifacts.hpp"

namespace train {

// The model's parameter leaves on one device, in the artifacts' order.
class Params {
 public:
  Params(std::vector<pjrt::Buffer> leaves, pjrt::Device device);
  pjrt::Device device() const;
  std::span<const pjrt::Buffer> leaves() const;
  Params CopyTo(pjrt::Device d) const;
  std::vector<pjrt::Buffer> Release() &&;     // for programs that consume them
};

class AdamState {
 public:
  AdamState(std::vector<pjrt::Buffer> leaves, pjrt::Device device);
  pjrt::Device device() const;
  std::span<const pjrt::Buffer> leaves() const;
  std::vector<pjrt::Buffer> Release() &&;
};

struct Progress {
  int64_t iteration = 0;     // the next iteration to run
  int64_t turns = 0;         // dragon-turns collected so far
};

struct Checkpoint {
  Params params;
  AdamState adam;
  Progress progress;
};

// Written to a temporary file, checked, then renamed: a checkpoint on disk is
// always complete.
void SaveCheckpoint(const std::string& path, const Checkpoint& c, const Artifacts& a);
Checkpoint LoadCheckpoint(const std::string& path, const Artifacts& a, const pjrt::Runtime& rt, pjrt::Device d);
Checkpoint InitialCheckpoint(const Artifacts& a, const pjrt::Runtime& rt, pjrt::Device d);

// The newest checkpoint in `dir` that loads; nullopt if none does.
std::optional<std::string> LatestCheckpoint(const std::string& dir, const Artifacts& a);

}  // namespace train
