// What dragons_model/export.py wrote: the shared config, the programs, and
// the parameter / Adam leaves; plus the training state held on a device.
#pragma once

#include <map>
#include <string>
#include <vector>

#include "pjrt.hpp"

namespace train {

struct Leaf {
  std::string group;   // "params" or "opt"
  std::string path;
  pjrt::DType type;
  std::vector<int64_t> dims;
  size_t offset, bytes;
};

struct Artifacts {
  std::string dir;
  std::map<std::string, int64_t> config;
  std::vector<Leaf> leaves;   // params first, then opt, in the programs' flat order
  int num_params = 0, num_opt = 0;

  static Artifacts Load(const std::string& dir);
  int64_t Get(const std::string& key) const;
};

// Parameters and optimiser state on one device.
struct TrainState {
  int device = 0;
  std::vector<pjrt::Buffer> params, opt;

  // from a state.bin laid out as state.txt says (the initial one, or a checkpoint)
  static TrainState Load(const pjrt::Runtime& rt, const Artifacts& a, const std::string& bin, int device);
  void Save(const Artifacts& a, const std::string& path) const;   // writes a state.bin
  std::vector<pjrt::Buffer> CopyParams(int to_device) const;
};

}  // namespace train
