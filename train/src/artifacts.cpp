#include "artifacts.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace train {

Artifacts Artifacts::Load(const std::string& dir) {
  Artifacts a;
  a.dir = dir;
  {
    std::istringstream in(pjrt::ReadFile(dir + "/config.txt"));
    std::string key;
    int64_t value;
    while (in >> key >> value) a.config[key] = value;
  }
  std::istringstream in(pjrt::ReadFile(dir + "/state.txt"));
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream f(line);
    int index;
    std::string dtype, shape;
    Leaf l;
    f >> index >> l.group >> l.path >> dtype >> shape >> l.offset >> l.bytes;
    l.type = pjrt::ParseDType(dtype);
    if (shape != "-") {
      std::istringstream s(shape);
      for (std::string tok; std::getline(s, tok, ',');) l.dims.push_back(std::stoll(tok));
    }
    if (index != static_cast<int>(a.leaves.size())) throw std::runtime_error("state.txt is out of order");
    (l.group == "params" ? a.num_params : a.num_opt) += 1;
    a.leaves.push_back(std::move(l));
  }
  return a;
}

int64_t Artifacts::Get(const std::string& key) const {
  auto it = config.find(key);
  if (it == config.end()) throw std::runtime_error("config.txt has no " + key);
  return it->second;
}

TrainState TrainState::Load(const pjrt::Runtime& rt, const Artifacts& a, const std::string& bin, int device) {
  const std::string data = pjrt::ReadFile(bin);
  TrainState s;
  s.device = device;
  for (const Leaf& l : a.leaves) {
    if (l.offset + l.bytes > data.size()) throw std::runtime_error(bin + " is shorter than state.txt says");
    pjrt::Buffer b = rt.Upload(device, data.data() + l.offset, l.type, l.dims);
    (l.group == "params" ? s.params : s.opt).push_back(std::move(b));
  }
  return s;
}

void TrainState::Save(const Artifacts& a, const std::string& path) const {
  std::string data;
  size_t k = 0;
  for (const Leaf& l : a.leaves) {
    data.resize(l.offset + l.bytes);
    const pjrt::Buffer& b = k < params.size() ? params[k] : opt[k - params.size()];
    b.ToHost(data.data() + l.offset, l.bytes);
    ++k;
  }
  const std::string tmp = path + ".tmp";
  std::ofstream(tmp, std::ios::binary).write(data.data(), static_cast<std::streamsize>(data.size()));
  if (std::rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error("cannot write " + path);
}

std::vector<pjrt::Buffer> TrainState::CopyParams(int to_device) const {
  std::vector<pjrt::Buffer> out;
  out.reserve(params.size());
  for (const pjrt::Buffer& b : params) out.push_back(b.CopyTo(to_device));
  return out;
}

}  // namespace train
