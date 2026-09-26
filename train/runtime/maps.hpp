// A pool of map texts (written by the build step: dragons_rl/export.py --maps).
// File: records of `<bytes>\n<map text>`.
#pragma once

#include <atomic>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rt {

class MapPool {
 public:
  explicit MapPool(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    size_t n;
    while (f >> n) {
      f.get();
      std::string text(n, '\0');
      f.read(text.data(), static_cast<std::streamsize>(n));
      maps_.push_back(std::move(text));
    }
    if (maps_.empty()) throw std::runtime_error("no maps in " + path);
  }
  size_t size() const { return maps_.size(); }
  // thread-safe: a fixed permutation walk with a random start per call site is enough here
  std::string next() {
    const uint64_t i = next_.fetch_add(0x9e3779b97f4a7c15ULL);
    return maps_[(i >> 17) % maps_.size()];
  }

 private:
  std::vector<std::string> maps_;
  std::atomic<uint64_t> next_{0};
};

}  // namespace rt
