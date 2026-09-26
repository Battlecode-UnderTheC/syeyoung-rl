// Weights as named float arrays, their gradients, and Adam (on the host).
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "pjrt.hpp"

namespace rt {

// Named float32 arrays in a fixed order (the exported programs' leaf order).
struct Params {
  std::vector<HostArray> arrays;

  static Params load(const std::string& path) { return Params{read_arrays(path)}; }
  void save(const std::string& path) const { write_arrays(path, arrays); }
  size_t size() const { return arrays.size(); }
  const std::string& name(size_t i) const { return arrays[i].spec.name; }
  float* data(size_t i) { return reinterpret_cast<float*>(arrays[i].data.data()); }
  const float* data(size_t i) const { return reinterpret_cast<const float*>(arrays[i].data.data()); }
  size_t count(size_t i) const { return arrays[i].data.size() / sizeof(float); }
  int64_t total() const {
    int64_t n = 0;
    for (size_t i = 0; i < size(); ++i) n += static_cast<int64_t>(count(i));
    return n;
  }
  int index(const std::string& name) const {
    for (size_t i = 0; i < arrays.size(); ++i)
      if (arrays[i].spec.name == name) return static_cast<int>(i);
    throw std::invalid_argument("no parameter " + name);
  }
  Params zeros_like() const {
    Params z{arrays};
    for (auto& a : z.arrays) std::fill(a.data.begin(), a.data.end(), 0);
    return z;
  }
};

// Gradients: one float vector per parameter array.
struct Grads {
  std::vector<std::vector<float>> g;
  explicit Grads(const Params& p) : g(p.size()) {
    for (size_t i = 0; i < p.size(); ++i) g[i].assign(p.count(i), 0.0f);
  }
  void zero() {
    for (auto& v : g) std::fill(v.begin(), v.end(), 0.0f);
  }
  void add(const Grads& o) {
    for (size_t i = 0; i < g.size(); ++i)
      for (size_t k = 0; k < g[i].size(); ++k) g[i][k] += o.g[i][k];
  }
  bool finite() const {
    for (const auto& v : g)
      for (const float x : v)
        if (!std::isfinite(x)) return false;
    return true;
  }
  double norm() const {
    double s = 0;
    for (const auto& v : g)
      for (const float x : v) s += static_cast<double>(x) * x;
    return std::sqrt(s);
  }
  // clip by global norm; returns the norm before clipping
  double clip(double limit) {
    const double n = norm();
    if (n > limit && n > 0) {
      const float s = static_cast<float>(limit / n);
      for (auto& v : g)
        for (float& x : v) x *= s;
    }
    return n;
  }
};

struct Adam {
  double lr = 3e-4, b1 = 0.9, b2 = 0.999, eps = 1e-5;
  int64_t t = 0;
  Params m, v;

  void init(const Params& p) {
    m = p.zeros_like();
    v = p.zeros_like();
    t = 0;
  }
  void step(Params& p, const Grads& grads) {
    ++t;
    const double c1 = 1 - std::pow(b1, static_cast<double>(t)), c2 = 1 - std::pow(b2, static_cast<double>(t));
    for (size_t i = 0; i < p.size(); ++i) {
      float* w = p.data(i);
      float* mi = m.data(i);
      float* vi = v.data(i);
      const std::vector<float>& g = grads.g[i];
      for (size_t k = 0; k < g.size(); ++k) {
        mi[k] = static_cast<float>(b1 * mi[k] + (1 - b1) * g[k]);
        vi[k] = static_cast<float>(b2 * vi[k] + (1 - b2) * g[k] * g[k]);
        w[k] = static_cast<float>(w[k] - lr * (mi[k] / c1) / (std::sqrt(vi[k] / c2) + eps));
      }
    }
  }
};

}  // namespace rt
