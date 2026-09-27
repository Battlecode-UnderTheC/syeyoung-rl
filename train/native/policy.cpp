#include "policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "facts.hpp"

namespace rl {

namespace {

constexpr int kW = kWin;                        // 17: the window side, head at its centre
constexpr int kN = kWinTiles;                   // 289
constexpr int kCentre = (kW / 2) * kW + kW / 2;
constexpr int kCols = 56;                       // per-tile features
constexpr int kWidth = 32;
constexpr int kGridFeat = 15;
constexpr int kMaxTok = 8;
constexpr int kBos = 5, kStop = 4;
constexpr float kInf = std::numeric_limits<float>::infinity();
const float kLog501 = std::log(501.0f);
const float kTwoPi = static_cast<float>(2.0 * M_PI);

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float clip01(float x) { return std::min(std::max(x, 0.0f), 1.0f); }
inline float cnt(float x) { return std::min(std::log1p(std::max(x, 0.0f)) / kLog501, 1.0f); }
inline float loglen(float x, float area) { return std::log1p(std::max(x, 0.0f)) / std::log1p(area); }
inline float timer(float x) {
  const float s = x > 0 ? 1.0f : (x < 0 ? -1.0f : 0.0f);
  return s * std::min(std::log1p(std::fabs(x)) / kLog501, 1.0f);
}
inline float age(float t, float now, bool valid) { return valid ? clip01((now - t) / 500.0f) : 0.0f; }
inline void angles(float x, float y, float w, float h, float* out) {
  const float tx = kTwoPi * x / w, ty = kTwoPi * y / h;
  out[0] = std::sin(tx);
  out[1] = std::cos(tx);
  out[2] = std::sin(ty);
  out[3] = std::cos(ty);
}
inline void one_hot(int32_t i, int n, bool valid, float* out) {
  for (int k = 0; k < n; ++k) out[k] = (valid && i == k) ? 1.0f : 0.0f;
}

// float -> IEEE half (round to nearest even) -> float: the model's fp16 layers
float round16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = x & 0x80000000u;
  uint32_t ax = x & 0x7fffffffu;
  if (ax >= 0x7f800000u) return f;                               // inf, nan
  float r;
  if (ax >= 0x477ff000u) {                                        // >= 65520: overflows
    r = kInf;
  } else if (ax < 0x38800000u) {                                  // half subnormal: steps of 2^-24
    const float q = std::ldexp(1.0f, -24);
    r = std::nearbyint(std::fabs(f) / q) * q;
  } else {
    ax += 0xfffu + ((ax >> 13) & 1u);
    ax &= ~0x1fffu;
    std::memcpy(&r, &ax, 4);
  }
  return sign ? -r : r;
}

// y[o] += sum_i x[i] wT[i, o]   (wT [in, out]: the loop over outputs vectorises; zero inputs skipped)
inline void accum(const float* wT, const float* x, int in, int out, float* y) {
  for (int i = 0; i < in; ++i) {
    const float xi = x[i];
    if (xi == 0.0f) continue;
    const float* wr = wT + static_cast<size_t>(i) * out;
    for (int o = 0; o < out; ++o) y[o] += xi * wr[o];
  }
}

// y = b + x W^T, with W given transposed (wT [in, out])
inline void linear(const float* wT, const float* b, const float* x, int in, int out, float* y) {
  for (int o = 0; o < out; ++o) y[o] = b ? b[o] : 0.0f;
  accum(wT, x, in, out, y);
}

void log_softmax(const float* x, const bool* mask, int n, float* out) {
  float m = -kInf;
  for (int i = 0; i < n; ++i)
    if (mask[i]) m = std::max(m, x[i]);
  if (!std::isfinite(m)) m = 0;
  double s = 0;
  for (int i = 0; i < n; ++i)
    if (mask[i]) s += std::exp(x[i] - m);
  const float lse = static_cast<float>(std::log(s > 0 ? s : 1.0)) + m;
  for (int i = 0; i < n; ++i) out[i] = mask[i] ? x[i] - lse : -kInf;
}

// inverse-CDF categorical sample from one uniform (as model.sample)
int sample(const float* logits, const bool* mask, int n, float u) {
  float m = -kInf;
  for (int i = 0; i < n; ++i)
    if (mask[i]) m = std::max(m, logits[i]);
  std::vector<float> p(n, 0.0f);
  float s = 0;
  for (int i = 0; i < n; ++i)
    if (mask[i]) s += (p[i] = std::exp(logits[i] - m));
  float cum = 0;
  std::vector<float> c(n);
  for (int i = 0; i < n; ++i) c[i] = (cum += mask[i] ? p[i] / s : 0.0f);
  const float thr = u * c[n - 1];
  int last = -1;
  for (int i = 0; i < n; ++i) {
    if (!mask[i]) continue;
    if (c[i] > thr) return i;
    last = i;
  }
  return last;
}

// 3x3 "SAME" convolution of x [h, w, cin] with wt [3, 3, cin, cout] (rearranged), stride s
void conv(const float* x, int h, int w, int cin, const float* wt, const float* b, int cout, int s, float* y) {
  const int oh = (h + s - 1) / s, ow = (w + s - 1) / s;
  const int ph = std::max((oh - 1) * s + 3 - h, 0), pw = std::max((ow - 1) * s + 3 - w, 0);
  const int top = ph / 2, left = pw / 2;
  for (int i = 0; i < oh; ++i)
    for (int j = 0; j < ow; ++j) {
      float* out = y + (static_cast<size_t>(i) * ow + j) * cout;
      for (int o = 0; o < cout; ++o) out[o] = b[o];
      for (int dy = 0; dy < 3; ++dy) {
        const int r = i * s + dy - top;
        if (r < 0 || r >= h) continue;
        for (int dx = 0; dx < 3; ++dx) {
          const int c = j * s + dx - left;
          if (c < 0 || c >= w) continue;
          const float* xi = x + (static_cast<size_t>(r) * w + c) * cin;
          accum(wt + static_cast<size_t>(dy * 3 + dx) * cin * cout, xi, cin, cout, out);
        }
      }
    }
}

// GRU (model.GRU): w_i [3, H, I], w_h [3, H, H], b_i, b_h [3, H]
void gru(const float* wi, const float* bi, const float* wh, const float* bh, const float* x, int in, const float* h,
         int H, float* out) {
  std::vector<float> gi(3 * H), gh(3 * H);
  linear(wi, bi, x, in, 3 * H, gi.data());
  linear(wh, bh, h, H, 3 * H, gh.data());
  for (int k = 0; k < H; ++k) {
    const float r = sigmoid(gi[k] + gh[k]);
    const float z = sigmoid(gi[H + k] + gh[H + k]);
    const float n = std::tanh(gi[2 * H + k] + r * gh[2 * H + k]);
    out[k] = (1.0f - z) * n + z * h[k];
  }
}

void masked_mean_max(const std::vector<float>& x, const std::vector<char>& mask, int rows, int d, float* mean,
                     float* mx) {
  int n = 0;
  for (int j = 0; j < d; ++j) {
    mean[j] = 0;
    mx[j] = -kInf;
  }
  for (int r = 0; r < rows; ++r) {
    if (!mask[r]) continue;
    ++n;
    for (int j = 0; j < d; ++j) {
      mean[j] += x[static_cast<size_t>(r) * d + j];
      mx[j] = std::max(mx[j], x[static_cast<size_t>(r) * d + j]);
    }
  }
  for (int j = 0; j < d; ++j) {
    mean[j] = n ? mean[j] / static_cast<float>(n) : 0.0f;
    if (!n) mx[j] = 0;
  }
}

// physical_vector: kind one-hot, eight direction one-hots, split/A, length/A
void physical_vector(int kind, const int32_t* dirs, int split, float length, float area, float* C) {
  const bool is_split = kind == 1;
  one_hot(kind, 2, true, C);
  for (int t = 0; t < kMaxTok; ++t) one_hot(dirs[t], 4, dirs[t] >= 0 && !is_split, C + 2 + 4 * t);
  C[34] = is_split ? static_cast<float>(split) / area : 0.0f;
  C[35] = length / area;
}

}  // namespace

// ---- weights ----------------------------------------------------------------------------------------

Policy::Policy(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  Load(f, path);
}

Policy Policy::FromBytes(const char* data, size_t size) {
  struct View : std::streambuf {                   // reads the bytes in place (no copy)
    View(const char* b, size_t n) {
      char* p = const_cast<char*>(b);
      setg(p, p, p + n);
    }
  } view(data, size);
  std::istream f(&view);
  Policy p;
  p.Load(f, "the compiled-in weights");
  return p;
}

void Policy::Load(std::istream& f, const std::string& path) {
  auto take = [&](void* dst, size_t n) {
    if (!f.read(static_cast<char*>(dst), static_cast<std::streamsize>(n))) throw std::runtime_error("truncated " + path);
  };
  uint32_t count;
  take(&count, 4);
  for (uint32_t a = 0; a < count; ++a) {
    uint32_t len;
    take(&len, 4);
    std::string name(len, '\0');
    take(name.data(), len);
    uint8_t code;
    uint32_t rank;
    take(&code, 1);
    take(&rank, 4);
    Array arr;
    arr.dims.resize(rank);
    take(arr.dims.data(), 8 * rank);
    size_t n = 1;
    for (const int64_t d : arr.dims) n *= static_cast<size_t>(d);
    if (code != 0) throw std::runtime_error(path + ": " + name + " is not float32");
    arr.data.resize(n);
    take(arr.data.data(), 4 * n);
    arrays_.emplace(std::move(name), std::move(arr));
  }
  // prepared forms: conv weights as [out, dy, dx, in] rounded to fp16, the tile MLP's second layer in fp16
  for (const char* conv_name : {"local/conv1", "local/conv2", "near/conv1", "near/conv2"}) {
    const Array& w = arrays_.at(std::string(conv_name) + "/w");
    const int64_t co = w.dims[0], ci = w.dims[1];
    Array t;
    t.dims = {3, 3, ci, co};
    t.data.resize(w.data.size());
    for (int64_t o = 0; o < co; ++o)
      for (int64_t i = 0; i < ci; ++i)
        for (int dy = 0; dy < 3; ++dy)
          for (int dx = 0; dx < 3; ++dx)
            t.data[((dy * 3 + dx) * ci + i) * co + o] = round16(w.data[((o * ci + i) * 3 + dy) * 3 + dx]);
    arrays_.emplace(std::string(conv_name) + "/w16", std::move(t));
  }
  Array t2 = arrays_.at("tile/l2/w");
  for (float& v : t2.data) v = round16(v);
  arrays_.emplace("tile/l2/w16", std::move(t2));
  // every layer's matrix transposed ([in, out]; a GRU's [3, H, I] as [I, 3H]): name + "T"
  std::vector<std::pair<std::string, Array>> add;
  for (const auto& [name, a] : arrays_) {
    const bool matrix = name.size() > 2 && (name.compare(name.size() - 2, 2, "/w") == 0 || name.find("/w_") != std::string::npos ||
                                            name.compare(name.size() - 4, 4, "/w16") == 0);
    if (!matrix || a.dims.size() < 2 || a.dims.size() > 3) continue;
    const int64_t in = a.dims.back(), out = static_cast<int64_t>(a.data.size()) / in;
    Array t;
    t.dims = {in, out};
    t.data.resize(a.data.size());
    for (int64_t o = 0; o < out; ++o)
      for (int64_t i = 0; i < in; ++i) t.data[i * out + o] = a.data[o * in + i];
    add.emplace_back(name + "T", std::move(t));
  }
  for (auto& [name, t] : add) arrays_.emplace(std::move(name), std::move(t));
}

const float* Policy::W(const std::string& name, size_t expect) const {
  auto it = arrays_.find(name);
  if (it == arrays_.end()) throw std::runtime_error("no weight " + name);
  if (expect && it->second.data.size() != expect)
    throw std::runtime_error("weight " + name + " has " + std::to_string(it->second.data.size()) + " values, expected " +
                             std::to_string(expect));
  return it->second.data.data();
}

// ---- one decision -----------------------------------------------------------------------------------

PolicyOutput Policy::Act(const PolicyInput& in, const float* h_old, const float* u) const {
  const int32_t* st = in.status;
  const float Wd = static_cast<float>(st[kSWidth]), Hd = static_cast<float>(st[kSHeight]);
  const float A = Wd * Hd;
  const int32_t Lint = st[kSLength];
  const float Lf = static_cast<float>(Lint);
  const float now = static_cast<float>(st[kSRound]);
  const float hx = static_cast<float>(st[kSHeadX]), hy = static_cast<float>(st[kSHeadY]);

  // ---- the window's tiles: 56 feature columns, and what D() and the decoder read ----
  std::vector<float> X(static_cast<size_t>(kN) * kCols, 0.0f);
  std::vector<char> valid(kN), seen(kN);
  std::vector<std::array<int32_t, 4>> etype(kN), J(kN);
  std::vector<int32_t> flags(kN);
  for (int t = 0; t < kN; ++t) {
    const int32_t* r = in.win + static_cast<size_t>(t) * kTileFields;
    const int32_t* dest = in.wdest + static_cast<size_t>(t) * 4;
    valid[t] = r[kTValid] == 1;
    seen[t] = r[kTSeenRound] >= 0;
    flags[t] = r[kTFlags];
    for (int d = 0; d < 4; ++d) {
      etype[t][d] = (r[kTEdges] >> (2 * d)) & 3;
      J[t][d] = valid[t] ? dest[d] : -1;
    }
    if (!valid[t]) continue;
    float* c = X.data() + static_cast<size_t>(t) * kCols;
    const bool sn = seen[t];
    const float sm = sn ? 1.0f : 0.0f;
    const float seen_round = static_cast<float>(r[kTSeenRound]);
    const int32_t countdown = r[kTCountdown];
    const int32_t visits = r[kTVisits], visit_round = r[kTVisitRound];
    const int32_t index = r[kTBodyIndex];
    const bool known_body = index >= 0;
    const int32_t prev_round = r[kTPrevRound];
    const bool prev_valid = prev_round >= 0 && sn;
    const float pvf = prev_valid ? 1.0f : 0.0f;
    const int32_t prev_cd = r[kTPrevCountdown];
    const int32_t occ = r[kTOcc], order = r[kTOccOrder], facing = r[kTOccFacing];
    c[0] = sm;
    c[1] = static_cast<float>(r[kTVisible]);
    c[2] = age(seen_round, now, sn);
    c[3] = static_cast<float>(r[kTPearl]) * sm;
    c[4] = (countdown != -1 ? 1.0f : 0.0f) * sm;
    c[5] = timer(static_cast<float>(countdown)) * sm;
    c[6] = visits > 0 ? 1.0f : 0.0f;
    c[7] = cnt(static_cast<float>(visits));
    c[8] = age(static_cast<float>(visit_round), now, visit_round >= 0);
    c[9] = (known_body && index == 0) ? 1.0f : 0.0f;
    c[10] = known_body ? 1.0f : 0.0f;
    c[11] = known_body ? static_cast<float>(index) / static_cast<float>(std::max(Lint - 1, 1)) : 0.0f;
    c[12] = (known_body && index == Lint - 1) ? 1.0f : 0.0f;
    c[13] = pvf;
    c[14] = age(static_cast<float>(prev_round), now, prev_valid);
    c[15] = static_cast<float>(r[kTPrevPearl]) * pvf;
    c[16] = (prev_cd != -1 ? 1.0f : 0.0f) * pvf;
    c[17] = timer(static_cast<float>(prev_cd)) * pvf;
    c[18] = prev_valid ? clip01((seen_round - static_cast<float>(prev_round)) / 500.0f) : 0.0f;
    c[19] = (order == 1 ? 1.0f : 0.0f) * sm;
    c[20] = (order == 2 ? 1.0f : 0.0f) * sm;
    for (int k = 0; k < 7; ++k) c[21 + k] = (sn && occ > 0 && occ - 1 == k) ? 1.0f : 0.0f;
    for (int k = 0; k < 4; ++k) c[28 + k] = (sn && facing >= 0 && facing == k) ? 1.0f : 0.0f;
    for (int d = 0; d < 4; ++d)
      for (int k = 0; k < 4; ++k) c[32 + 4 * d + k] = etype[t][d] == k ? 1.0f : 0.0f;
    for (int d = 0; d < 4; ++d) c[48 + d] = dest[d] != -1 ? 1.0f : 0.0f;
    angles(static_cast<float>(t % kW - kW / 2), static_cast<float>(t / kW - kW / 2), Wd, Hd, c + 52);
  }
  // D(t): [4, 12] per direction
  auto D = [&](int t, float* out) {
    const int32_t* dest = in.wdest + static_cast<size_t>(t) * 4;
    const float v = valid[t] ? 1.0f : 0.0f;
    for (int d = 0; d < 4; ++d) {
      float* o = out + 12 * d;
      for (int k = 0; k < 4; ++k) o[k] = (k == d ? 1.0f : 0.0f) * v;
      for (int k = 0; k < 4; ++k) o[4 + k] = (etype[t][d] == k ? 1.0f : 0.0f) * v;
      o[8] = (dest[d] != -1 ? 1.0f : 0.0f) * v;
      o[9] = (((flags[t] >> d) & 1) == 1 && etype[t][d] == 1 ? 1.0f : 0.0f) * v;
      o[10] = (((flags[t] >> (4 + d)) & 1) == 1 && etype[t][d] == 3 ? 1.0f : 0.0f) * v;
      o[11] = (dest[d] >= 0 && seen[std::min(dest[d], kN - 1)] ? 1.0f : 0.0f) * v;
    }
  };

  // ---- tile states: the tile MLP (first layer fp32, second fp16 with fp32 accumulation) ----
  std::vector<float> z(static_cast<size_t>(kN) * kWidth);
  {
    const float* w1 = W("tile/l1/wT", kWidth * kCols);
    const float* b1 = W("tile/l1/b", kWidth);
    const float* w2 = W("tile/l2/w16T", kWidth * kWidth);
    const float* b2 = W("tile/l2/b", kWidth);
    float h1[kWidth];
    for (int t = 0; t < kN; ++t) {
      linear(w1, b1, X.data() + static_cast<size_t>(t) * kCols, kCols, kWidth, h1);
      for (float& v : h1) v = round16(silu(v));
      float* zt = z.data() + static_cast<size_t>(t) * kWidth;
      linear(w2, b2, h1, kWidth, kWidth, zt);
      for (int k = 0; k < kWidth; ++k) zt[k] = silu(zt[k]);
    }
  }

  // ---- local: the 7x7 crop around the head, two convs ----
  float local[64];
  {
    std::vector<float> crop(49 * kCols), y1(49 * kWidth), y2(49 * kWidth);
    for (int i = 0; i < 7; ++i)
      for (int j = 0; j < 7; ++j) {
        const int t = (i - 3 + kW / 2) * kW + (j - 3 + kW / 2);
        for (int k = 0; k < kCols; ++k) crop[(i * 7 + j) * kCols + k] = round16(X[static_cast<size_t>(t) * kCols + k]);
      }
    conv(crop.data(), 7, 7, kCols, W("local/conv1/w16", 0), W("local/conv1/b", kWidth), kWidth, 1, y1.data());
    for (float& v : y1) v = round16(silu(v));
    conv(y1.data(), 7, 7, kWidth, W("local/conv2/w16", 0), W("local/conv2/b", kWidth), kWidth, 1, y2.data());
    for (float& v : y2) v = silu(v);
    float feat[96];
    for (int k = 0; k < kWidth; ++k) {
      feat[k] = y2[(3 * 7 + 3) * kWidth + k];
      float s = 0, m = -kInf;
      for (int p = 0; p < 49; ++p) {
        s += y2[p * kWidth + k];
        m = std::max(m, y2[p * kWidth + k]);
      }
      feat[32 + k] = s / 49.0f;
      feat[64 + k] = m;
    }
    linear(W("local/out/wT", 64 * 96), W("local/out/b", 64), feat, 96, 64, local);
    for (float& v : local) v = silu(v);
  }

  // ---- near: the window's central 15x15 of tile states, two stride-2 convs ----
  float near[64];
  {
    std::vector<float> x(15 * 15 * kWidth), y1(8 * 8 * kWidth), y2(4 * 4 * kWidth);
    for (int i = 0; i < 15; ++i)
      for (int j = 0; j < 15; ++j) {
        const int t = (i - 7 + kW / 2) * kW + (j - 7 + kW / 2);
        for (int k = 0; k < kWidth; ++k) x[(i * 15 + j) * kWidth + k] = round16(z[static_cast<size_t>(t) * kWidth + k]);
      }
    conv(x.data(), 15, 15, kWidth, W("near/conv1/w16", 0), W("near/conv1/b", kWidth), kWidth, 2, y1.data());
    for (float& v : y1) v = round16(silu(v));
    conv(y1.data(), 8, 8, kWidth, W("near/conv2/w16", 0), W("near/conv2/b", kWidth), kWidth, 2, y2.data());
    for (float& v : y2) v = silu(v);
    linear(W("near/out/wT", 64 * 512), W("near/out/b", 64), y2.data(), 512, 64, near);
    for (float& v : near) v = silu(v);
  }

  // ---- the whole map: 8x8 head-relative cells ----
  float glob[64];
  {
    const float* pos = W("grid/position", 64 * 8);
    const float* w1 = W("grid/cell/l1/wT", 32 * (kGridFeat + 8));
    const float* b1 = W("grid/cell/l1/b", 32);
    const float* w2 = W("grid/cell/l2/wT", 32 * 32);
    const float* b2 = W("grid/cell/l2/b", 32);
    std::vector<float> cells(64 * 32);
    for (int c = 0; c < 64; ++c) {
      const int32_t* g = in.grid + static_cast<size_t>(c) * kGridFields;
      const float tiles = static_cast<float>(g[kGTiles]);
      const float tm = std::max(tiles, 1.0f);
      const float on = tiles > 0 ? 1.0f : 0.0f;
      const float n_seen = static_cast<float>(g[kGSeen]), n_vis = static_cast<float>(g[kGVisited]);
      float f[kGridFeat + 8];
      f[0] = on;
      f[1] = tiles / 64.0f;
      f[2] = n_seen / tm;
      f[3] = static_cast<float>(g[kGVisible]) / tm;
      f[4] = cnt(static_cast<float>(g[kGPearl]));
      f[5] = cnt(static_cast<float>(g[kGBed]));
      f[6] = n_vis / tm;
      f[7] = static_cast<float>(g[kGSelfBody]) / tm;
      f[8] = cnt(static_cast<float>(g[kGAlly]));
      f[9] = cnt(static_cast<float>(g[kGEnemy]));
      f[10] = static_cast<float>(g[kGEmpty]) / tm;
      f[11] = static_cast<float>(g[kGKelp]) / std::max(2.0f * tiles, 1.0f);
      f[12] = cnt(static_cast<float>(g[kGPortal]));
      f[13] = n_seen > 0 ? clip01((now * n_seen - static_cast<float>(g[kGSeenRoundSum])) / std::max(n_seen, 1.0f) / 500.0f)
                         : 0.0f;
      f[14] = n_vis > 0 ? clip01((now * n_vis - static_cast<float>(g[kGVisitRoundSum])) / std::max(n_vis, 1.0f) / 500.0f)
                        : 0.0f;
      for (int k = 0; k < kGridFeat; ++k) f[k] *= on;
      for (int k = 0; k < 8; ++k) f[kGridFeat + k] = pos[c * 8 + k];
      float h1[32];
      linear(w1, b1, f, kGridFeat + 8, 32, h1);
      for (float& v : h1) v = silu(v);
      linear(w2, b2, h1, 32, 32, cells.data() + c * 32);
      for (int k = 0; k < 32; ++k) cells[c * 32 + k] = silu(cells[c * 32 + k]);
    }
    linear(W("grid/out/wT", 64 * 2048), W("grid/out/b", 64), cells.data(), 2048, 64, glob);
    for (float& v : glob) v = silu(v);
  }

  // ---- entities in view ----
  float ents[64];
  {
    std::vector<float> U(static_cast<size_t>(kMaxEntities) * 20, 0.0f), seg_sum(static_cast<size_t>(kMaxEntities) * 4, 0.0f);
    std::vector<char> ev(kMaxEntities);
    for (int s = 0; s < kWindowTiles; ++s) {
      const int32_t* g = in.segment + static_cast<size_t>(s) * kSegmentFields;
      if (g[3] != 1 || g[0] < 0 || g[0] >= kMaxEntities) continue;
      float a[4];
      angles(static_cast<float>(g[1]) - hx, static_cast<float>(g[2]) - hy, Wd, Hd, a);
      for (int k = 0; k < 4; ++k) seg_sum[g[0] * 4 + k] += a[k];
    }
    const float* w1 = W("entity/l1/wT", 32 * 20);
    const float* b1 = W("entity/l1/b", 32);
    const float* w2 = W("entity/l2/wT", 32 * 32);
    const float* b2 = W("entity/l2/b", 32);
    std::vector<float> e(static_cast<size_t>(kMaxEntities) * 32, 0.0f);
    for (int r = 0; r < kMaxEntities; ++r) {
      const int32_t* g = in.entity + static_cast<size_t>(r) * kEntityFields;
      ev[r] = g[kEValid] == 1;
      if (!ev[r]) continue;
      float* x = U.data() + static_cast<size_t>(r) * 20;
      const int32_t count = g[kECount];
      const bool head_vis = g[kEHeadVisible] == 1;
      one_hot(g[kERelation], 3, true, x);
      one_hot(g[kEOrder], 3, true, x + 3);
      x[6] = static_cast<float>(count) / 49.0f;
      x[7] = head_vis ? 1.0f : 0.0f;
      angles(static_cast<float>(g[kEHeadX]) - hx, static_cast<float>(g[kEHeadY]) - hy, Wd, Hd, x + 8);
      for (int k = 0; k < 4; ++k) x[8 + k] *= x[7];
      one_hot(g[kEHeadFacing], 4, head_vis, x + 12);
      for (int k = 0; k < 4; ++k) x[16 + k] = seg_sum[r * 4 + k] / static_cast<float>(std::max(count, 1));
      float h1[32];
      linear(w1, b1, x, 20, 32, h1);
      for (float& v : h1) v = silu(v);
      linear(w2, b2, h1, 32, 32, e.data() + static_cast<size_t>(r) * 32);
      for (int k = 0; k < 32; ++k) e[r * 32 + k] = silu(e[r * 32 + k]);
    }
    float mm[64];
    masked_mean_max(e, ev, kMaxEntities, 32, mm, mm + 32);
    linear(W("readout/entities/wT", 64 * 64), W("readout/entities/b", 64), mm, 64, 64, ents);
    for (float& v : ents) v = silu(v);
  }

  // ---- status ----
  float status[32];
  {
    const bool prev_ok = st[kSPrevValid] == 1;
    const float pv = prev_ok ? 1.0f : 0.0f;
    const bool prev_move = prev_ok && st[kSPrevKind] == 0, prev_split = prev_ok && st[kSPrevKind] == 1;
    const float prev_len = static_cast<float>(st[kSPrevLength]);
    const int32_t turns = st[kSTurnCount];
    float s[32];
    s[0] = Lf / A;
    s[1] = loglen(Lf, A);
    s[2] = now / 499.0f;
    s[3] = (500.0f - now) / 500.0f;
    s[4] = Wd / 64.0f;
    s[5] = Hd / 64.0f;
    s[6] = static_cast<float>(st[kSUnitCount]) / static_cast<float>(st[kSUnitLimit]);
    s[7] = static_cast<float>(st[kSUnitLimit]) / 64.0f;
    s[8] = static_cast<float>(st[kSKnownBody]) / Lf;
    s[9] = static_cast<float>(st[kSTailKnown]);
    s[10] = turns == 0 ? 1.0f : 0.0f;
    s[11] = static_cast<float>(turns) / 500.0f;
    one_hot(st[kSFacing], 4, true, s + 12);
    angles(hx, hy, Wd, Hd, s + 16);
    s[20] = pv;
    s[21] = prev_move ? 1.0f : 0.0f;
    s[22] = prev_split ? 1.0f : 0.0f;
    s[23] = pv * static_cast<float>(st[kSPrevSteps]) / 8.0f;
    s[24] = pv * static_cast<float>(st[kSPrevSplit]) / std::max(prev_len, 1.0f);
    s[25] = pv * (Lf - prev_len) / A;
    s[26] = static_cast<float>(st[kSSeenTiles]) / A;
    s[27] = static_cast<float>(st[kSKnownEdges]) / (2.0f * A);
    s[28] = cnt(static_cast<float>(st[kSInboxCount]));
    s[29] = static_cast<float>(st[kSEchoPresent]);
    s[30] = static_cast<float>(st[kSHeadBed]);
    s[31] = static_cast<float>(st[kSHeadPearl]);
    float h1[32];
    linear(W("status/l1/wT", 32 * 32), W("status/l1/b", 32), s, 32, 32, h1);
    for (float& v : h1) v = silu(v);
    linear(W("status/l2/wT", 32 * 32), W("status/l2/b", 32), h1, 32, 32, status);
    for (float& v : status) v = silu(v);
  }

  // ---- words: byte + position embeddings, an MLP ----
  const float* byte_emb = W("word/byte", 256 * 8);
  auto encode_word = [&](const uint8_t* bytes, float* out) {
    const float* pos = W("word/position", 64);
    float e[64], h1[32];
    for (int j = 0; j < 8; ++j)
      for (int k = 0; k < 8; ++k) e[j * 8 + k] = byte_emb[bytes[j] * 8 + k] + pos[j * 8 + k];
    linear(W("word/mlp/l1/wT", 32 * 64), W("word/mlp/l1/b", 32), e, 64, 32, h1);
    for (float& v : h1) v = silu(v);
    linear(W("word/mlp/l2/wT", 32 * 32), W("word/mlp/l2/b", 32), h1, 32, 32, out);
    for (int k = 0; k < 32; ++k) out[k] = silu(out[k]);
  };

  float inbox[32];
  {
    const int32_t count = st[kSInboxCount];
    const int n = std::min(count, in.k_pad);
    std::vector<float> words(static_cast<size_t>(std::max(n, 1)) * 32);
    std::vector<char> mask(std::max(n, 1), 0);
    for (int w = 0; w < n; ++w) {
      encode_word(in.inbox + static_cast<size_t>(w) * 8, words.data() + static_cast<size_t>(w) * 32);
      mask[w] = 1;
    }
    float x[66];
    masked_mean_max(words, mask, std::max(n, 1), 32, x, x + 32);
    x[64] = cnt(static_cast<float>(count));
    x[65] = count == 0 ? 1.0f : 0.0f;
    linear(W("inbox/wT", 32 * 66), W("inbox/b", 32), x, 66, 32, inbox);
    for (float& v : inbox) v = silu(v);
  }

  float echo[32];
  {
    const float* pos = W("echo/position", 32 * 8);
    const float* w1 = W("echo/mlp/l1/wT", 16 * 8);
    const float* b1 = W("echo/mlp/l1/b", 16);
    const float* w2 = W("echo/mlp/l2/wT", 16 * 16);
    const float* b2 = W("echo/mlp/l2/b", 16);
    float x[85];
    for (int f = 0; f < 5; ++f) {
      const int n = std::min(in.echo_len[f], in.e_pad);
      float acc[16] = {0};
      for (int j = 0; j < n; ++j) {
        float e[8], h1[16], h2[16];
        const uint8_t b = in.echo[static_cast<size_t>(f) * in.e_pad + j];
        for (int k = 0; k < 8; ++k) e[k] = byte_emb[b * 8 + k] + pos[j * 8 + k];
        linear(w1, b1, e, 8, 16, h1);
        for (float& v : h1) v = silu(v);
        linear(w2, b2, h1, 16, 16, h2);
        for (int k = 0; k < 16; ++k) acc[k] += silu(h2[k]);
      }
      for (int k = 0; k < 16; ++k) x[f * 16 + k] = acc[k] / static_cast<float>(std::max(n, 1));
    }
    for (int f = 0; f < 5; ++f) x[80 + f] = static_cast<float>(st[kSEchoPresent] == 1);
    linear(W("echo/out/wT", 32 * 85), W("echo/out/b", 32), x, 85, 32, echo);
    for (float& v : echo) v = silu(v);
  }

  // ---- the previous command ----
  const float area = A;
  float previous[32];
  {
    const bool prev_ok = st[kSPrevValid] == 1;
    const float pv = prev_ok ? 1.0f : 0.0f;
    const bool prev_split = prev_ok && st[kSPrevKind] == 1;
    float C[36], phys[32];
    physical_vector(prev_split ? 1 : 0, in.prev_dirs, st[kSPrevSplit], static_cast<float>(st[kSPrevLength]), area, C);
    for (float& v : C) v *= pv;
    linear(W("physical/wT", 32 * 36), W("physical/b", 32), C, 36, 32, phys);
    for (float& v : phys) v = silu(v);
    float x[97];
    std::memcpy(x, phys, sizeof phys);
    const int32_t mask = st[kSPrevSonarMask];
    const float* sw = W("prev_sonar/wT", 16 * 37);
    const float* sb = W("prev_sonar/b", 16);
    for (int d = 0; d < 4; ++d) {
      const float flag = (((mask >> d) & 1) == 1 ? 1.0f : 0.0f) * pv;
      float word[32], y[37], o[16];
      encode_word(in.prev_words + 8 * d, word);
      for (int k = 0; k < 32; ++k) y[k] = flag * word[k];
      y[32] = flag;
      for (int k = 0; k < 4; ++k) y[33 + k] = k == d ? 1.0f : 0.0f;
      linear(sw, sb, y, 37, 16, o);
      for (int k = 0; k < 16; ++k) x[32 + 16 * d + k] = silu(o[k]);
    }
    x[96] = pv;
    linear(W("previous/wT", 32 * 97), W("previous/b", 32), x, 97, 32, previous);
    for (float& v : previous) v = silu(v) * pv;
  }

  // ---- readout at the head, fusion, GRU ----
  const float* unknown = W("readout/unknown", kWidth);
  float directional[128];
  {
    float Dh[48];
    D(kCentre, Dh);
    for (int d = 0; d < 4; ++d) {
      float x[44];
      const int32_t jh = J[kCentre][d];
      const float* v = jh >= 0 ? z.data() + static_cast<size_t>(std::min(jh, kN - 1)) * kWidth : unknown;
      std::memcpy(x, v, kWidth * sizeof(float));
      std::memcpy(x + 32, Dh + 12 * d, 12 * sizeof(float));
      linear(W("readout/direction/wT", 32 * 44), W("readout/direction/b", 32), x, 44, 32, directional + 32 * d);
    }
    for (float& v : directional) v = silu(v);
  }
  float fused_in[512], fused[kPolicyHidden];
  {
    float* p = fused_in;
    for (const auto& [src, n] : std::initializer_list<std::pair<const float*, int>>{
             {local, 64}, {near, 64}, {glob, 64}, {ents, 64}, {status, 32}, {inbox, 32}, {echo, 32}, {previous, 32},
             {directional, 128}}) {
      std::memcpy(p, src, n * sizeof(float));
      p += n;
    }
    linear(W("fusion/wT", kPolicyHidden * 512), W("fusion/b", kPolicyHidden), fused_in, 512, kPolicyHidden, fused);
    for (float& v : fused) v = silu(v);
  }
  PolicyOutput out;
  float* h = out.h.data();
  gru(W("gru/w_iT", 3 * 192 * 192), W("gru/b_i", 3 * 192), W("gru/w_hT", 3 * 192 * 192), W("gru/b_h", 3 * 192), fused,
      kPolicyHidden, h_old, kPolicyHidden, h);

  // ---- decode: kind ----
  const bool split_ok = Lint >= 4 && st[kSUnitCount] < st[kSUnitLimit];
  const bool kind_mask[2] = {true, split_ok};
  float kind_logits[2], kind_lp[2];
  linear(W("kind/wT", 2 * 192), W("kind/b", 2), h, 192, 2, kind_logits);
  log_softmax(kind_logits, kind_mask, 2, kind_lp);
  out.kind = sample(kind_logits, kind_mask, 2, u[0]);
  float logp = kind_lp[out.kind];

  std::array<int32_t, kMaxTok> dirs;
  dirs.fill(-1);
  int phys_draws = 1;
  if (out.kind == 1) {
    // SPLIT n = 2..L-2: the h part of the scorer's first layer once
    const float* w1 = W("split/l1/wT", 64 * 198);
    const float* b1 = W("split/l1/b", 64);
    const float* w2 = W("split/l2/wT", 64);
    const float* b2 = W("split/l2/b", 1);
    float base[64];
    linear(w1, b1, h, 192, 64, base);
    std::vector<float> logits(kPolicyMaxSplit, 0.0f), lp(kPolicyMaxSplit);
    std::unique_ptr<bool[]> mask(new bool[kPolicyMaxSplit]);
    for (int idx = 0; idx < kPolicyMaxSplit; ++idx) {
      const int n = idx + 2;
      mask[idx] = n <= Lint - 2;
      if (!mask[idx]) continue;
      const float nf = static_cast<float>(n);
      const float x6[6] = {nf / A, (Lf - nf) / A, nf / Lf, (Lf - nf) / Lf, loglen(nf, A), loglen(Lf - nf, A)};
      float hid[64];
      std::memcpy(hid, base, sizeof hid);
      accum(w1 + 192 * 64, x6, 6, 64, hid);
      float s2 = b2[0];
      for (int o = 0; o < 64; ++o) s2 += w2[o] * silu(hid[o]);
      logits[idx] = s2;
    }
    log_softmax(logits.data(), mask.get(), kPolicyMaxSplit, lp.data());
    const int idx = sample(logits.data(), mask.get(), kPolicyMaxSplit, u[1]);
    logp += lp[idx];
    out.split = idx + 2;
  } else {
    // MOVE: autoregressive direction tokens, walking the window from the head
    const float* dw1 = W("move/direction/l1/wT", 64 * 314);
    const float* db1 = W("move/direction/l1/b", 64);
    const float* dw2 = W("move/direction/l2/wT", 64);
    const float* db2 = W("move/direction/l2/b", 1);
    const float* sw1 = W("move/stop/l1/wT", 64 * 266);
    const float* sb1 = W("move/stop/l1/b", 64);
    const float* sw2 = W("move/stop/l2/wT", 64);
    const float* sb2 = W("move/stop/l2/b", 1);
    const float* tok_emb = W("move/token", 6 * 8);
    // the h_new parts of both scorers' first layers, once
    float dbase[64], sbase[64];
    linear(dw1 + 64 * 64, db1, h, 192, 64, dbase);        // rows 64..255 of the inputs: h_new
    linear(sw1 + 64 * 64, sb1, h, 192, 64, sbase);
    float r[64], r0[64];
    linear(W("move/init/wT", 64 * 192), W("move/init/b", 64), h, 192, 64, r0);
    for (int k = 0; k < 64; ++k) r[k] = std::tanh(r0[k]);
    int pos = kCentre, heading = st[kSFacing], prev_tok = kBos, n_dirs = 0;
    bool known = true;
    int t = 0;
    for (; t < kMaxTok; ++t) {
      float q[10] = {static_cast<float>(n_dirs) / 8.0f, known ? 1.0f : 0.0f, 0, 0, 0, 0, 0, 0, 0, 0};
      if (known) angles(static_cast<float>(pos % kW - kW / 2), static_cast<float>(pos / kW - kW / 2), Wd, Hd, q + 2);
      q[6 + std::min(std::max(heading, 0), 3)] = (heading >= 0 && heading < 4) ? 1.0f : 0.0f;
      float gx[18], rn[64];
      std::memcpy(gx, tok_emb + prev_tok * 8, 8 * sizeof(float));
      std::memcpy(gx + 8, q, 10 * sizeof(float));
      gru(W("move/gru/w_iT", 3 * 64 * 18), W("move/gru/b_i", 3 * 64), W("move/gru/w_hT", 3 * 64 * 64),
          W("move/gru/b_h", 3 * 64), gx, 18, r, 64, rn);
      std::memcpy(r, rn, sizeof r);
      const int pc = std::min(std::max(pos, 0), kN - 1);
      int32_t jp[4];
      float edge[48];
      if (known) D(pc, edge);
      for (int d = 0; d < 4; ++d) {
        jp[d] = known ? J[pc][d] : -1;
        if (!known) {
          float* e = edge + 12 * d;
          for (int k = 0; k < 12; ++k) e[k] = 0;
          e[d] = 1;
          e[4] = 1;
        }
      }
      float logits[5];
      for (int d = 0; d < 4; ++d) {
        const float* vfeat = jp[d] >= 0 ? z.data() + static_cast<size_t>(std::min(jp[d], kN - 1)) * kWidth : unknown;
        float hid[64];
        for (int o = 0; o < 64; ++o) hid[o] = dbase[o] + dw1[(310 + d) * 64 + o];
        accum(dw1, r, 64, 64, hid);
        accum(dw1 + 256 * 64, vfeat, 32, 64, hid);
        accum(dw1 + 288 * 64, edge + 12 * d, 12, 64, hid);
        accum(dw1 + 300 * 64, q, 10, 64, hid);
        float s2 = db2[0];
        for (int o = 0; o < 64; ++o) s2 += dw2[o] * silu(hid[o]);
        logits[d] = s2;
      }
      {
        float hid[64];
        std::memcpy(hid, sbase, sizeof hid);
        accum(sw1, r, 64, 64, hid);
        accum(sw1 + 256 * 64, q, 10, 64, hid);
        float s2 = sb2[0];
        for (int o = 0; o < 64; ++o) s2 += sw2[o] * silu(hid[o]);
        logits[4] = s2;
      }
      const bool mask[5] = {true, true, true, true, t > 0};
      float lps[5];
      log_softmax(logits, mask, 5, lps);
      const int tok = sample(logits, mask, 5, u[1 + t]);
      logp += lps[tok];
      if (tok == kStop) {
        ++t;
        break;
      }
      dirs[n_dirs++] = tok;
      const int32_t nxt = jp[std::min(tok, 3)];
      pos = (known && nxt >= 0) ? nxt : 0;
      known = known && nxt >= 0;
      heading = std::min(tok, 3);
      prev_tok = tok;
    }
    phys_draws = t;                                  // one uniform per sampled token
    out.nsteps = n_dirs;
  }

  // ---- sonar, conditioned on the physical command ----
  {
    float C[36], phys[32];
    physical_vector(out.kind, dirs.data(), out.split, Lf, A, C);
    linear(W("physical/wT", 32 * 36), W("physical/b", 32), C, 36, 32, phys);
    for (float& v : phys) v = silu(v);
    const float* w1 = W("sonar/l1/wT", 64 * 228);
    const float* b1 = W("sonar/l1/b", 64);
    const float* w2 = W("sonar/l2/wT", kSonarSymbols * 64);
    const float* b2 = W("sonar/l2/b", kSonarSymbols);
    float base[64];
    linear(w1, b1, h, 192, 64, base);
    accum(w1 + 192 * 64, phys, 32, 64, base);
    static const std::array<bool, kSonarSymbols> all = [] {
      std::array<bool, kSonarSymbols> a{};
      a.fill(true);
      return a;
    }();
    for (int d = 0; d < 4; ++d) {
      float hid[64], logits[kSonarSymbols], lp[kSonarSymbols];
      for (int o = 0; o < 64; ++o) hid[o] = silu(base[o] + w1[(224 + d) * 64 + o]);
      linear(w2, b2, hid, 64, kSonarSymbols, logits);
      log_softmax(logits, all.data(), kSonarSymbols, lp);
      out.sonar[d] = sample(logits, all.data(), kSonarSymbols, u[1 + phys_draws + d]);
      logp += lp[out.sonar[d]];
    }
  }
  for (int t = 0; t < kMaxTok; ++t) out.dirs[t] = std::max(dirs[t], 0);
  out.logp = logp;
  out.draws = 1 + phys_draws + 4;
  return out;
}

}  // namespace rl
