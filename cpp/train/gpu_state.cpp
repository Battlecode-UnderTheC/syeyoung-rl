#include "gpu_state.hpp"

#include <algorithm>
#include <cmath>
#include <set>

namespace dragons::train {

using spatial::Cell;
using spatial::LifetimeState;

void RawCell(const LifetimeState& state, const std::map<Cell, int>& body_index, int x, int y,
             float* values) {
  const spatial::FactualMemory& f = state.facts;
  const size_t k = f.Index(x, y);
  std::fill(values, values + kRaw, 0.0f);
  values[R_SEEN] = f.seen[k];
  values[R_LAST_SEEN] = f.last_seen_round[k];
  values[R_PEARL] = f.pearl[k];
  values[R_COUNTDOWN] = static_cast<float>(f.countdown[k]);
  values[R_PREV_PEARL] = f.prev_pearl[k];
  values[R_PREV_COUNTDOWN] = static_cast<float>(f.prev_countdown[k]);
  values[R_PREV_SEEN] = f.prev_seen_round[k];
  values[R_OCC_KIND] = f.occupant_kind[k];
  const int id = f.occupant_id[k];
  values[R_OCC_REL] = id == -1 ? -1.0f : (id == state.dragon_id() ? 0.0f : (id < state.dragon_id() ? 1.0f : 2.0f));
  values[R_OCC_FACING] = f.occupant_facing[k];
  values[R_VISITS] = f.visit_count[k];
  values[R_LAST_VISIT] = f.last_visit_round[k];
  values[R_EDGE_KNOWN_N] = f.edge_known[f.EdgeIndex(0, x, y)];
  values[R_EDGE_TYPE_N] = f.edge_type[f.EdgeIndex(0, x, y)];
  values[R_EDGE_KNOWN_W] = f.edge_known[f.EdgeIndex(1, x, y)];
  values[R_EDGE_TYPE_W] = f.edge_type[f.EdgeIndex(1, x, y)];
  if (state.body) {
    const auto it = body_index.find(Cell{static_cast<int16_t>(x), static_cast<int16_t>(y)});
    if (it != body_index.end()) {
      const int L = state.body->size();
      values[R_BODY_KNOWN] = 1.0f;
      values[R_BODY_INDEX] = static_cast<float>(it->second) / static_cast<float>(std::max(L - 1, 1));
      if (it->second == L - 1 && L > 1) values[R_TAIL_KNOWN] = 1.0f;
      if (it->second == 0) values[R_HEAD] = 1.0f;
    }
  }
}

void ChangedCells(const LifetimeState& state, Cell old_head, bool had_old_head,
                  const std::vector<Cell>& old_body, std::vector<Cell>& out) {
  const int W = state.width(), H = state.height();
  std::set<Cell> cells;
  auto block = [&](Cell c) {
    for (int dy = -4; dy <= 4; ++dy) {
      for (int dx = -4; dx <= 4; ++dx) {
        cells.insert(Cell{static_cast<int16_t>(((c.x + dx) % W + W) % W),
                          static_cast<int16_t>(((c.y + dy) % H + H) % H)});
      }
    }
  };
  block(state.head);
  if (had_old_head) block(old_head);
  for (const Cell& c : old_body) cells.insert(c);
  if (state.body) {
    for (const auto& c : state.body->cells) {
      if (c) cells.insert(*c);
    }
  }
  out.assign(cells.begin(), cells.end());
}

void ResetSlots(torch::Tensor raw, const torch::Tensor& slots) {
  if (slots.numel() == 0) return;
  torch::Tensor rows = raw.index_select(0, slots.to(raw.device()));
  rows.zero_();
  for (int c : {R_LAST_SEEN, R_PREV_SEEN, R_OCC_REL, R_LAST_VISIT}) rows.select(1, c).fill_(-1.0f);
  raw.index_copy_(0, slots.to(raw.device()), rows);
}

void ApplyUpdates(torch::Tensor raw, const std::vector<RawUpdate>& updates) {
  if (updates.empty()) return;
  const int64_t n = static_cast<int64_t>(updates.size());
  torch::Tensor slots = torch::empty({n}, torch::kLong);
  torch::Tensor ys = torch::empty({n}, torch::kLong);
  torch::Tensor xs = torch::empty({n}, torch::kLong);
  torch::Tensor values = torch::empty({n, kRaw}, torch::kFloat);
  auto s = slots.accessor<int64_t, 1>();
  auto y = ys.accessor<int64_t, 1>();
  auto x = xs.accessor<int64_t, 1>();
  auto v = values.accessor<float, 2>();
  for (int64_t i = 0; i < n; ++i) {
    s[i] = updates[i].slot;
    y[i] = updates[i].y;
    x[i] = updates[i].x;
    for (int c = 0; c < kRaw; ++c) v[i][c] = updates[i].values[c];
  }
  const torch::Device dev = raw.device();
  // raw[slot, :, y, x] = values  ->  index_put on a [slots, H, W, 20] view
  torch::Tensor view = raw.permute({0, 2, 3, 1});
  view.index_put_({slots.to(dev), ys.to(dev), xs.to(dev)}, values.to(dev));
}

torch::Tensor Materialize(const torch::Tensor& raw, const torch::Tensor& rounds) {
  namespace sp = spatial;
  const int64_t B = raw.size(0), H = raw.size(2), W = raw.size(3);
  const auto opts = raw.options();
  torch::Tensor out = torch::zeros({B, sp::kFactChannels, H, W}, opts);
  auto ch = [&](int c) { return raw.select(1, c); };
  auto set = [&](int c, const torch::Tensor& t) { out.select(1, c).copy_(t.to(opts.dtype())); };
  const torch::Tensor round = rounds.to(opts.dtype()).view({B, 1, 1});
  const torch::Tensor seen = ch(R_SEEN) > 0;
  const torch::Tensor seen_f = seen.to(opts.dtype());
  const torch::Tensor last_seen = ch(R_LAST_SEEN);
  set(sp::F_SEEN, seen_f);
  set(sp::F_VISIBLE, seen & (last_seen == round));
  set(sp::F_AGE_SEEN, torch::where(seen, torch::clamp(round - last_seen, 0, 500) / 500.0, torch::zeros_like(seen_f)));
  set(sp::F_PEARL, seen & (ch(R_PEARL) > 0));
  const torch::Tensor cd = ch(R_COUNTDOWN);
  set(sp::F_COUNTDOWN, torch::where(seen & (cd >= 0), 1.0 / (1.0 + torch::clamp_min(cd, 0)), torch::zeros_like(seen_f)));
  set(sp::F_NEVER, seen & (cd < 0));
  const torch::Tensor kind = ch(R_OCC_KIND);
  for (int k = 0; k < 7; ++k) set(sp::F_OCC + k, seen & (kind == k));
  const torch::Tensor occupied = seen & (kind != 0);
  const torch::Tensor facing = ch(R_OCC_FACING);
  for (int d = 0; d < 4; ++d) set(sp::F_FACING + d, occupied & (facing == d));
  const torch::Tensor kn_n = ch(R_EDGE_KNOWN_N), ty_n = ch(R_EDGE_TYPE_N);
  const torch::Tensor kn_w = ch(R_EDGE_KNOWN_W), ty_w = ch(R_EDGE_TYPE_W);
  const torch::Tensor kn_s = torch::roll(kn_n, -1, 1), ty_s = torch::roll(ty_n, -1, 1);
  const torch::Tensor kn_e = torch::roll(kn_w, -1, 2), ty_e = torch::roll(ty_w, -1, 2);
  const torch::Tensor kns[4] = {kn_n, kn_e, kn_s, kn_w};
  const torch::Tensor tys[4] = {ty_n, ty_e, ty_s, ty_w};
  for (int d = 0; d < 4; ++d) {
    const torch::Tensor known = kns[d] > 0;
    set(sp::F_EDGE_KNOWN + d, known);
    set(sp::F_KELP + d, known & (tys[d] == 1));
    set(sp::F_PORTAL + d, known & (tys[d] == 2));
  }
  const torch::Tensor visits = ch(R_VISITS);
  const torch::Tensor visited = visits > 0;
  set(sp::F_VISIT_LOG, torch::clamp(torch::log1p(visits) / std::log(501.0), 0, 1));
  set(sp::F_AGE_VISIT, torch::where(visited, torch::clamp(round - ch(R_LAST_VISIT), 0, 500) / 500.0, torch::zeros_like(seen_f)));
  set(sp::F_VISITED, visited);
  const torch::Tensor prev_seen = ch(R_PREV_SEEN);
  const torch::Tensor prev = seen & (prev_seen >= 0);
  set(sp::F_PREV_EXISTS, prev);
  set(sp::F_PREV_PEARL, prev & (ch(R_PREV_PEARL) > 0));
  const torch::Tensor pcd = ch(R_PREV_COUNTDOWN);
  set(sp::F_PREV_COUNTDOWN, torch::where(prev & (pcd >= 0), 1.0 / (1.0 + torch::clamp_min(pcd, 0)), torch::zeros_like(seen_f)));
  set(sp::F_INTERVAL, torch::where(prev, torch::clamp(last_seen - prev_seen, 0, 500) / 500.0, torch::zeros_like(seen_f)));
  const torch::Tensor rel = ch(R_OCC_REL);
  set(sp::F_ID_EARLIER, occupied & (rel == 1));
  set(sp::F_ID_LATER, occupied & (rel == 2));
  set(sp::F_BODY_KNOWN, ch(R_BODY_KNOWN));
  set(sp::F_BODY_INDEX, ch(R_BODY_INDEX));
  set(sp::F_TAIL_KNOWN, ch(R_TAIL_KNOWN));
  set(sp::F_HEAD, ch(R_HEAD));
  return out;
}

torch::Tensor CropWindows(const torch::Tensor& map, const torch::Tensor& head_xy) {
  const int64_t B = map.size(0), H = map.size(2), W = map.size(3);
  const torch::Tensor off = torch::arange(-spatial::kRadius, spatial::kRadius + 1, head_xy.options());
  const torch::Tensor ys = (head_xy.select(1, 1).unsqueeze(1) + off.unsqueeze(0)).remainder(H);  // [B, 7]
  const torch::Tensor xs = (head_xy.select(1, 0).unsqueeze(1) + off.unsqueeze(0)).remainder(W);
  const torch::Tensor b = torch::arange(B, head_xy.options()).view({B, 1, 1});
  // map[b, :, ys[:, :, None], xs[:, None, :]] -> [B, 7, 7, C]
  torch::Tensor rows = map.index({b, torch::indexing::Slice(), ys.unsqueeze(2), xs.unsqueeze(1)});
  return rows.permute({0, 3, 1, 2}).contiguous();
}

torch::Tensor GatherCells(const torch::Tensor& map, const torch::Tensor& xy) {
  const int64_t B = map.size(0), N = xy.size(1);
  const torch::Tensor b = torch::arange(B, xy.options()).view({B, 1}).expand({B, N});
  return map.index({b, torch::indexing::Slice(), xy.select(2, 1), xy.select(2, 0)});   // [B, N, C]
}

}  // namespace dragons::train
