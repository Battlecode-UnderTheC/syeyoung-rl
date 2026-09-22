// The factual map on the GPU (spec 4.2/4.3, 12.3).
//
// Each lifetime slot holds a compact *raw* store of 20 channels -- the
// structured per-tile records, not the 42 derived features. The CPU side
// (agent/spatial) owns the truth and, after each turn, emits only the cells
// that turn changed; the GPU applies them with one scatter. The 42-channel
// map the actor consumes is derived on the GPU by elementwise ops at the
// current round, so ages are computed on read exactly as the CPU does it, and
// nothing dense ever crosses PCIe per step.
//
// Materialize() here and LifetimeState::Materialize() on the CPU must agree
// to float precision; selftest.cpp checks that on random play.
#pragma once

#include <cstdint>
#include <vector>

#include <torch/torch.h>

#include "agent/spatial/memory.hpp"

namespace dragons::train {

inline constexpr int kRaw = 20;
enum Raw : int {
  R_SEEN = 0, R_LAST_SEEN = 1, R_PEARL = 2, R_COUNTDOWN = 3, R_PREV_PEARL = 4,
  R_PREV_COUNTDOWN = 5, R_PREV_SEEN = 6, R_OCC_KIND = 7, R_OCC_REL = 8, R_OCC_FACING = 9,
  R_VISITS = 10, R_LAST_VISIT = 11, R_EDGE_KNOWN_N = 12, R_EDGE_TYPE_N = 13,
  R_EDGE_KNOWN_W = 14, R_EDGE_TYPE_W = 15, R_BODY_KNOWN = 16, R_BODY_INDEX = 17,
  R_TAIL_KNOWN = 18, R_HEAD = 19,
};
// R_OCC_REL: -1 none, 0 self, 1 an id earlier than self, 2 an id later than self

// One changed cell of one slot: its 20 raw values.
struct RawUpdate {
  int32_t slot;
  int32_t y, x;
  float values[kRaw];
};

// Fill `values` for cell (x, y) of a lifetime from its CPU state.
void RawCell(const spatial::LifetimeState& state, const std::map<spatial::Cell, int>& body_index,
             int x, int y, float* values);

// The cells a turn may have changed: the 9x9 block around the head (the
// window plus the ring whose south/east edges the window revealed), the
// previous window's block, and the old and new body cells.
void ChangedCells(const spatial::LifetimeState& state, spatial::Cell old_head, bool had_old_head,
                  const std::vector<spatial::Cell>& old_body, std::vector<spatial::Cell>& out);

// Reset slots to the empty-lifetime sentinels (-1 rounds, no occupant).
void ResetSlots(torch::Tensor raw, const torch::Tensor& slots);

// Apply a batch of updates to raw [slots, 20, H, W] (any device) in place.
void ApplyUpdates(torch::Tensor raw, const std::vector<RawUpdate>& updates);

// raw [B, 20, H, W] float, rounds [B] long -> facts [B, 42, H, W] float.
torch::Tensor Materialize(const torch::Tensor& raw, const torch::Tensor& rounds);

// Gather [B, C, 7, 7] windows at (x, y) heads from a [B, C, H, W] map, wrapped.
torch::Tensor CropWindows(const torch::Tensor& map, const torch::Tensor& head_xy);
// Gather [B, 49, C] values at the 49 visible coordinates [B, 49, 2] (x, y).
torch::Tensor GatherCells(const torch::Tensor& map, const torch::Tensor& xy);

}  // namespace dragons::train
