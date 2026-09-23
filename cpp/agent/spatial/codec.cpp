#include "codec.hpp"

#include <algorithm>
#include <cmath>

namespace dragons::spatial {

PrefixSimulator::PrefixSimulator(const LifetimeState& state)
    : state_(state), facts_(state.facts), K_(state.max_steps),
      area_(static_cast<double>(state.width()) * state.height()),
      pos_(state.head), old_length_(state.length), length_(state.length) {
  if (state.body) body_index_ = state.body->IndexOf();
}

PrefixSimulator::Occ PrefixSimulator::Occupancy(Cell c) const {
  const size_t k = facts_.Index(c.x, c.y);
  for (const Cell& t : trail_) {
    if (t == c) return Occ::Self;
  }
  const bool visible = facts_.visible_round == state_.round && facts_.visible[k];
  const uint8_t kind = facts_.occupant_kind[k];
  if (visible && kind >= OCC_ALLY_HEAD) {
    return (kind == OCC_ALLY_HEAD || kind == OCC_ENEMY_HEAD) ? Occ::Head : Occ::Other;
  }
  if (visible && (kind == OCC_SELF_HEAD || kind == OCC_SELF_BODY)) {
    const auto it = body_index_.find(c);
    if (it != body_index_.end()) return OldPresent(it->second) ? Occ::Self : Occ::Empty;
    // the index is not known: at the first step every old cell is still
    // there, later on it may have been the tail that moved
    return steps_ == 0 ? Occ::Self : Occ::Unknown;
  }
  if (!visible) return Occ::Unknown;
  return Occ::Empty;
}

void PrefixSimulator::Look(int d, FactualMemory::StepKind& kind, bool& has_landing, Cell& landing,
                           Occ& occ, bool& pearl) const {
  has_landing = false;
  occ = Occ::Unknown;
  pearl = false;
  if (!pos_known_) {
    kind = FactualMemory::StepKind::Unknown;
    return;
  }
  kind = facts_.Step(pos_.x, pos_.y, d, landing);
  if (kind != FactualMemory::StepKind::Empty && kind != FactualMemory::StepKind::Portal) return;
  has_landing = true;
  occ = Occupancy(landing);
  const size_t k = facts_.Index(landing.x, landing.y);
  pearl = facts_.visible_round == state_.round && facts_.visible[k] && facts_.pearl[k];
}

void PrefixSimulator::Features(float* out) const {
  std::fill(out, out + kPrefixFeatures, 0.0f);
  out[P_STEP] = static_cast<float>(static_cast<double>(steps_) / K_);
  out[P_POSITION_KNOWN] = pos_known_;
  out[P_ALIVE_KNOWN] = !fatal_known_ && pos_known_;
  out[P_FATAL_KNOWN] = fatal_known_;
  out[P_LENGTH_KNOWN] = length_known_;
  out[P_LENGTH] = length_known_ ? static_cast<float>(length_ / area_) : 0.0f;
  out[P_CAN_PAY] = length_known_ && length_ > kMinSize;
  out[P_RESOURCE_UNCERTAIN] = !length_known_;
  if (last_ >= 0) out[P_PREV + last_] = 1.0f;
  for (int d = 0; d < 4; ++d) {
    FactualMemory::StepKind kind;
    bool has_landing, pearl;
    Cell landing;
    Occ occ;
    Look(d, kind, has_landing, landing, occ, pearl);
    if (kind == FactualMemory::StepKind::Empty) out[P_AHEAD_EMPTY + d] = 1.0f;
    else if (kind == FactualMemory::StepKind::Kelp) out[P_AHEAD_KELP + d] = 1.0f;
    else if (kind == FactualMemory::StepKind::Portal || kind == FactualMemory::StepKind::PortalUnknown)
      out[P_AHEAD_PORTAL + d] = 1.0f;
    else out[P_AHEAD_UNKNOWN + d] = 1.0f;
    switch (occ) {
      case Occ::Self: out[P_OCC_SELF + d] = 1.0f; break;
      case Occ::Other: out[P_OCC_OTHER + d] = 1.0f; break;
      case Occ::Head: out[P_OCC_HEAD + d] = 1.0f; break;
      case Occ::Unknown: out[P_OCC_UNKNOWN + d] = 1.0f; break;
      case Occ::Empty: break;
    }
    out[P_PEARL + d] = pearl;
  }
  out[P_PEARLS_EATEN] = static_cast<float>(static_cast<double>(eaten_) / K_);
}

void PrefixSimulator::Mask(std::array<bool, kTokens>& out) const {
  out.fill(true);
  if (steps_ == 0) {
    out[kStop] = false;
    return;
  }
  if (fatal_known_ || (length_known_ && length_ <= kMinSize)) {
    for (int t = 0; t < kStop; ++t) out[t] = false;
  }
}

void PrefixSimulator::Advance(int token) {
  if (token == kStop) return;
  const bool pay = steps_ >= 1;
  FactualMemory::StepKind kind;
  bool has_landing, pearl;
  Cell landing;
  Occ occ;
  Look(token, kind, has_landing, landing, occ, pearl);
  if (kind == FactualMemory::StepKind::Kelp || occ == Occ::Self || occ == Occ::Other || occ == Occ::Head) {
    fatal_known_ = true;
  }
  if (!has_landing) {
    pos_known_ = false;
    if (pay) --length_;
    length_known_ = false;
  } else {
    pos_ = landing;
    trail_.push_back(landing);
    const size_t k = facts_.Index(landing.x, landing.y);
    const bool visible = facts_.visible_round == state_.round && facts_.visible[k];
    if (visible) {
      if (pearl) {
        ++eaten_;
        ++length_;
      }
    } else {
      length_known_ = false;
    }
    if (pay) --length_;
  }
  ++steps_;
  last_ = token;
}

SplitTable MakeSplitTable(const LifetimeState& state) {
  SplitTable out;
  if (!state.split_legal) return out;
  const int L = state.length;
  const double area = static_cast<double>(state.width()) * state.height();
  for (int n = kMinSize; n <= L - kMinSize; ++n) {
    out.sizes.push_back(n);
    out.features.push_back({static_cast<float>(static_cast<double>(n) / L),
                            static_cast<float>(static_cast<double>(L - n) / L),
                            static_cast<float>(std::log1p(static_cast<double>(n)) / std::log1p(area)),
                            static_cast<float>(std::log1p(static_cast<double>(L - n)) / std::log1p(area))});
  }
  return out;
}

}  // namespace dragons::spatial
