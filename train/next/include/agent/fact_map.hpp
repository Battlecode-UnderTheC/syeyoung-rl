// A dragon's memory, kept by code: everything it has seen, never learned.
//
// Contract:
//   * fed only by the dragon's own Observations and the Commands it sent;
//   * deterministic: equal inputs give equal maps (so rollouts can store its
//     encoded views instead of the map);
//   * sized to the board (W x H tiles), host memory only;
//   * a split child starts from an empty FactMap, as the engine gives it nothing.
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "agent/actions.hpp"
#include "agent/observation.hpp"

namespace agent {

// What is known about one tile. Absent fields were never observed.
struct TileFacts {
  std::optional<int32_t> seen_round;              // last round in view
  std::optional<EdgeView> north, west;            // edges never change: known once seen
  std::optional<bool> is_bed;
  std::optional<int32_t> countdown_base;          // countdown when seen + that round (predicted = base - now)
  bool pearl_when_seen = false;
  enum class Occupant : uint8_t { None, AllyBody, AllyHead, EnemyBody, EnemyHead };
  Occupant occupant_when_seen = Occupant::None;
  std::optional<int32_t> own_head_round;          // last round this dragon's head was here
};

struct EnemyFacts {
  int32_t dragon_id;
  int32_t longest_seen;      // most segments seen at once: a lower bound on its length
  int32_t last_seen_round;
  Point last_head;           // valid if its head was ever seen
};

class FactMap {
 public:
  explicit FactMap(const SpawnInfo& self);

  // A new turn: everything in view is recorded, the own body advanced.
  void Observe(const Observation& obs);
  // The command this dragon sent this turn (the trail needs sprint midpoints).
  void Commit(const Command& sent);

  int32_t width() const;
  int32_t height() const;
  const TileFacts& At(Point p) const;             // wraps p onto the board
  std::span<const Point> OwnBody() const;         // head first; may stop short where unknown (see MODEL.md)
  std::span<const EnemyFacts> Enemies() const;
  // where crossing the portal edge on side `d` of tile `p` leads, once both ends were seen
  std::optional<Point> PortalExit(Point p, Dir d) const;

 private:
  // representation is the implementation's business (about 8 bytes per tile)
  struct Impl;
  std::vector<uint8_t> storage_;
};

}  // namespace agent
