// Private factual memory for one dragon lifetime, in C++ (spec 3, 4.2-4.5).
//
// This mirrors snake_rl/env/memory.py and observation.py exactly: the same
// per-tile records, the same canonical north/west edges, the same own-body
// deque rebuilt from the dragon's own moves, the same 42-channel
// materialisation. tests/test_spatial_native.py compares every tensor against
// the Python reference on real matches.
//
// The difference is where the observation comes from. Python parses the turn
// block text; this reads the same 7x7 window straight out of the native sim,
// through the same rules cpp/agent/encoder.cpp uses to print that text. Nothing
// outside the window is read: the sim is the source of the observation, not of
// the memory.
#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "dragons/game.hpp"

namespace dragons::spatial {

inline constexpr int kVision = 7;
inline constexpr int kRadius = 3;
inline constexpr int kTiles = kVision * kVision;
inline constexpr int kMaxRounds = 500;
inline constexpr int kLastRound = 499;
inline constexpr int kMaxBoard = 64;
inline constexpr int kHistoryTurns = 32;
inline constexpr int kMinSize = 2;

// schema.py, kept in step by tests/test_spatial_native.py
inline constexpr int kFactChannels = 42;
inline constexpr int kStatus = 36;
inline constexpr int kHistoryFeatures = 30;
inline constexpr int kEntityFeatures = 16;
inline constexpr int kMaxEntities = kTiles;

enum Fact : int {
  F_SEEN = 0, F_VISIBLE = 1, F_AGE_SEEN = 2, F_PEARL = 3, F_COUNTDOWN = 4, F_NEVER = 5,
  F_OCC = 6, F_FACING = 13, F_EDGE_KNOWN = 17, F_KELP = 21, F_PORTAL = 25,
  F_VISIT_LOG = 29, F_AGE_VISIT = 30, F_VISITED = 31, F_PREV_EXISTS = 32, F_PREV_PEARL = 33,
  F_PREV_COUNTDOWN = 34, F_INTERVAL = 35, F_ID_EARLIER = 36, F_ID_LATER = 37,
  F_BODY_KNOWN = 38, F_BODY_INDEX = 39, F_TAIL_KNOWN = 40, F_HEAD = 41,
};

enum Occupant : uint8_t {
  OCC_EMPTY = 0, OCC_SELF_HEAD, OCC_SELF_BODY, OCC_ALLY_HEAD, OCC_ALLY_BODY,
  OCC_ENEMY_HEAD, OCC_ENEMY_BODY,
};

enum EdgeType : uint8_t { EDGE_EMPTY = 0, EDGE_KELP = 1, EDGE_PORTAL = 2 };

struct Cell {
  int16_t x, y;
  bool operator==(const Cell& o) const { return x == o.x && y == o.y; }
  bool operator<(const Cell& o) const { return y != o.y ? y < o.y : x < o.x; }
};

// A canonical edge: axis 0 is the north edge of (x, y), axis 1 its west edge.
struct CanonicalEdge {
  int axis, x, y;
  bool operator==(const CanonicalEdge& o) const { return axis == o.axis && x == o.x && y == o.y; }
  bool operator<(const CanonicalEdge& o) const {
    if (axis != o.axis) return axis < o.axis;
    if (x != o.x) return x < o.x;
    return y < o.y;
  }
};

// One visible segment, as the turn block prints it.
struct Part {
  int team;          // 0 = A, 1 = B
  int dragon_id;
  int x, y;
  int facing;        // 0..3 N E S W
  bool is_head;
};

// One dragon's view of one turn: everything the wire block carries.
struct Window {
  int round = 0;
  int facing = 0;
  int length = 0;
  int unit_count = 0;
  int head_x = 0, head_y = 0;
  std::array<int16_t, kTiles> xs{}, ys{};
  std::array<uint8_t, kTiles> pearl{};
  std::array<int32_t, kTiles> countdown{};
  // edge per tile per direction: kind and portal id (-1)
  std::array<std::array<uint8_t, 4>, kTiles> edge_kind{};
  std::array<std::array<int32_t, 4>, kTiles> edge_portal{};
  std::vector<Part> parts;
  std::vector<uint32_t> inbox;
};

// Reads dragon `index`'s window out of the sim, exactly as TurnBlock prints it.
Window ReadWindow(const Game& game, int index, const std::vector<int>& portal_ids);

class FactualMemory {
 public:
  FactualMemory(int width, int height);

  int width() const { return width_; }
  int height() const { return height_; }

  CanonicalEdge Canonical(int x, int y, int d) const;
  // (known, kind, portal id) of the edge crossed stepping d out of (x, y)
  void Edge(int x, int y, int d, bool& known, uint8_t& kind, int32_t& portal) const;
  std::optional<CanonicalEdge> PortalPartner(int x, int y, int d) const;
  // "empty"/"portal" with a landing, "portal" without one, "kelp", "unknown"
  enum class StepKind { Empty, Portal, PortalUnknown, Kelp, Unknown };
  StepKind Step(int x, int y, int d, Cell& landing) const;
  std::vector<std::pair<CanonicalEdge, CanonicalEdge>> ConfirmedPortalPairs() const;
  std::vector<CanonicalEdge> UnpairedPortals() const;

  void Observe(const Window& w, int dragon_id, int team);
  void MarkVisited(Cell c, int round);

  // per-tile records (row-major [H][W])
  std::vector<uint8_t> seen;
  std::vector<int16_t> last_seen_round;
  std::vector<uint8_t> pearl;
  std::vector<int32_t> countdown;
  std::vector<uint8_t> prev_pearl;
  std::vector<int32_t> prev_countdown;
  std::vector<int16_t> prev_seen_round;
  std::vector<uint8_t> occupant_kind;
  std::vector<int32_t> occupant_id;
  std::vector<uint8_t> occupant_facing;
  std::vector<uint16_t> visit_count;
  std::vector<int16_t> last_visit_round;
  // canonical edges [2][H][W]
  std::vector<uint8_t> edge_known;
  std::vector<uint8_t> edge_type;
  std::vector<int32_t> edge_portal;
  std::map<int, std::set<CanonicalEdge>> portals;
  int visible_round = -1;
  std::vector<uint8_t> visible;

  size_t Index(int x, int y) const { return static_cast<size_t>(y) * width_ + x; }
  size_t EdgeIndex(int axis, int x, int y) const { return static_cast<size_t>(axis) * width_ * height_ + Index(x, y); }

 private:
  void SetEdge(int x, int y, int d, uint8_t kind, int32_t portal);
  int width_, height_;
};

// Head-first order of the dragon's own segments; unknown cells are nullopt.
class BodyHistory {
 public:
  BodyHistory() = default;
  BodyHistory(int length, Cell head);
  std::vector<std::optional<Cell>> cells;
  int size() const { return static_cast<int>(cells.size()); }
  bool TailKnown() const { return !cells.empty() && cells.back().has_value(); }
  float KnownFraction() const;
  std::map<Cell, int> IndexOf() const;
  void ApplyMove(const std::vector<std::optional<Cell>>& path, int new_length);
  void ApplySplit(int detached);
  void Reconcile(const Window& w, int dragon_id);
};

struct CommandRecord {
  bool is_move = true;
  std::string dirs;
  int split_n = 0;
  int round = 0;
  Cell old_head{0, 0}, new_head{0, 0};
  int old_heading = 0, new_heading = 0;
  int old_length = 0, new_length = 0;
  bool intermediates_known = true;
};

struct PendingCommand {
  bool is_move = true;
  std::string dirs;
  int split_n = 0;
  int round = 0;
  Cell head{0, 0};
  int heading = 0;
  int length = 0;
};

// Everything private to one lifetime, plus the tensors prepare() derives.
class LifetimeState {
 public:
  LifetimeState(int dragon_id, int team, int width, int height, int unit_limit, int max_steps = 8);

  int dragon_id() const { return dragon_id_; }
  int team() const { return team_; }
  int width() const { return facts.width(); }
  int height() const { return facts.height(); }

  // prepare(): advance the state by this window and fill the tensors. Idempotent
  // for a repeated round. Buffers must be sized as documented in Sizes().
  struct Buffers {
    float* local;              // [42, 7, 7]
    float* facts;              // [42, H, W]
    int64_t* visible_xy;       // [49, 2]
    float* visible_prev;       // [49, 42]
    float* visible_age;        // [49]
    float* status;             // [kStatus]
    float* history;            // [32, kHistoryFeatures]
    uint8_t* history_mask;     // [32]
    float* entities;           // [49, kEntityFeatures]
    uint8_t* entity_mask;      // [49]
  };
  void Prepare(const Window& w, const Buffers& out);
  // side tables, sized by the caller after the call
  std::vector<std::array<int64_t, 4>> portal_pairs;   // (y1, x1, y2, x2)
  std::vector<int64_t> portal_axis;
  std::vector<std::array<int64_t, 2>> portal_unpaired; // (y, x)
  std::vector<uint32_t> inbox;
  int length = 0, heading = 0, unit_count = 0, round = 0;
  bool split_legal = false;
  Cell head{0, 0};

  void CommitMove(const std::string& dirs);
  void CommitSplit(int n);
  void Materialize(int round_num, float* out) const;   // [42, H, W]
  std::vector<std::optional<Cell>> TracePath(Cell start, const std::string& dirs, bool& known) const;

  FactualMemory facts;
  std::optional<BodyHistory> body;
  std::deque<CommandRecord> history;
  std::optional<PendingCommand> pending;
  int own_turns = 0;
  int last_processed_round = -1;
  int unit_limit;
  int max_steps;

 private:
  void CompletePending(const Window& w);
  void Status(const Window& w, float* out) const;
  void HistoryFeatures(int round_num, float* out, uint8_t* mask) const;
  void Entities(const Window& w, float* out, uint8_t* mask) const;
  int dragon_id_, team_;
  // the pre-write features of the last processed round, for idempotent redelivery
  std::vector<float> prewrite_prev_;
  std::vector<float> prewrite_age_;
  int prewrite_round_ = -1;
};

}  // namespace dragons::spatial
