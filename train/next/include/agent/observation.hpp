// A dragon's view of one turn: what the judge sends, and nothing else.
//
// Built either from a dragons-env record (training) or from the judge's text
// (the bot); both must produce equal Observations for the same turn, which a
// test checks on recorded games. Plain data: no engine types, no allocation
// beyond the inbox.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace agent {

inline constexpr int kVision = 7;
inline constexpr int kVisionTiles = kVision * kVision;

enum class Dir : uint8_t { North, East, South, West };
enum class Team : uint8_t { A, B };

struct Point {
  int x, y;
  friend bool operator==(Point, Point) = default;
};

// What an edge of the window shows.
struct EdgeView {
  enum class Kind : uint8_t { Empty, Kelp, Portal } kind = Kind::Empty;
  int32_t portal_id = -1;   // meaningful only for Kind::Portal
};

// A living dragon's segment on a window tile.
struct SegmentView {
  int32_t dragon_id;
  Team team;
  Dir facing;               // towards the head; a head's facing is its heading
  bool is_head;
  int32_t rank;             // its line in the judge's DRAGON_BODIES list
};

struct TileView {
  Point at;                 // absolute board coordinates
  bool pearl;
  std::optional<int32_t> countdown;   // present only on pearl beds
  std::optional<SegmentView> segment;
};

// Sent once, when the dragon spawns.
struct SpawnInfo {
  int32_t id;
  Team team;
  int32_t width, height, unit_limit;
};

struct Observation {
  SpawnInfo self;
  int32_t round;
  Dir facing;
  int32_t length;
  int32_t unit_count;
  int32_t turn;             // turns this dragon has taken before this one (0 = just spawned)
  std::vector<uint32_t> inbox;
  std::array<TileView, kVisionTiles> tiles;              // row-major from the window's top left
  std::array<std::array<EdgeView, kVision>, kVision + 1> north_sides;   // row r's north sides; row 7 = south of row 6
  std::array<std::array<EdgeView, kVision + 1>, kVision> west_sides;    // column c's west sides; column 7 = east of column 6

  Point head() const { return tiles[kVisionTiles / 2].at; }
  // the side `d` of window tile (row, col)
  EdgeView Side(int row, int col, Dir d) const;
};

// From a dragons-env observation record (the layout comes from dragons::Layout).
struct RecordLayout;   // offsets into a record; mirrors dragons::Layout without depending on it
Observation FromRecord(std::span<const uint8_t> record, const RecordLayout& layout, const SpawnInfo& self);

// From the judge's text; `turn` is counted by the caller.
Observation FromRoundBlock(std::string_view round_block, const SpawnInfo& self, int32_t turn);
SpawnInfo FromInitBlock(std::string_view init_block);

}  // namespace agent
