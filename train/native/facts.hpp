// One dragon lifetime's private facts (spec section 3) and their exact integer
// encoding for the network (the "actor codec"; features.py turns it into the
// float tensors of section 4).
//
// A Lifetime only ever reads its own observations and its own commands.
// Nothing here knows the rules beyond what a bot is told, and nothing here
// computes routes, distances or reachability: J is one step along an edge
// whose far side is known.
#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "observation.hpp"

namespace rl {

inline constexpr int kMaxMoveTokens = 8;       // policy cap on a MOVE (spec section 1)
inline constexpr int kUnknown = -1;

enum EdgeType : uint8_t { kEdgeUnknown = 0, kEdgeOpen = 1, kEdgeKelp = 2, kEdgePortal = 3 };
enum Occupant : uint8_t {
  kOccUnknown = 0, kOccEmpty, kOccSelfHead, kOccSelfBody, kOccAllyHead, kOccAllyBody, kOccEnemyHead, kOccEnemyBody
};

// A command as issued (and as the policy's action codec stores it).
struct Command {
  uint8_t kind = 0;                        // 0 MOVE, 1 SPLIT
  uint8_t nsteps = 0;                      // MOVE: 1..8 directions
  std::array<uint8_t, kMaxMoveTokens> dirs{};
  int32_t split = 0;                       // SPLIT: child segments
  uint8_t sonar_mask = 0;                  // bit d: a word sent in direction d
  std::array<uint64_t, 4> words{};
};

// Compact (int16: rounds, counts, timers and ids are all small): a lifetime holds one per board tile.
struct TileFact {
  int16_t seen_round = kUnknown;           // last round the tile was in view
  int16_t pearl = 0, countdown = -1;       // as last seen
  int16_t prev_round = kUnknown;           // the record before that one
  int16_t prev_pearl = 0, prev_countdown = -1;
  uint8_t occ = kOccUnknown;
  int8_t occ_facing = -1;
  int16_t occ_id = -1;
  int16_t visits = 0, visit_round = kUnknown;
};

struct EdgeFact {
  uint8_t type = kEdgeUnknown;
  int16_t portal = -1;
};

// Field order of one tile's integer record (TILE_FIELDS columns).
enum TileField : int {
  kTSeenRound = 0, kTPearl, kTCountdown, kTVisible, kTVisits, kTVisitRound, kTBodyIndex,
  kTPrevRound, kTPrevPearl, kTPrevCountdown, kTOcc, kTOccOrder, kTOccFacing,
  kTEdges,      // 2 bits per direction: EdgeType
  kTFlags,      // bit d: ordinary wrapped seam crossing; bit 4+d: confirmed portal pair
  kTValid,      // 1 on a board tile, 0 on batch padding
  kTileFields
};
// Visible entities: one row per dragon id in view, sorted by id.
enum EntityField : int {
  kEValid = 0, kERelation /*0 self 1 ally 2 enemy*/, kEOrder /*0 earlier 1 same 2 later*/, kECount,
  kEHeadVisible, kEHeadX, kEHeadY, kEHeadFacing, kEntityFields
};
inline constexpr int kMaxEntities = kWindowTiles;
// Visible segments: entity row, x, y, valid.
inline constexpr int kSegmentFields = 4;
enum StatusField : int {
  kSLength = 0, kSRound, kSWidth, kSHeight, kSUnitCount, kSUnitLimit, kSKnownBody, kSTailKnown, kSTurnCount,
  kSFacing, kSHeadX, kSHeadY, kSPrevValid, kSPrevKind, kSPrevSteps, kSPrevSplit, kSPrevLength,
  kSSeenTiles, kSKnownEdges, kSInboxCount, kSEchoPresent, kSHeadBed, kSHeadPearl, kSHeadTile,
  kSPrevSonarMask, kStatusFields
};
inline constexpr int kPrevDirsFields = kMaxMoveTokens;   // previous MOVE directions, -1 unused

// One decision's integer inputs, written into caller-owned, zeroed rows.
struct DecisionRows {
  int32_t* tile;      // [n_pad, kTileFields]
  int32_t* dest;      // [n_pad, 4]
  int32_t* entity;    // [kMaxEntities, kEntityFields]
  int32_t* segment;   // [kWindowTiles, kSegmentFields]
  int32_t* status;    // [kStatusFields]
  int32_t* prev_dirs; // [kPrevDirsFields]
  uint8_t* prev_words;// [4, 8] little-endian bytes of the previous words
  int32_t n_pad;
};

// simple1: what one decision sees of its remembered map, fixed-size whatever the board.
inline constexpr int kWinRadius = 8;                          // the move decoder walks at most 8 steps
inline constexpr int kWin = 2 * kWinRadius + 1;               // 17
inline constexpr int kWinTiles = kWin * kWin;                 // 289, head at the centre
inline constexpr int kWinOutside = -2;                        // a known destination outside the window
inline constexpr int kGrid = 8;                               // the whole map, head-relative, 8 x 8 cells
inline constexpr int kGridCells = kGrid * kGrid;
enum GridField : int {
  kGTiles = 0, kGSeen, kGVisible, kGPearl, kGBed, kGVisited, kGSelfBody, kGAlly, kGEnemy, kGEmpty,
  kGKelp, kGPortal, kGSeenRoundSum, kGVisitRoundSum, kGridFields
};

struct WindowRows {
  int32_t* win;       // [kWinTiles, kTileFields]: the records, row-major from (-8, -8) to (+8, +8)
  int32_t* wdest;     // [kWinTiles, 4]: J as window cells, kUnknown, or kWinOutside
  int32_t* grid;      // [kGridCells, kGridFields]
};

class Lifetime {
 public:
  explicit Lifetime(const Spawn& spawn);

  // Section 3 steps 1-3 for one own observation. Returns false, changing
  // nothing, when this observation was already processed (same key).
  bool Observe(const Observation& o);
  // Step 4's commit: the command issued for the last observation.
  void Commit(const Command& c);

  // The encoding of the facts as of the last observation `o` (the whole board;
  // tests and tools: simple1 encodes EncodeWindow).
  void Encode(const Observation& o, DecisionRows& rows) const;
  // A copy whose tile records are computed from scratch (the reference for the incremental ones).
  Lifetime Rebuilt(const Observation& o) const;
  // The same, computed from scratch: the reference the incremental tables are tested against.
  void EncodeReference(const Observation& o, DecisionRows& rows) const;

  // simple1: the window around the head and the summary grid of the whole map.
  void EncodeWindow(WindowRows& rows) const;
  // Only what Encode writes besides the tile records.
  void EncodeSmall(const Observation& o, DecisionRows& rows) const { EncodeRest(o, rows); }
  int32_t size() const { return n_; }

  // J: the tile one step `dir` from `tile` leads to, if privately known.
  int32_t Dest(int32_t tile, int dir) const;

  const Spawn& spawn() const { return spawn_; }
  int32_t turns() const { return turns_; }
  int32_t seen_tiles() const { return seen_tiles_; }
  double seen_fraction() const { return static_cast<double>(seen_tiles_) / (spawn_.width * spawn_.height); }
  const std::vector<int32_t>& body() const { return body_; }
  const TileFact& tile(int32_t t) const { return tiles_[t]; }
  const Command& previous() const { return prev_; }
  bool previous_valid() const { return prev_valid_; }
  // canonical edges: 2t is the north side of tile t, 2t + 1 its west side
  const EdgeFact& edge(int32_t e) const { return edges_[e]; }
  int32_t SideEdge(int32_t tile, int dir) const;

 private:
  int32_t Wrap(int x, int y) const;
  int32_t Neighbour(int32_t tile, int dir) const;
  void Reconcile(const Observation& o);
  void Record(const Observation& o);
  void LinkBody(const Observation& o);
  void Visit(int32_t tile, int32_t round);
  void EncodeTile(int32_t t, int32_t* row, int32_t* dest) const;   // one tile's record, from the facts
  void EncodeRest(const Observation& o, DecisionRows& rows) const;  // everything but the tile records

  Spawn spawn_;
  int32_t n_;
  std::vector<TileFact> tiles_;
  std::vector<EdgeFact> edges_;                                  // [2n]
  std::unordered_map<int32_t, std::vector<int32_t>> portals_;    // id -> canonical edges seen
  std::vector<int32_t> body_;                                    // head first, kUnknown
  Command prev_;
  bool prev_valid_ = false;
  // the last observation's head, length and round, and their values when the
  // previous command was issued (spec: head_before, length_before)
  int32_t last_head_ = kUnknown, last_length_ = 0, last_round_ = kUnknown;
  int32_t cmd_head_ = kUnknown, cmd_length_ = 0, cmd_round_ = kUnknown;
  int32_t turns_ = 0;
  int32_t last_key_ = kUnknown;                                  // round of the last observation
  int32_t seen_tiles_ = 0, known_edges_ = 0;
  std::vector<int32_t> body_index_;              // [n]: index in body_, -1
  std::vector<uint8_t> in_view_;                 // [n]
  std::vector<int32_t> window_;
};

}  // namespace rl
