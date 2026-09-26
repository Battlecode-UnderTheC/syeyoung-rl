// The training-only critic's view (spec section 8): the full state at an
// event, the boundary right before the next living dragon acts. It reads the
// game directly and is never given to an actor.
//
// Teams are absolute here (A = 0, B = 1); features.py makes them relative to
// the perspective team p, so one encoding serves both values of an event.
#pragma once

#include <cstdint>

#include "dragons/core.hpp"

namespace rl {

// simple1: the true board as an absolute 8 x 8 grid of cells, per event:
// counts per cell (teams absolute; features.py makes them relative).
inline constexpr int kCriticGrid = 8;
inline constexpr int kCriticCells = kCriticGrid * kCriticGrid;
enum CriticCellField : int {
  kCTiles = 0, kCPearls, kCBeds, kCTimerSum, kCHeadA, kCBodyA, kCHeadB, kCBodyB, kCKelp, kCPortal, kCriticCellFields
};
void EncodeCriticGrid(const dragons::Game& game, int32_t* cells);

// One row per living dragon, by id.
enum CriticDragonField : int {
  kDTeam = 0, kDLength, kDHeadX, kDHeadY, kDFacing, kDRank, kDActed, kDId, kDValid, kCriticDragonFields
};
enum CriticGlobalField : int {
  kQRound = 0, kQWidth, kQHeight, kQUnitLimit,
  kQLongestA, kQLongestB, kQTotalA, kQTotalB, kQCountA, kQCountB,
  kQNextTeam, kQNextRank, kQAlive, kQDue, kQNextLength, kCriticGlobalFields
};

// Writes the event's grid of cells, living dragons (up to d_pad; returns how
// many there are) and globals. The game must be waiting on an actor.
int32_t EncodeEvent(const dragons::Game& game, int32_t* cells, int32_t d_pad, int32_t* dragon_rows, int32_t* global);

}  // namespace rl
