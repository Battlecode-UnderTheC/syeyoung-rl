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

// Per board, fixed: [n_pad, 2] (edge types packed 2 bits per direction as
// 1 empty 2 kelp 3 portal, valid) and the true destinations [n_pad, 4] (-1 kelp).
void EncodeBoard(const dragons::Board& board, int32_t n_pad, int32_t* tile, int32_t* dest);

// Per event, the tile grid [n_pad]: bit 0 pearl, bits 1-3 occupant
// (0 empty, 1 A head, 2 A body, 3 B head, 4 B body), bit 4 bed, bits 8.. the
// countdown + 1024 (beds only).
inline constexpr int kCountdownBias = 1024;

// One row per living dragon, by id.
enum CriticDragonField : int {
  kDTeam = 0, kDLength, kDHeadX, kDHeadY, kDFacing, kDRank, kDActed, kDId, kDValid, kCriticDragonFields
};
enum CriticGlobalField : int {
  kQRound = 0, kQWidth, kQHeight, kQUnitLimit,
  kQLongestA, kQLongestB, kQTotalA, kQTotalB, kQCountA, kQCountB,
  kQNextTeam, kQNextRank, kQAlive, kQDue, kQNextLength, kCriticGlobalFields
};

// Writes the event's grid, living dragons (up to d_pad; returns how many there
// are) and globals. The game must be waiting on an actor.
int32_t EncodeEvent(const dragons::Game& game, int32_t n_pad, int32_t* grid, int32_t d_pad, int32_t* dragon_rows,
                    int32_t* global);

}  // namespace rl
