// One rollout's data, read-only once built. The trajectory structure (which
// turns belong to one dragon, how each trajectory ends) is explicit.
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "agent/actions.hpp"
#include "train/agents.hpp"

namespace train {

using Row = int64_t;         // a turn's index in the batch

// How a dragon's last turn in the batch continues.
enum class TurnEnd : uint8_t {
  Continues,                 // it has a next turn in this batch
  Outcome,                   // its game ended in this batch: the team's result is its return
  Bootstrap,                 // still running when the rollout ended: the critic's value
  Untrained,                 // played without its GRU state (no free slot): not trained on
};

struct TurnRecord {
  agent::ActionIndex action;
  float log_prob;
  float value;
  int32_t round;
};

struct Ending {
  TurnEnd kind;
  float value;               // the outcome, or the bootstrap value
  int32_t round;             // when that value applies
};

// Up to ModelConfig::chunk_turns consecutive turns of one dragon, replayed
// from `start_h` (its GRU state before the first of them).
struct Chunk {
  std::span<const Row> rows;
  std::span<const float> start_h;
};

class Batch {
 public:
  int64_t turns() const;
  std::span<const uint8_t> Features(Row r) const;
  std::span<const uint8_t> Critic(Row r) const;
  const TurnRecord& Turn(Row r) const;
  std::optional<Row> Next(Row r) const;          // the same dragon's next turn
  std::optional<Ending> End(Row r) const;        // present on a dragon's last turn in the batch
  std::span<const Chunk> chunks() const;

 private:
  friend class BatchBuilder;
  // storage is contiguous, reserved up front from the rollout's size
};

// Filled by the collector; turns into a Batch once, then nothing can change it.
class BatchBuilder {
 public:
  BatchBuilder(int64_t expected_turns, int64_t feature_bytes, int64_t critic_bytes, int32_t chunk_turns,
               int32_t hidden);
  Row Add(const AgentKey& who, std::span<const uint8_t> features, std::span<const uint8_t> critic,
          const TurnRecord& turn, bool starts_chunk);
  void SetStartState(Row chunk_start, std::span<const float> h);
  void End(const AgentKey& who, const Ending& how);   // the dragon's latest row
  Batch Finish() &&;                                  // every trajectory must have ended (checked)
};

}  // namespace train
