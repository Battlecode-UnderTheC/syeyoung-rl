// Everything the trainer keeps per dragon while its game runs: its GRU slot,
// its FactMap, and where its turns go in the batch being built.
#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include "agent/fact_map.hpp"
#include "train/slots.hpp"

namespace train {

struct AgentKey {
  int32_t env;
  int64_t game;              // counts games per env, so ids never collide across games
  int32_t dragon;
  friend bool operator==(const AgentKey&, const AgentKey&) = default;
};

struct Agent {
  AgentKey key;
  agent::Team team;
  std::optional<Slot> slot;  // absent: dead, or waiting for a free slot
  agent::FactMap memory;
  bool alive = true;
};

// The agents of one collector's envs.
class AgentTable {
 public:
  AgentTable(int32_t envs, SlotPool& slots);

  // The dragon about to act: created on its first turn (with a slot if one is
  // free, and an empty FactMap).
  Agent& Acting(int32_t env, const agent::SpawnInfo& self, bool first_turn);
  void Died(int32_t env, int32_t dragon);        // releases its slot; keeps it until the game ends
  std::vector<Agent> EndGame(int32_t env);       // releases every slot; returns the game's agents
  std::vector<const Agent*> Alive(int32_t env) const;
};

}  // namespace train
