// Collecting experience: every env's acting dragon, every step, through the
// `act` program; transitions stored per dragon in chunks that the update
// replays from exact start states.
//
// Per-dragon state (GRU h, memory map, bin sums) lives in a pool of slots on
// the rollout device. A dragon gets a slot on its first turn and gives it back
// when it dies or its game ends. If the pool runs out, the dragon plays that
// turn from a zero state in a scratch slot (and that turn is its own chunk),
// which is counted in the stats.
//
// Returns: the reward is the game's outcome for the dragon's team (+1 win,
// -1 loss, 0 draw), at the game's end. A dragon's turns are a trajectory,
// discounted per round (gamma^rounds between its turns); after its last turn
// in a rollout it bootstraps: from the outcome if the game ended in this
// rollout (dead or alive: a dragon that died still shares its team's result),
// otherwise from the critic's value of the env's position at the rollout's
// end, negated for the team not acting there (the game is zero-sum).
#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "artifacts.hpp"
#include "dragons/vec_env.hpp"
#include "pjrt.hpp"

namespace train {

struct Storage {
  int64_t record_bytes = 0;
  // one entry per transition (a dragon's turn)
  std::vector<uint8_t> records;          // [n, record_bytes]
  std::vector<int32_t> action, round;
  std::vector<float> log_prob, value, advantage, returns;
  std::vector<int64_t> next;             // the same dragon's next transition, -1
  std::vector<uint8_t> end;              // for a last transition: 1 outcome, 2 bootstrap
  std::vector<float> end_value;          // the outcome or the bootstrap value
  std::vector<int32_t> end_round;
  // chunks: up to `chunk` consecutive turns of one dragon, replayed from a saved start state
  struct Chunk {
    int64_t state;
    std::vector<int64_t> rows;
  };
  std::vector<Chunk> chunks;
  std::vector<float> start_h;            // [k, hidden]
  std::vector<uint16_t> start_map;       // [k, cells, channels] float16 bits
  std::vector<float> start_bins;         // [k, bins, channels]

  int64_t size() const { return static_cast<int64_t>(action.size()); }
  void Clear();
};

struct RolloutOptions {
  int steps = 256;          // env steps per rollout (N transitions each)
  double gamma = 0.999;     // per round
  double lambda = 0.95;     // per turn
  int threads = 0;
  uint64_t seed = 0;
};

struct RolloutStats {
  int64_t transitions = 0, games = 0, wins[2] = {0, 0}, draws = 0, rounds = 0, eliminations = 0;
  int64_t deaths[5] = {0, 0, 0, 0, 0};
  int64_t splits = 0, sprints = 0, overflow = 0, max_slots = 0;
  double seconds = 0, act_seconds = 0, env_seconds = 0, gather_seconds = 0;
};

class Rollout {
 public:
  Rollout(const pjrt::Runtime& rt, const Artifacts& art, const std::vector<std::string>& maps, int device,
          const RolloutOptions& opts);

  // Plays opts.steps steps with `params` (on this rollout's device) into `out`
  // (cleared first) and computes advantages and returns.
  RolloutStats Collect(const std::vector<pjrt::Buffer>& params, Storage& out);
  int device() const { return device_; }

 private:
  struct Track {                 // a dragon of an env's current game
    int32_t slot = -1;
    uint8_t team = 0;
    int64_t chunk = -1;          // its current chunk in this rollout
    int32_t chunk_turns = 0;
    int64_t last = -1;           // its latest transition in this rollout
  };
  struct PendingGather {
    int32_t slot;
    bool fresh;
    int64_t state;
  };

  int32_t AllocSlot();
  void FreeSlot(int32_t slot);
  void EndGame(int env, Storage& out, RolloutStats& stats);
  void RunGathers(std::vector<PendingGather>& pending, Storage& out);
  void ComputeAdvantages(Storage& out) const;

  const pjrt::Runtime& rt_;
  const Artifacts& art_;
  int device_;
  RolloutOptions opts_;
  int64_t N_, S_, R_, G_, H_, C_, M_, B_, chunk_;
  dragons::VecEnv env_;
  pjrt::Program act_, value_, gather_;
  pjrt::Buffer pool_h_, pool_map_, pool_bins_, key_;
  std::vector<int32_t> free_slots_;
  int32_t scratch_;
  std::vector<std::unordered_map<int32_t, Track>> tracks_;   // [env] dragon id -> track
  std::mt19937_64 rng_;
};

}  // namespace train
