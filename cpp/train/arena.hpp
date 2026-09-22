// Many arenas of one board size, stepped in lockstep against GPU-resident
// per-lifetime state (spec 8, 12).
//
// Within a round dragons act one at a time in id order, so what is batched is
// a *slot*: dragon index k of every arena. The CPU side reads each acting
// dragon's window out of its sim, advances its private factual state, and
// emits the changed cells; the GPU applies them, derives the actor's map,
// runs the encoder and the decoder, and the sampled commands go back into the
// sims. A slot is (arena, dragon index), so children born during a round act
// in later slots of the same round exactly as the engine schedules them.
//
// Recorded arenas keep everything PPO needs: the raw updates and small
// tensors per decision, a snapshot of the slot's raw store, M and h every
// `snapshot_every` turns of a lifetime, the stored command and log
// probability, and the exact critic state at every event.
#pragma once

#include <deque>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "agent/spatial/memory.hpp"
#include "dragons/game.hpp"
#include "train/gpu_state.hpp"
#include "train/model.hpp"

namespace dragons::train {

struct TurnRecord {
  int event;                  // index into the arena's event list
  int round;
  int turn_index;             // this lifetime's turn count before acting
  std::vector<RawUpdate> updates;
  std::vector<float> status, history, entities;   // [36], [32*30], [49*16]
  std::vector<uint8_t> history_mask, entity_mask; // [32], [49]
  std::vector<int64_t> visible_xy;                // [98]
  int head_x, head_y;
  std::vector<std::array<int64_t, 4>> portal_pairs;
  std::vector<int64_t> portal_axis;
  std::vector<std::array<int64_t, 2>> portal_unpaired;
  std::vector<uint32_t> inbox;
  // stored command (row of StoredCommand), on the CPU
  int64_t top, split_index, sonar;
  std::vector<int64_t> tokens;                    // [K]
  std::vector<uint8_t> active;                    // [K]
  std::vector<uint8_t> masks;                     // [K*5]
  std::vector<float> prefix;                      // [K*49]
  std::vector<int64_t> split_sizes;               // [C]
  std::vector<float> split_feats;                 // [C*4]
  bool split_legal;
  float log_prob;
};

struct Snapshot {
  int turn_index;
  torch::Tensor raw;     // [20, H, W] cpu
  torch::Tensor M;       // [16, H, W] cpu
  torch::Tensor h;       // [128] cpu
};

struct LifetimeRecord {
  int dragon_id, team;
  bool learner;
  std::vector<TurnRecord> turns;
  std::vector<Snapshot> snapshots;
  bool dead = false;
};

struct EventRecord {
  int round;
  int acting_team;
  int delta_rounds = 0;
  bool terminal = false;
  float reward[2] = {0.0f, 0.0f};
  float value[2] = {0.0f, 0.0f};        // V_old, filled by the critic at collection
  torch::Tensor map_grid;               // [20, H, W] half, cpu
  torch::Tensor dragons;                // [N, 16] float
  torch::Tensor dragon_team;            // [N] long
  torch::Tensor team_feats;             // [2, 4]
  torch::Tensor phase;                  // [3]
};

struct ArenaRecord {
  std::map<int, LifetimeRecord> lifetimes;   // by dragon id
  std::vector<EventRecord> events;
  int arena;
};

struct GroupOptions {
  int arenas = 1024;
  int max_slots = 16;
  int record_arenas = 64;
  int snapshot_every = 16;
  int max_steps = 8;
  bool sonar = false;
  // team A of every arena is the learner; team B is the learner in self-play
  // arenas and a frozen opponent elsewhere (opponent handling lives in the driver)
  bool self_play = true;
  uint64_t seed = 0;
};

class ArenaGroup {
 public:
  ArenaGroup(int width, int height, std::vector<std::string> maps, const GroupOptions& opts,
             Model& model, torch::Device device);

  int width() const { return W_; }
  int height() const { return H_; }
  int arenas() const { return static_cast<int>(games_.size()); }
  int64_t decisions() const { return decisions_; }

  // One full round for every arena: pearls tick, then slot by slot.
  void PlayRound();
  bool AllFinished() const;
  // Restart the arenas whose match is over (or all of them), returning how many.
  int ResetFinished(bool all = false);

  // Records of the recorded arenas; the driver takes them at the segment end.
  std::vector<ArenaRecord> TakeRecords();
  // V(S) of each recorded arena's current state, for bootstrapping a cutoff.
  std::vector<std::array<float, 2>> BootstrapValues();

  std::vector<std::string> Results() const;

 private:
  struct Slot {
    std::unique_ptr<spatial::LifetimeState> state;
    std::vector<spatial::Cell> old_body;
    spatial::Cell old_head{0, 0};
    bool had_head = false;
    int turns = 0;
  };
  // slots are allocated per lifetime from a free list: dragon indices grow
  // with every split and are never reused, living dragons are what is bounded
  int SlotIndex(int arena, int index) const { return arena * opts_.max_slots + slot_of_[arena][index]; }
  Slot& SlotFor(int arena, int index) { return slots_[arena][slot_of_[arena][index]]; }
  void ResetArena(int arena);
  void SpawnNew(int arena);
  void Reap(int arena);
  void PlaySlot(int k);
  void RecordEvent(int arena, int index);
  void CriticBatch(const std::vector<int>& arenas, const std::vector<int>& indices,
                   std::vector<EventRecord*>& out_events);

  int W_, H_;
  GroupOptions opts_;
  Model& model_;
  torch::Device device_;
  std::vector<std::string> maps_;
  std::mt19937_64 rng_;
  std::vector<Game> games_;
  std::vector<std::vector<int>> portal_ids_;
  std::vector<std::vector<Slot>> slots_;         // [arena][slot]
  std::vector<std::vector<int>> slot_of_;        // [arena][dragon index] -> slot
  std::vector<std::vector<int>> free_slots_;     // [arena]
  int max_index_ = 0;                            // highest dragon index in any arena
  std::vector<int> known_dragons_, known_deaths_;
  std::vector<ArenaRecord> records_;             // one per recorded arena
  std::vector<int> record_of_;                   // arena -> index into records_ or -1
  torch::Tensor raw_, M_, h_;
  int64_t decisions_ = 0;
  std::vector<float> status_buf_, history_buf_, entity_buf_;
  std::vector<uint8_t> hmask_buf_, emask_buf_;
  std::vector<int64_t> xy_buf_;
};

}  // namespace dragons::train
