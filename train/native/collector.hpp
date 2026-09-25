// Plays a batch of complete matches, one dragon-turn per match per step, and
// keeps what training needs (spec section 9.1):
//
//   * every lifetime's facts (one Lifetime per dragon, created at its first
//     turn, dropped at its death) and its latest persistent state h;
//   * for learner lifetimes, the raw observations and issued commands, so
//     training replays them from birth (Replay);
//   * per event (the boundary before each dragon-turn), the actor, both old
//     values, and the command applied, with a game snapshot every 64 events,
//     so the critic's inputs are rebuilt losslessly (Replay::EncodeEvents);
//   * every decision's h and seen fraction, the critic's per-dragon summary.
//
// The caller (Python) owns the policy: each step it asks which matches wait on
// a decision, has them encoded into its buffers, runs the networks and hands
// back the commands.
#pragma once

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "critic.hpp"
#include "facts.hpp"
#include "dragons/vec_env.hpp"

namespace rl {

inline constexpr int kHidden = 192;
inline constexpr int kSnapshotEvery = 64;
inline constexpr int32_t kProtocol = 3;       // every reply declares it: the ECHOES line is on

// Scripted evaluation opponents (validation and test only), as policy slots
// below zero. Deterministic given the match seed; they never sonar or split.
enum Scripted : int32_t {
  kGreedy = -1,   // "greedy-v1": a step that is neither kelp nor a body in view, toward the nearest pearl in view
  kRandom = -2,   // "random-v1": a uniformly random such step
};

struct MatchSetup {
  std::string map_text;
  int32_t policy[2] = {0, 0};   // the policy slot playing each team (the caller's numbering), or Scripted
  bool record[2] = {true, true};// keep that team's observations for training
  uint64_t seed = 0;            // scripted opponents' randomness
};

// What the caller needs to size and route one waiting decision.
struct Pending {
  int32_t match = -1, dragon = -1, team = 0, policy = 0, record = 0;
  int32_t tiles = 0, length = 0, inbox = 0, echo_bytes = 0;
};

struct EventRecord {
  int32_t dragon;
  Command command;
  float value[2] = {0, 0};     // V_old from each team's perspective (NaN where not evaluated)
};

struct LifetimeRecord {
  int32_t match = 0, dragon = 0, team = 0;  // match: index in its Pack / Replay
  bool recorded = false;                    // observations kept (learner side)
  std::vector<Observation> observations;    // recorded only
  std::vector<Command> commands;            // recorded only
  std::vector<float> logp;                  // behaviour joint log-probabilities, recorded only
  std::vector<int32_t> events;              // event index of each decision
  std::vector<std::array<float, kHidden>> h;// h after each decision
  std::vector<float> seen;                  // private seen fraction at each decision
  int8_t death = -1;                        // engine death cause if it died, -1 alive at the end
};

struct MatchRecord {
  std::shared_ptr<const dragons::Board> board;
  MatchSetup setup;
  std::vector<EventRecord> events;
  std::vector<dragons::Game> snapshots;     // before events 0, 64, 128, ...
  std::vector<LifetimeRecord> lives;        // every lifetime of the match
  std::vector<int32_t> lifetimes;           // dragon id -> index into lives, -1
  bool finished = false;
  int8_t winner = -1;                       // -1 draw, 0 A, 1 B
  int32_t rounds = 0;
  // deaths by team and engine cause (wall, self, other body, head-to-head, no valid action)
  std::array<int32_t, 2 * 5> deaths{};
};

// Finished matches, moved out of a Collector for training.
struct Pack {
  std::vector<MatchRecord> matches;
};

// Integer buffers for a batch of decisions, row b of each (caller-owned, C order).
struct ActorBatch {
  int32_t n_pad = 0, k_pad = 0, e_pad = 0;  // tiles, inbox words, echo bytes per field
  int32_t *tile, *dest, *entity, *segment, *status, *prev_dirs;
  uint8_t *prev_words, *inbox, *echo;       // inbox [B, k_pad, 8]; echo [B, 5, e_pad]
  int32_t *echo_len;                        // [B, 5]
  float* h;                                 // [B, kHidden] h_old
};

struct CriticBatch {
  int32_t n_pad = 0, d_pad = 0;
  int32_t *board_tile, *board_dest, *grid, *dragons, *global;
  float* summary;                           // [B, d_pad, kHidden + 1]: h, seen fraction
};

void EncodeObservation(const Lifetime& life, const Observation& o, const ActorBatch& out, int64_t row);

class Collector {
 public:
  Collector(std::vector<MatchSetup> setups, int threads);

  int32_t num_matches() const { return static_cast<int32_t>(games_.size()); }
  bool done() const;                         // every slot finished
  bool finished(int32_t m) const { return records_[m].finished; }
  // Observes every unfinished match's actor (facts steps 1-3) and lists them.
  std::vector<Pending> Prepare();
  // Encodes the prepared decisions of `matches` into rows 0.. of `out` / `critic`.
  void EncodeActors(const std::vector<int32_t>& matches, const ActorBatch& out);
  void EncodeCritics(const std::vector<int32_t>& matches, const CriticBatch& out);
  // Commits and plays the commands of `matches`. value[b] is V_old from the
  // actor's team's perspective (NaN if not evaluated); the other team's is -value.
  void Apply(const std::vector<int32_t>& matches, const std::vector<Command>& commands, const float* logp,
             const float* h_new, const float* value);
  // A new match in slot m, which must be finished; the old record is kept for Harvest.
  void Reset(int32_t m, const MatchSetup& setup);
  // Every finished match not yet harvested, moved out (slots stay finished until Reset).
  std::shared_ptr<Pack> Harvest();

  const std::vector<MatchRecord>& match_records() const { return records_; }

 private:
  struct Live {
    std::unique_ptr<Lifetime> life;
    Observation obs;
    std::array<float, kHidden> h{};
    float seen = 0;
  };
  void Start(int32_t m, const MatchSetup& setup);
  void PlayScripted(int32_t m);
  std::vector<dragons::Game> games_;
  std::vector<std::vector<Live>> live_;     // [match][dragon id]
  std::vector<MatchRecord> records_;
  std::vector<uint8_t> prepared_, harvested_;
  std::vector<MatchRecord> pending_;        // finished and replaced, awaiting Harvest
  dragons::Pool pool_;
};

// Replays finished matches for training: learner lifetimes from birth with
// fresh facts, and the critic's events.
class Replay {
 public:
  // `keep`: which recorded lifetimes (indices into the recorded ones, in pack
  // order) this replay covers; none given keeps them all.
  Replay(std::vector<std::shared_ptr<const Pack>> packs, int threads,
         const std::optional<std::vector<int32_t>>& keep = std::nullopt);

  int32_t num_lifetimes() const { return static_cast<int32_t>(lives_.size()); }
  const LifetimeRecord& lifetime(int32_t i) const { return *lives_[i]; }
  int32_t lifetime_match(int32_t i) const { return life_match_[i]; }
  // Fresh facts for every lifetime (the start of an epoch).
  void Reset();
  // Decisions [start, start + count) of lifetime `which[b]` into rows
  // b * steps + t, advancing each lifetime's facts. A lifetime's chunks must
  // come in order within an epoch.
  void EncodeChunk(const std::vector<int32_t>& which, const std::vector<int32_t>& start, int32_t steps,
                   const ActorBatch& out);

  // Decision `step[e]` of lifetime `which[e]` into row e, for every e with
  // which[e] >= 0. Each lifetime's steps must continue where its facts stand
  // (in order across calls), which a packed row of consecutive chunks does.
  void EncodeList(const std::vector<int32_t>& which, const std::vector<int32_t>& step, const ActorBatch& out);

  int32_t num_matches() const { return static_cast<int32_t>(matches_.size()); }
  const MatchRecord& match(int32_t m) const { return *matches_[m]; }
  // Critic inputs of the given (sorted) events of `match` into rows 0..
  void EncodeEvents(int32_t match, const std::vector<int32_t>& events, const CriticBatch& out) const;

 private:
  std::vector<std::shared_ptr<const Pack>> packs_;
  std::vector<const MatchRecord*> matches_;
  std::vector<const LifetimeRecord*> lives_;          // recorded lifetimes kept
  std::vector<int32_t> life_match_;
  std::vector<std::unique_ptr<Lifetime>> facts_;
  std::vector<int32_t> next_;
  dragons::Pool pool_;
};

}  // namespace rl
