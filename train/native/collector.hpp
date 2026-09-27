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
#include <deque>
#include <random>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "critic.hpp"
#include "facts.hpp"
#include "dragons/vec_env.hpp"

namespace rl {

inline constexpr int kHidden = 192;
// Training replays chunks of kChunk decisions, each from its stored starting h:
// only those (a lifetime's h after decisions kChunk-1, 2 kChunk-1, ...) are kept.
inline constexpr int kChunk = 16;
// Encoding caps (collection and training alike): the latest kMaxInbox inbox
// words, the first kMaxEcho bytes of each echo field.
inline constexpr int kMaxInbox = 64;
inline constexpr int kMaxEcho = 32;
inline constexpr int kSnapshotEvery = 1024;   // game copies are large: long matches kept too many
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
  int32_t tiles = 0, length = 0, inbox = 0, echo_bytes = 0, alive = 0;
};

struct EventRecord {   // trivially copyable: packs serialize it as raw bytes
  int32_t dragon;
  Command command;
  float value[2] = {0, 0};     // V_old from each team's perspective (NaN where not evaluated)
};

static_assert(std::is_trivially_copyable_v<EventRecord>);

struct LifetimeRecord {
  int32_t match = 0, dragon = 0, team = 0;  // match: index in its Pack / Replay
  bool recorded = false;                    // observations kept (learner side)
  std::vector<Observation> observations;    // recorded only
  std::vector<Command> commands;            // recorded only
  std::vector<float> logp;                  // behaviour joint log-probabilities, recorded only
  std::vector<int32_t> events;              // event index of each decision
  std::vector<std::array<float, kHidden>> h;// h after decisions kChunk-1, 2 kChunk-1, ... (chunk starts)
  std::vector<float> seen;                  // private seen fraction at each decision
  int8_t death = -1;                        // engine death cause if it died, -1 alive at the end
  int16_t rammed = 0;                       // enemies that died hitting this dragon's body
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

// A Pack as bytes (to move it between processes): setups, events, results and
// the per-decision numbers only. Games are deterministic given the board and
// the commands, so the snapshots and the learner's observations are rebuilt
// by replaying each match (checked: every event's actor must match).
std::string SerializePack(const Pack& pack);
std::shared_ptr<Pack> DeserializePack(const std::string& bytes);
// A pack collected without observations (keep_observations false): replays its
// matches to rebuild the snapshots and the recorded lifetimes' observations and commands.
void RebuildPack(Pack& pack);
void RebuildMatch(MatchRecord& match);

// Integer buffers for a batch of decisions, row b of each (caller-owned, C order).
struct ActorBatch {
  int32_t k_pad = 0, e_pad = 0;             // inbox words, echo bytes per field
  int32_t *win, *wdest, *grid;              // [B, kWinTiles, kTileFields], [B, kWinTiles, 4], [B, kGridCells, kGridFields]
  int32_t *entity, *segment, *status, *prev_dirs;
  uint8_t *prev_words, *inbox, *echo;       // inbox [B, k_pad, 8]; echo [B, 5, e_pad]
  int32_t *echo_len;                        // [B, 5]
  float* h;                                 // [B, kHidden] h_old
};

struct CriticBatch {
  int32_t d_pad = 0;
  int32_t *cells, *dragons, *global;        // [B, kCriticCells, kCriticCellFields], [B, d_pad, ...], [B, ...]
  float* summary;                           // [B, d_pad, kHidden + 1]: h, seen fraction
};

void EncodeObservation(const Lifetime& life, const Observation& o, const ActorBatch& out, int64_t row,
                       bool reference = false);


class Collector {
 public:
  // keep_observations false: recorded lifetimes keep only logp, h, seen and
  // events, and matches keep no snapshots (their packs travel serialized; the
  // observations and snapshots are rebuilt by replay).
  Collector(std::vector<MatchSetup> setups, int threads, bool keep_observations = true);

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

  // ---- the native step loop (collection workers) ----------------------------------
  // Row m of a step's buffers is match m. A finished match is replaced at once,
  // on the next queued map (or a recent one): self-play (both sides the
  // learner, slot 0, both recorded) with probability self_play, else the
  // learner against slot 1 (the worker's current opponent) on a random side.
  void PushMaps(std::vector<std::string> maps);
  size_t queued_maps() const { return maps_.size(); }
  void SetMatchmaking(double self_play, uint64_t seed);
  // Refills, prepares and encodes the waiting decisions: the learner's (slot 0)
  // into rows 0.. of `learner` (at most l_cap), the opponent's into rows 0.. of
  // `opponent` (at most o_cap; more wait for a later step). counts[0..2]: learner
  // rows, opponent rows, critic rows. The critic runs on at most critic_cap
  // learner decisions whose match value is critic_every of its turns old
  // (oldest first): encoded into rows 0.. of `critic`, their learner row in crow,
  // perspective team in cteam.
  void StepEncode(const ActorBatch& learner, int32_t l_cap, const ActorBatch& opponent, int32_t o_cap,
                  const CriticBatch& critic, int32_t critic_cap, int32_t critic_every, int32_t* crow, int32_t* cteam,
                  int32_t* counts);
  // Plays the encoded decisions. Command rows: the learner rows, then o_cap
  // opponent rows (row l_cap + j): kind, nsteps, dirs [R, 8], split, sonar [R, 4]
  // (0 silent, 1 + j sends word j), logp, h_new [R, kHidden] (a nonfinite row:
  // h zeroed, logp NaN); value: the critic rows' V (from cteam's side). Other
  // recorded decisions take their match's latest V (negated for the other team).
  void StepApply(const int32_t* kind, const int32_t* nsteps, const int32_t* dirs, const int32_t* split,
                 const int32_t* sonar, const float* logp, float* h_new, const float* value);
  int64_t turns() const { return turns_; }
  int64_t nonfinite() const { return nonfinite_; }

 private:
  struct Live {
    std::unique_ptr<Lifetime> life;
    Observation obs;
    std::array<float, kHidden> h{};
    std::array<float, kHidden> h_chunk{};    // h at the start of its current chunk: the critic's summary
    float seen = 0;
  };
  void Drop(Live& L);               // a lifetime ends: its facts go
  void DropAll(int32_t m);
  void Start(int32_t m, const MatchSetup& setup);
  void PlayScripted(int32_t m);
  std::vector<dragons::Game> games_;
  std::vector<std::vector<Live>> live_;     // [match][dragon id]
  std::vector<MatchRecord> records_;
  std::vector<uint8_t> prepared_, harvested_;
  std::vector<MatchRecord> pending_;        // finished and replaced, awaiting Harvest
  bool keep_observations_;
  MatchSetup NextSetup();
  std::deque<std::string> maps_;
  std::vector<std::string> recent_maps_;
  double self_play_ = 0.5;
  std::mt19937_64 rng_{0x5eed};
  std::vector<float> v_value_;                 // the match's latest V, from team v_team_'s side
  std::vector<int32_t> v_team_, v_age_;        // and its age in the match's turns
  std::vector<int32_t> step_active_, step_row_, step_critic_;   // matches, their command rows, critic matches
  int32_t critic_every_ = 8;
  int64_t turns_ = 0, nonfinite_ = 0;
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
                   const ActorBatch& out, bool reference = false);

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
