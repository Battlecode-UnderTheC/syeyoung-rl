# Trainer redesign

This is a design for review; nothing is implemented. It replaces the current `train/` trainer.

It has three aims:

- **Throughput:** millions of dragon-turns per second through the env, with the GPUs busy.
- **One code path for training and the bot:** a dragon perceives and acts through the same code in both.
- **A structure that stays maintainable:** each concern lives in exactly one place.

---

## 1. Layers

```
trainer ──► rl ──► game          (dragons-env)
   │         ├───► agent ──► game (types only)
   │         └───► policy
   └───────► policy               (construct, sync weights, save/load)

bot ──────► agent  +  a small forward pass of the model (later)

model  (JAX, train/model/, offline) ──exports──► artifacts ──loaded by──► policy
```

| layer | knows | must not know |
|---|---|---|
| **game** | rules, the judge's observation | agents, models, training |
| **agent** | one dragon's memory, the model's input encoding (actor, plus the critic's full-state encoding), the action mask, action → engine action | GPUs, the model's internals, trajectories, returns |
| **policy** | running the exported model on GPUs, per-dragon recurrent state, parameters | what features mean, games, trajectories |
| **rl** | trajectories, returns, PPO, dragon lifetimes | devices, memory, feature layout, JAX |
| **trainer** | configuration, devices, the loop, checkpoints, logs | anything the layers already own |

**Rules:**

1. Dependencies never point upward and never form a cycle. A layer may use any layer below it. Each layer is its own CMake library that sees only the include paths of the layers it may use, so an upward include fails to build.
2. Each layer has exactly one public header. Everything else lives in that layer's `src/`.
3. Layers exchange plain data: bytes, integers, and spans into buffers someone else owns. Every call covers a whole batch, and data is written where the next layer reads it.
4. Knowledge stays in its owning layer. Only `agent` knows memory exists. Only `policy` knows about JAX, PJRT or program argument names. Only `policy` and the trainer, which assigns GPUs, know about devices. Only `rl` knows about trajectories.

**On disk:** each layer is one directory and one CMake target, and it links only to the layers it's allowed to use.

```
dragons-env/        game      target dragons       (existing submodule)
train/agent/        agent     target agent         links dragons
train/policy/       policy    target policy        links nothing of ours (PJRT is loaded at run time)
train/model/        model     Python (JAX), offline; not a C++ target
train/rl/           rl        target rl            links dragons, agent, policy
train/trainer/      trainer   executable trainer   links rl, policy, agent, dragons
train/bot/          bot       judge submission     agent + a small forward pass (later)
```

Each directory has one public header, `<layer>/<layer>.hpp` (the game's is `dragons/vec_env.hpp`), then `src/` and `tests/`.

`agent` and `policy` never include each other. Their one shared contract is the byte layout of the model's input, which the trainer checks at start-up: `agent::Agents::input_spec().description == policy::Model::input_description()`.

---

## 2. Three ideas the design rests on

**Handles.** Every live dragon holds a small integer, its handle, from its first turn until it dies or its game ends. `rl` hands handles out and takes them back; nothing else tracks lifetimes. Both per-dragon stores are indexed by handle:

- `agent`: the dragon's memory, in host RAM;
- `policy::Actor`: the dragon's recurrent state, on the GPU. This is just the GRU, 2 KB per dragon, so a million handles fit.

A collector's `Agents`, `Actor` and `Rollout` are built with the same capacity, and the trainer passes it to all three.

- **Sizing:** the default capacity is the worst case, envs × 2 × the largest unit limit among its boards. For 2,048 envs and a limit of 64, that's 262k handles: 0.5 GB of GPU state. Host memory is taken only for live dragons, sized to their board.
- **Fallback:** rl reserves handle 0 as scratch (`agent` and `policy` just see a handle). A dragon that finds no free handle plays on the scratch handle, whose memory and state are reset every turn, and its turns are recorded as `Untrained`. That only happens if a map sets an unusually large unit limit.

**One collector per GPU.** Each collecting GPU gets its own share of the envs (a `VecEnv`), its own `Agents`, its own `Actor`, and its own handle space (handles are numbered per collector). No object ever needs to map a dragon to a device. Collectors run in parallel threads, and the trainer splits the CPU cores between them: each collector's `VecEnv` and `Agents` get cores ÷ collectors threads, so four collectors don't each start a full pool.

**Actor and learner are separate objects.** An actor does inference for its dragons on one GPU. The learner trains data-parallel across GPUs. Parameters flow one way, from the learner to the actors, as a `Weights` snapshot.

---

## 3. Public interfaces

### game (dragons-env, `dragons/vec_env.hpp`)

This is the existing API, with these changes:

```cpp
struct Action {                                   // replaces Act(kind, steps, nsteps, split, sonar)
  struct Move { std::array<uint8_t, 8> steps; uint8_t count; };   // directions 0..3, 1..8 of them; no allocation
  struct Split { int32_t child_segments; };
  struct None {};                                 // the engine's "no valid action"
  std::variant<Move, Split, None> what;
  std::optional<uint32_t> sonar;
};

struct ObservationHeader { int32_t dragon, team, turn, round, width, height, unit_limit; };  // turn 0 = just spawned
struct Death   { int32_t env, dragon, team, round; DeathReason reason; };
struct GameEnd { int32_t env; Result result; int32_t final_round; };   // winner, end reason, both teams' standing

struct StepResult {                               // views, valid until the next Step
  std::span<const Death> deaths;                  // env order, then engine order
  std::span<const GameEnd> ended;                 // env, result, final round (the env has restarted it)
};

class VecEnv {
 public:
  VecEnv(int32_t envs, std::function<std::string(std::mt19937_64&)> next_board, int threads, uint64_t seed);
  std::span<const uint8_t> observations() const;  // [envs, record]: each env's acting dragon
  int64_t record_bytes() const;
  ObservationHeader header(int32_t env) const;    // id, team, turn, round, board size
  StepResult Step(std::span<const Action> actions);
  const Game& game(int32_t env) const;            // full state, read-only
};
```

Removed from dragons-env:

- the critic's state summary, which moves to `agent`;
- the public data members, which results replace;
- the exposed observer internals.

Boards come from a function the trainer supplies, drawing contest files or generated boards.

### agent (`agent/agent.hpp`)

```cpp
using Handle = int32_t;
// Batches are dense rows in someone else's buffer: observations are record_bytes
// each (from the env), features and critic records their InputSpec sizes.

class Agents {
 public:
  Agents(int32_t capacity, int32_t threads);     // threads: this collector's share of the cores
  int32_t capacity() const;
  static InputSpec input_spec();                  // feature bytes, critic bytes, description

  void Begin(Handle, int32_t board_width, int32_t board_height);   // empty memory
  void End(Handle);

  // many dragons at once, threaded inside; row i is handles[i].
  // Each observation must be perceived exactly once: Perceive advances the dragon's memory.
  void Perceive(std::span<const Handle>, std::span<const uint8_t> observations, int64_t record_bytes,
                std::span<uint8_t> features);
  void EncodeCritic(const dragons::VecEnv&, std::span<uint8_t> critic) const;   // training only
  void Decide(std::span<const Handle>, std::span<const int32_t> action, std::span<dragons::Action> out);
};
```

Private to this layer:

- each dragon's memory (section 5);
- feature encoding, including the action mask;
- the critic's encoding;
- the action table.

The bot uses this same class for one dragon (capacity 1), with observations parsed from the judge's text.

### policy (`policy/policy.hpp`)

```cpp
class Runtime { Runtime(plugin_path, memory_fraction); };        // the process's GPUs
class Model   { Model(artifacts_dir);                            // programs, shapes, input description
                input_description(), feature_bytes(), critic_bytes(), state_bytes(),
                rows(), chunk_turns(), chunks_per_update(); };
class Weights;                                                   // parameters, device-independent snapshot
struct Acted { std::span<const int32_t> action; std::span<const float> log_prob, value; };   // valid until the next call

class Actor {                                                    // one GPU, `capacity` handles of state
 public:
  Actor(Runtime&, const Model&, int device, int32_t capacity);
  int32_t capacity() const;
  void SetWeights(const Weights&);
  // Protocol: fill features() and critic() (rows() records each, row i for handles[i]),
  // then call Act or Value, which read them. The buffers are pinned and reused.
  std::span<uint8_t> features();
  std::span<uint8_t> critic();
  Acted Act(std::span<const Handle>, std::span<const uint8_t> fresh);      // action, log_prob, value
  std::span<const float> Value(std::span<const Handle>, std::span<const uint8_t> fresh);   // states unchanged
  void CopyStates(std::span<const Handle>, std::span<const uint8_t> fresh, std::span<uint8_t> out);
};

class Learner {                                                  // parameters + Adam on devices[0]
 public:
  Learner(Runtime&, const Model&, std::vector<int> devices);
  const Model& model() const;
  UpdateStats Update(const Minibatch&, const Hyper&);            // split across GPUs, averaged, one Adam step
  Weights weights() const;
  void Save(const std::string& path) const;                      // temporary file, then rename
  void Load(const std::string& path);                            // checked against the Model
};
```

```cpp
struct Minibatch {                          // host memory, time-major [turns, chunks]
  int32_t chunks, turns;                    // Model::chunks_per_update(), Model::chunk_turns()
  std::span<const uint8_t> features, critic, valid;
  std::span<const int32_t> action;
  std::span<const float> log_prob, advantage, target;   // log_prob as sampled; advantage normalised
  std::span<const uint8_t> start_state;     // [chunks, state_bytes], opaque
};
struct Hyper { float learning_rate, clip, entropy_weight, value_weight, max_grad_norm; };
struct UpdateStats { float policy_loss, value_loss, entropy, approx_kl, clip_fraction,
                     actor_grad_norm, critic_grad_norm; bool skipped; };
```

Recurrent state is opaque bytes to everyone outside `policy`.

The action mask travels inside the feature bytes. `policy`'s C++ never reads it; only the exported program does, and it knows the layout from the export. That's why the input description check is enough to keep them in step.

Private to this layer:

- the PJRT C API;
- the named programs (`act`, `value`, `gather_states`, `grad_step`, `apply`) and their signatures, checked by name at load;
- the state pools;
- the parameter replicas;
- loss scaling and skipping non-finite steps.

### model (JAX, `train/model/`, offline)

- `inputs.py` casts the feature and critic records. It never computes a channel.
- `model.py` holds `init`, `actor_step(params, inputs, h) → (log_probs, h)` and `critic_value(params, critic, h)`.
- `programs.py` builds exactly the functions that are exported, and tests call those same functions.
- `export.py` writes the programs with named signatures, the initial parameters, and a manifest: the configuration, the code revision, and the input description.

### rl (`rl/rl.hpp`)

```cpp
class Batch {                                    // one collector's rollout: move-only, contents private to rl
 public:
  Batch(Batch&&) noexcept; Batch& operator=(Batch&&) noexcept; ~Batch();
  int64_t turns() const;
 private:
  struct Data;
  std::unique_ptr<Data> data_;
};

class Rollout {                                  // one collector: its env, agents, actor
 public:
  Rollout(dragons::VecEnv&, agent::Agents&, policy::Actor&);   // checks the three agree on rows and capacity
  Batch Collect(int32_t steps, RolloutStats&);   // games carry on across calls
};

PpoStats TrainPpo(policy::Learner&, std::span<const Batch> batches, const PpoConfig&);
```

`RolloutStats`, `PpoConfig` and `PpoStats` are plain structs of named fields:

- `RolloutStats`: games, wins by team, draws, eliminations, deaths by cause, splits, sprints, untrained turns, peak handles, timings;
- `PpoConfig`: γ, λ, epochs, and the `policy::Hyper` values;
- `PpoStats`: mean losses, gradient norms, skipped steps, explained variance, replay KL, timings.

Inside, a `Batch` is flat arrays:

- **per turn:** action, log_prob, value, round, and the same dragon's next turn;
- **per trajectory end:** `enum class End { Outcome, Bootstrap, Untrained }` with its value and round;
- **turn records:** the feature and critic bytes;
- **chunks:** runs of rows, each with an opaque start state.

The trainer only moves batches from `Collect` to `TrainPpo`; it never sees inside one.

Private to this layer:

- the handle registry;
- trajectory and chunk building;
- the bootstrap at the end of a rollout, which perceives the current observations once. It keeps those features for the next `Collect`'s first step, so no turn is perceived twice;
- the returns: team-outcome GAE, per round and per turn, as a pure function;
- advantage normalisation;
- minibatch packing and the epoch loop.

### trainer and bot

**trainer** (`main.cpp`) reads the configuration and builds, in order: one `Runtime`, the `Model`, one learner, and one collector per GPU (a `VecEnv`, `Agents`, an `Actor` and a `Rollout` each). At start-up it checks three things, and refuses to run if any fails:

- the input spec matches (`agent` against `policy`);
- envs per collector equal `Model::rows()`;
- handle capacity is at least envs × 2 × the largest unit limit.

It then runs the loop. Every N iterations it writes a checkpoint (the learner's `Save` plus a small progress file of iteration and turns). Each iteration it records one metrics type, rendered both as a log line and as a JSON line, with non-finite values written as null.

**bot** (`main.cpp`) parses the judge's text, calls `Agents::Perceive` and then a small hand-written forward pass of the actor, and calls `Decide`. It's designed later, against the learner's saved format.

---

## 4. Data flow

**One step of one collector:**

```
obs     = env.observations()                                   game-owned, no copy
handles = registry.For(env headers)                            new dragon: agents.Begin, fresh = 1
agents.Perceive(handles, obs, actor.features())                encoded straight into the pinned input
agents.EncodeCritic(env, actor.critic())
actor.CopyStates(chunk-start handles) → batch.start_states     only for chunks starting now
acted   = actor.Act(handles, fresh)                            one upload, one download
batch  ← feature and critic rows, acted                        the one deliberate copy
agents.Decide(handles, acted.action, actions)
result  = env.Step(actions)                                    deaths / ends: free handles, agents.End
```

The order matters in three places:

- **Chunk-start states are copied before `Act`.** `Act` advances the state, and a chunk must start from the state before its first turn.
- **Handles freed by `Step` are free for the next step.** A finished game restarts inside `Step`, so its new dragons get handles on the next call.
- **Handles are freed before being reused.** A new dragon that reuses a freed handle is passed with `fresh = 1`, so it never inherits a dead dragon's state.

**One iteration:**

```
all collectors (one per GPU, in parallel threads):  batch_k = rollout_k.Collect(steps)
learner (all GPUs):                                  TrainPpo(learner, {batch_0..batch_3})
all actors:                                          actor_k.SetWeights(learner.weights())
every N iterations:                                  learner.Save, progress file, metrics
```

Collection and learning alternate on the same GPUs. The recurrent state is 2 KB per dragon, so an actor's pool and the learner's working memory fit on one card together. Overlapping the two phases is a later optimisation inside the trainer loop; no interface changes for it.

**Resume:** a checkpoint holds the parameters, the Adam state, and the iteration and turn counts. Games in progress aren't saved; after a restart every env starts a new game, which costs one rollout of short games.

A checkpoint is written in a fixed order:

1. `<it>.state`, via `Learner::Save`: a temporary file, then a rename;
2. `<it>.progress`, the same way, and only once the state file exists.

A progress file therefore marks a complete checkpoint. On start-up the trainer takes the newest progress file whose state file loads, and falls back to the one before it if not.

---

## 5. A dragon's memory (inside `agent`)

It's kept by code from the dragon's own observations and the commands it sent. It's deterministic, sized to the board, and a split child starts with it empty.

Because it lives in host RAM and is sized to the actual board, there's no board-size cap any more: training and the bot handle anything the engine accepts, up to 256×256. The learned map's 64×64 limit goes away.

**Per tile:**

- when last seen;
- north and west edges (empty, kelp, portal id), final once seen;
- whether the tile is a pearl bed, and its countdown as last seen (predicted forward);
- whether a pearl was there;
- the occupant last seen: ally or enemy, body or head;
- when this dragon's own head was last there.

**Per dragon:**

- its own body, rebuilt from its path, its `LENGTH` and its splits;
- portal pairs, once both ends have been seen;
- for each enemy seen, a lower bound on its length and where and when it was last seen.

**What the model sees of it (not yet decided):**

- **(a) Proposed:** a 13×13 full-resolution view around the head, a 13×13 view at 4 tiles per cell further out, and distance fields over the whole known board (to pearls, beds about to spawn, unexplored space, enemies), sampled on the near view.
  - **What blocks the distance fields:** known kelp, plus bodies seen in the last few rounds (older sightings are too stale to block). The dragon's own segment k counts only for paths shorter than the length − k steps it takes that segment to move away. Portals are followed only when both ends are known.
- **(b) Alternative:** the whole board at full resolution. It's about 80 KB per turn and much more compute.

Only the feature layout changes between the two; no interface changes.

---

## 6. Throughput budget

| stage | per turn |
|---|---|
| env step (existing) | ~0.1–0.2 µs of CPU per turn |
| `Perceive`: update memory, encode the views | ~5–10 µs of CPU |
| `Perceive`: distance fields | one breadth-first search from each target type (pearls, beds due, unexplored, enemies) over the known board: ~4 × W·H tile visits, ~50–100 µs on 64×64 and much less on small boards. The largest new cost |
| `Act` | one batched GPU call per step per collector |
| rollout storage | about 10 KB per turn: features ≈ 8 KB (window, scalars, mask, near and far views, distance fields) plus critic ≈ 2 KB. That's ≈ 21 GB per 2.1M-turn rollout; 128 steps halves it |
| update | no map in the recurrence; the start states are the 2 KB GRU |

**Total for a 2.1M-turn rollout:** at ~80 µs per turn, 2.1M turns × 80 µs ≈ 170 CPU-seconds, or about 6 s of wall time on 28 threads. That's well under today's 23 s rollout, and small against the update. In the judge, where the bot runs the same code for one dragon, it's roughly 16k tile visits: about 1M points of the 30M per-turn budget. If it ever dominates, the distance fields can be recomputed only when the known board changes (a new tile seen, a pearl appeared or was eaten) instead of every turn.

The pieces removed are the map's backward pass (about 30% of the update), the 38 KB `gather` per chunk, and pool overflow.

---

## 7. Testing, per layer

- **game:** lockstep against the official engine (existing), plus the new `Action` and `StepResult` API.
- **agent:** memory against the true game state (edges exact, countdowns exact while in view, own body exact except across unseen portals). The features must be identical whether the observation came from a record or from the judge's text. Masked actions must be fatal (existing test).
- **policy:** replay parity (the same log-probabilities from `Act` and from replaying a chunk), grad-then-apply against a single-GPU reference, and save → load round trips.
- **rl:** returns on hand-built batches (outcome, bootstrap, untrained, and a dragon that dies mid-game); handle recycling; chunk boundaries.
- **trainer:** a smoke run of a few iterations on small artifacts, in CI or on the GPU box.
- **bot** (when built): its hand-written forward pass gives the same log-probabilities as `Actor::Act` on recorded games, and a turn stays within the judge's per-turn budget, measured in the sandbox.

---

## 8. Where today's code goes

| today (`train/`) | becomes |
|---|---|
| `src/pjrt.*`, `src/artifacts.*` | `policy/src/`: runtime and programs, now looked up by name |
| `src/rollout.*` | `rl/src/`: registry, trajectories, bootstrap; returns split out as a pure function |
| `src/update.*` | `policy/src/` (the multi-GPU gradient and apply) plus `rl/src/` (epochs, minibatch packing) |
| `src/actions.hpp`, `dragons_model/features.py` (mask, window channels) | `agent/src/`: the only copy |
| `dragons_model/` | `model/`: the learned map is removed; inputs are cast only |
| `src/main.cpp` | `trainer/` |
| dragons-env `WriteState`, block counters | `agent/src/` (critic encoding) |

The replay-parity check, the fp16 handling, the systemd deployment and the wandb sidecar (which reads the trainer's JSON lines) carry over unchanged.

## 9. Deliberately not in this design

- plugin systems, factories, per-dragon objects, virtual interfaces for one implementation;
- a map-source class hierarchy (a function is enough);
- evaluation and opponent pools: they come later as another `Actor` (other weights) or a scripted `Decide`, using the same `Rollout`;
- the bot's forward pass: later, against `Learner::Save`'s format;
- sonar: waiting on the 1.0.0 source.

## 10. Open decisions

Each has a default. The defaults change only `agent`'s feature layout and the trainer's configuration, not the structure, so either can be revised later.

1. The memory views, (a) or (b) (section 5).
2. Whether the critic's encoding stays in `agent` (proposed: it's part of the model's input contract) or becomes its own small training-only component beside it.
3. Rollout RAM against steps per rollout: 256 steps × 8,192 envs × ~10 KB ≈ 21 GB (of 62 GB). Proposed: 128 steps (≈ 10 GB), with twice as many iterations.
