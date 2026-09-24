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

model  (JAX, offline) ──exports──► artifacts ──loaded by──► policy
```

| layer | knows | must not know |
|---|---|---|
| **game** | rules, the judge's observation | agents, models, training |
| **agent** | one dragon's memory, the model's input encoding, the action mask, action → engine action | GPUs, the model's internals, training |
| **policy** | running the exported model on GPUs, per-dragon recurrent state, parameters | what features mean, games, trajectories |
| **rl** | trajectories, returns, PPO, dragon lifetimes | devices, memory, feature layout, JAX |
| **trainer** | configuration, devices, the loop, checkpoints, logs | anything the layers already own |

**Rules:**

1. Dependencies never point upward and never form a cycle. A layer may use any layer below it. Each layer is its own CMake library, so an upward include fails to link.
2. Each layer has exactly one public header. Everything else lives in that layer's `src/`.
3. Layers exchange plain data: bytes, integers, and spans into buffers someone else owns. Every call covers a whole batch, and data is written where the next layer reads it.
4. Knowledge stays in its owning layer. Only `agent` knows memory exists. Only `policy` knows about JAX, PJRT, program argument names or devices. Only `rl` knows about trajectories.

`agent` and `policy` never include each other. Their one shared contract is the byte layout of the model's input, which the trainer checks at start-up: `agent::Agents::input_spec().description == policy::Model::input_description()`.

---

## 2. Three ideas the design rests on

**Handles.** Every live dragon holds a small integer, its handle, from its first turn until it dies or its game ends. `rl` hands handles out and takes them back; nothing else tracks lifetimes. Both per-dragon stores are indexed by handle:

- `agent`: the dragon's memory, in host RAM;
- `policy::Actor`: the dragon's recurrent state, on the GPU. This is just the GRU, 2 KB per dragon, so a million handles fit.

A collector's `Agents`, `Actor` and `Rollout` are built with the same capacity, and the trainer passes it to all three. If none is free, the dragon plays from zero state and its turn is recorded as `Untrained`. With capacity sized from the unit limit, that stays a safety valve rather than the norm.

**One collector per GPU.** Each collecting GPU gets its own share of the envs (a `VecEnv`), its own `Agents`, its own `Actor`, and its own range of handles. No object ever needs to map a dragon to a device. Collectors run in parallel threads.

**Actor and learner are separate objects.** An actor does inference for its dragons on one GPU. The learner trains data-parallel across GPUs. Parameters flow one way, from the learner to the actors, as a `Weights` snapshot.

---

## 3. Public interfaces

### game (dragons-env, `dragons/vec_env.hpp`)

This is the existing API, with four changes:

```cpp
struct Action {                                   // replaces Act(kind, steps, nsteps, split, sonar)
  struct Move { std::vector<uint8_t> steps; };    // directions 0..3, at least one
  struct Split { int32_t child_segments; };
  struct None {};                                 // the engine's "no valid action"
  std::variant<Move, Split, None> what;
  std::optional<uint32_t> sonar;
};

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
struct Rows { uint8_t* data; int64_t rows; int64_t stride; };   // someone else's buffer

class Agents {
 public:
  explicit Agents(int32_t capacity);
  static InputSpec input_spec();                  // feature bytes, critic bytes, description

  void Begin(Handle, int32_t board_width, int32_t board_height);   // empty memory
  void End(Handle);

  // many dragons at once, threaded inside; row i is handles[i]
  void Perceive(std::span<const Handle>, std::span<const uint8_t> observations, int64_t stride, Rows features);
  void EncodeCritic(const dragons::VecEnv&, Rows critic) const;                 // training only
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

class Actor {                                                    // one GPU, `capacity` handles of state
 public:
  Actor(Runtime&, const Model&, int device, int32_t capacity);
  void SetWeights(const Weights&);
  std::span<uint8_t> features();                                 // input buffers to fill (pinned, reused)
  std::span<uint8_t> critic();
  Acted Act(std::span<const Handle>, std::span<const uint8_t> fresh);      // action, log_prob, value
  std::span<const float> Value(std::span<const Handle>, std::span<const uint8_t> fresh);   // states unchanged
  void CopyStates(std::span<const Handle>, std::span<const uint8_t> fresh, std::span<uint8_t> out);
};

class Learner {                                                  // parameters + Adam on devices[0]
 public:
  Learner(Runtime&, const Model&, std::vector<int> devices);
  UpdateStats Update(const Minibatch&, const Hyper&);            // split across GPUs, averaged, one Adam step
  Weights weights() const;
  void Save(const std::string& path) const;                      // temporary file, then rename
  void Load(const std::string& path);                            // checked against the Model
};
```

`Minibatch` is host spans, time-major: features, critic, valid, action, log_prob, normalised advantage, target, and one opaque start state per chunk. Recurrent state is opaque bytes to everyone outside `policy`.

Private to this layer:

- the PJRT C API;
- the named programs (`act`, `value`, `gather_states`, `grad_step`, `apply`) and their signatures, checked by name at load;
- the state pools;
- the parameter replicas;
- loss scaling and skipping non-finite steps.

### model (JAX, `dragons_model/`, offline)

- `inputs.py` casts the feature and critic records. It never computes a channel.
- `model.py` holds `init`, `actor_step(params, inputs, h) → (log_probs, h)` and `critic_value(params, critic, h)`.
- `programs.py` builds exactly the functions that are exported, and tests call those same functions.
- `export.py` writes the programs with named signatures, the initial parameters, and a manifest: the configuration, the code revision, and the input description.

### rl (`rl/rl.hpp`)

```cpp
struct Batch {                                   // one collector's rollout; plain data, read-only once built
  enum class End : uint8_t { Continues, Outcome, Bootstrap, Untrained };
  struct Turn { int32_t action; float log_prob, value; int32_t round; int64_t next;
                End end; float end_value; int32_t end_round; };
  struct Chunk { int64_t offset; int32_t length; };   // into chunk_turns
  std::vector<Turn> turns;
  std::vector<uint8_t> features, critic;              // [turns, bytes]
  std::vector<int64_t> chunk_turns;
  std::vector<Chunk> chunks;
  std::vector<uint8_t> start_states;                  // [chunks, state_bytes]
};

class Rollout {                                  // one collector: its env, agents, actor
 public:
  Rollout(dragons::VecEnv&, agent::Agents&, policy::Actor&, int32_t capacity);
  Batch Collect(int32_t steps, RolloutStats&);   // games carry on across calls
};

PpoStats TrainPpo(policy::Learner&, const policy::Model&, const std::vector<const Batch*>&, const PpoConfig&);
```

Private to this layer:

- the handle registry;
- trajectory and chunk building;
- the returns: team-outcome GAE, per round and per turn, as a pure function;
- advantage normalisation;
- minibatch packing and the epoch loop.

### trainer and bot

- **trainer** (`main.cpp`) reads the configuration and builds, in order: one `Runtime`, the `Model`, one learner, and one collector per GPU (a `VecEnv`, `Agents`, an `Actor` and a `Rollout` each). It then checks the input spec and runs the loop, writing checkpoints (the learner's `Save` plus a small progress file of iteration and turns) and one metrics type rendered as a log line and a JSON line, with non-finite values written as null.
- **bot** (`main.cpp`) parses the judge's text, calls `Agents::Perceive` and then a small hand-written forward pass of the actor, and calls `Decide`. It's designed later, against the learner's saved format.

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

**One iteration:**

```
all collectors (one per GPU, in parallel threads):  batch_k = rollout_k.Collect(steps)
learner (all GPUs):                                  TrainPpo(learner, model, {batch_0..batch_3})
all actors:                                          actor_k.SetWeights(learner.weights())
every N iterations:                                  learner.Save, progress file, metrics
```

Collection and learning alternate on the same GPUs. The recurrent state is 2 KB per dragon, so an actor's pool and the learner's working memory fit on one card together. Overlapping the two phases is a later optimisation inside the trainer loop; no interface changes for it.

---

## 5. A dragon's memory (inside `agent`)

It's kept by code from the dragon's own observations and the commands it sent. It's deterministic, sized to the board, and a split child starts with it empty.

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
- **(b) Alternative:** the whole board at full resolution. It's about 80 KB per turn and much more compute.

Only the feature layout changes between the two; no interface changes.

---

## 6. Throughput budget

| stage | per turn |
|---|---|
| env step (existing) | ~0.1–0.2 µs of CPU per turn |
| `Perceive`: update memory, encode features and distance fields | ~20–40 µs of CPU, threaded, the largest new cost |
| `Act` | one batched GPU call per step per collector |
| rollout storage | feature and critic rows, about 7 KB per turn (≈15 GB per 2.1M-turn rollout; cut steps if RAM gets tight) |
| update | no map in the recurrence; the start states are the 2 KB GRU |

The pieces removed are the map's backward pass (about 30% of the update), the 38 KB `gather` per chunk, and pool overflow.

---

## 7. Testing, per layer

- **game:** lockstep against the official engine (existing), plus the new `Action` and `StepResult` API.
- **agent:** memory against the true game state (edges exact, countdowns exact while in view, own body exact except across unseen portals). The features must be identical whether the observation came from a record or from the judge's text. Masked actions must be fatal (existing test).
- **policy:** replay parity (the same log-probabilities from `Act` and from replaying a chunk), grad-then-apply against a single-GPU reference, and save → load round trips.
- **rl:** returns on hand-built batches (outcome, bootstrap, untrained, and a dragon that dies mid-game); handle recycling; chunk boundaries.
- **trainer:** a smoke run of a few iterations on small artifacts, in CI or on the GPU box.

---

## 8. Deliberately not in this design

- plugin systems, factories, per-dragon objects, virtual interfaces for one implementation;
- a map-source class hierarchy (a function is enough);
- evaluation and opponent pools: they come later as another `Actor` (other weights) or a scripted `Decide`, using the same `Rollout`;
- the bot's forward pass: later, against `Learner::Save`'s format;
- sonar: waiting on the 1.0.0 source.

## 9. Open decisions

1. The memory views, (a) or (b) (section 5).
2. Whether the critic's encoding stays in `agent` (proposed: it's part of the model's input contract) or becomes its own small training-only component beside it.
3. Rollout RAM against steps per rollout: 256 steps × 8,192 envs × ~7 KB ≈ 15 GB.
