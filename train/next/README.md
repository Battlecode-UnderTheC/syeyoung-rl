# train/next — interfaces for the redesign (no implementation)

Declarations and contracts only, for review before anything is built. Each
header says what a component owns, what it may not do, and who calls it.

## Principles

- **Names, not positions.** Program arguments and results are named, and
  checked against the exported signature when the program loads.
- **Return values, not out-parameters.** Nothing mutates its inputs; results
  are returned by value, and read-only views are handed out.
- **Types, not sentinels.** `std::optional`, `enum class` and small structs
  replace −1/−2 codes and magic indices.
- **One source of truth.** The agent-side code in `agent/` (observation
  parsing, the action table and mask, the fact map, feature encoding) is plain
  C++ shared by the trainer and the bot. JAX only reads the encoded features.
- **The env is the game.** dragons-env exposes rules and observations. Critic
  features and memory are built on the agent side.

## Layout

```
include/agent/        shared with the bot (no PJRT, no threads)
  observation.hpp     Observation: from a VecEnv record or from the judge's text
  actions.hpp         Action index <-> engine Command; the action mask
  fact_map.hpp        FactMap: a dragon's program-maintained memory
  features.hpp        FeatureLayout; EncodeFeatures (window, scalars, mask, memory views)

include/pjrt/         runtime layer
  pjrt.hpp            Device, DType, Shape, Buffer, Runtime, Program (named signatures)

include/train/        the trainer
  artifacts.hpp       Artifacts: exported programs + ModelConfig + parameter spec
  params.hpp          Params, AdamState, Checkpoint (save/load)
  programs.hpp        typed wrappers: Act, Value, GatherStates, GradStep, Apply
  maps.hpp            MapSource: contest files, generated boards
  critic.hpp          CriticFeatures from full game state (training only)
  slots.hpp           SlotPool: GRU slots on a device
  agents.hpp          AgentTable: per-dragon bookkeeping across games
  batch.hpp           Batch, Chunk, TurnEnd: one rollout's data, read-only
  returns.hpp         ComputeReturns: team-outcome GAE, pure
  collector.hpp       Collector: one device's envs -> Batch
  learner.hpp         Learner: data-parallel PPO over devices
  evaluation.hpp      Evaluator, Opponent: strength against fixed opponents
  metrics.hpp         IterationMetrics and its two renderings
  trainer.hpp         TrainerConfig, DevicePlan, Trainer

dragons_model/        JAX: the model and the exported programs
  inputs.py           decode the feature record (casts only)
  model.py            actor_step, critic_value, parameter init
  programs.py         Programs: exactly the functions that are exported
  export.py           named signatures, manifest with code hash

dragons-env.md        the dragons-env API changes this assumes
```

## Assumed but not yet decided

- The fact map's views (`agent/features.hpp`: `MemoryViewSpec`) follow option
  (a): full-resolution detail near the head, a coarse far view, and distance
  fields computed over the whole known board. Option (b), the whole raw
  board, would replace `MemoryViewSpec` and change the model's inputs; nothing
  else here would change.
