# syeyoung-rl

Reinforcement learning for the UNSW Battlecode dragons game.

The game itself is not here. It lives in the `dragons-env` submodule: the rules,
a driver for the official wasm engine, and the differential tests that pin one
to the other. This repository is everything an agent needs on top of that —
observations, policy, training, evaluation.

```sh
git clone --recurse-submodules <this repo>
uv venv .venv && uv pip install --python .venv/bin/python numpy torch unswbc pybind11 pytest
PYTHON=.venv/bin/python ./tools/build_native.sh     # observation encoder + batched env
PYTHON=.venv/bin/python ./tools/build_trainer.sh    # native PPO trainer (needs libtorch)
PYTHONPATH=.:cpp:dragons-env .venv/bin/python -m pytest tests -q
```

## What is here

```
snake_rl/env/        observation encoding, action space and masking, match runner
snake_rl/model/      actor, team critic, TorchScript export for the native trainer
snake_rl/algo/       team GAE, recurrent PPO
snake_rl/training/   config, map sampling, python trainer (superseded by the C++ one)
cpp/agent/           the same observation encoding in C++, plus batched stepping
cpp/train/           native rollout and PPO trainer, built on libtorch
tools/               build scripts, model export, replay playback, run supervision
```

The observation encoding exists twice, in Python and in C++, because training
runs natively for speed while the judge sandbox offers only Python and numpy.
`tests/test_cpp_obs.py` compares the two tensor for tensor; without it a policy
trained through one would silently drift when deployed through the other.

## Reading order

`SNAKE_MARL_IMPLEMENTATION_SPEC.md` is the design the work follows.
`GAME_ANALYSIS.md` records the engine's actual rules, read out of the
decompiled binary and confirmed against it — including the ones that decide
everything: the winner is the longest single living dragon, head-to-head kills
both regardless of length, and a dying dragon drops pearls on every second
segment.

## State

Two actors live side by side.

**`dragons-spatial-v1`** (`SPATIAL_MEMORY_RL_IMPLEMENTATION_SPEC.md`) is the
spatial-memory recurrent actor with the conditional MOVE/SPLIT command space,
implemented through the spec's phases 1, 2 and 4 in Python:

* `snake_rl/env/schema.py`, `memory.py`, `prepare.py` -- the versioned schema,
  the private per-lifetime factual map with canonical edges and portal
  identities, the own-body deque and command ring, and `prepare()`;
* `snake_rl/env/actions.py` (`CommandCodec`) -- MOVE token sequences up to K,
  SPLIT n over every legal size, the private prefix simulator and its masks;
* `snake_rl/model/spatial_memory.py`, `encoder.py`, `actor.py`
  (`SpatialActor`) -- sparse learned writes, the multiscale ConvGRU reader with
  portal exchange, local/global reads, the tactical GRU, the autoregressive
  move decoder, the split scorer, the optional sonar head and the auxiliary
  heads, behind `encode_and_update` / `decode_sample` / `decode_evaluate`;
* `snake_rl/model/critic.py` (`EventCritic`), `algo/gae.py` (`event_gae`),
  `algo/ppo.py` (`ppo_update_spatial`) -- the team-event critic on exact
  native state, event-indexed GAE with per-round discount, and PPO with one
  ratio per command, burn-in replay from snapshots, a fixed team-round
  denominator, unique team-event value loss and KL early stopping;
* `snake_rl/rollout/collector.py` (`play_event_match`) -- the global event
  timeline on the native sim, with lossless per-turn records and chunk
  snapshots instead of dense per-turn maps;
* `snake_rl/training/spatial_train.py` -- self-play with a historical league,
  forked workers and versioned checkpoints;
* `cpp/agent` -- the native sim now prints the engine's exact spawn and turn
  blocks (`game.turn_block`) and dumps full tiles/edges for the critic;
  `tests/test_spatial_sim.py` pins the text to the wasm engine byte for byte.

`tests/test_spatial_*.py` cover the section-16 acceptance list that can be
checked without a trained model: privacy, memory semantics, body order through
growth, sprints, splits and portals, codec masks against the engine's own
deaths, sampling/evaluation parity, step-vs-replay agreement including cold
children, the G - V identity, and PPO mechanics.

Not done, so it is not mistaken for finished work:

* no numpy deployment runtime and no judge-budget measurement (spec 15, gate B);
  `export_spatial` writes TorchScript encoder/decoder entry points and metadata,
  but nothing has been run under the sandbox limits;
* the C++ trainer (`cpp/train`) still runs the old local actor; the native
  factual encoder and the C++ side of the schema registry are not written;
* collection is Python-driven (~10 ms per decision), and no long training run,
  ablation or promotion gate (spec 17-18) has been attempted.

**The local CNN+GRU baseline** (`Actor`, `SNAKE_MARL_IMPLEMENTATION_SPEC.md`)
remains as the controlled comparison, unchanged, with its earlier gaps:

* the native trainer uses a value head on the actor trunk, which is spec
  section 7's IPPO *ablation*, not section 4's team critic;
* team GAE is still per-dragon there, wrong as soon as splitting is enabled;
* no deployable bot exists yet. `underthec/` is still the starter bot.
