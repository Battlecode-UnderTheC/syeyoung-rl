# syeyoung-rl

Reinforcement learning for the UNSW Battlecode dragons game: the
spatial-memory recurrent actor of `SPATIAL_MEMORY_RL_IMPLEMENTATION_SPEC.md`.

The game itself is not here. It lives in the `dragons-env` submodule: the rules,
a driver for the official wasm engine, and the differential tests that pin one
to the other. This repository is everything an agent needs on top of that.

```sh
git clone --recurse-submodules <this repo>
uv venv .venv && uv pip install --python .venv/bin/python numpy torch unswbc pybind11 pytest
PYTHON=.venv/bin/python ./tools/build_native.sh     # native sim + exact protocol text
PYTHONPATH=.:cpp:dragons-env .venv/bin/python -m pytest tests -q
```

## Layout

```
snake_rl/env/        schema registry, private factual memory, prepare(), command codec
snake_rl/model/      spatial-memory actor, team-event critic, per-lifetime runtime, export
snake_rl/algo/       event GAE, recurrent joint-command PPO
snake_rl/rollout/    event-timeline collection on the native sim
snake_rl/training/   config, python self-play driver with a historical league
snake_rl/policies/   scripted evaluation baselines (never teachers)
cpp/agent/           the native sim's exact spawn/turn blocks and full-state dumps
tools/               build script, TorchScript export
```

`COMPETITIVE_RL_DESIGN.md` is the strategic analysis behind the spec;
`GAME_ANALYSIS.md` records the engine's rules as read out of the binary.

## What exists

* **Environment** -- a versioned schema (`env/schema.py`); a per-lifetime
  factual memory with structured per-tile storage, canonical north/west edges,
  portal pairing by identity, the own-body deque rebuilt from the dragon's own
  moves and a 32-command ring, snapshot/restorable and idempotent under
  redelivery (`env/memory.py`); `prepare()` materialising the 42-channel map,
  the north-up crop, status, history, entity and portal tables
  (`env/observation.py`); the conditional MOVE/SPLIT codec with the private
  prefix simulator (`env/actions.py`).
* **Model** -- sparse learned writes into a 16-channel map, a multiscale
  ConvGRU reader with portal exchange, local and four global reads, the
  tactical GRU, the autoregressive move decoder, the split scorer, the optional
  sonar head and auxiliary heads, behind `encode_and_update` /
  `decode_sample` / `decode_evaluate`; TorchScript encoder and decoder entry
  points; a team-event critic on exact native state.
* **Training** -- event-indexed GAE, PPO with one ratio per command and burn-in
  replayed from snapshots, and a Python driver. The Python driver is a
  reference, not the training path: it reaches tens of decisions per second.
* **Native** -- `cpp/agent` prints the engine's exact spawn and turn blocks
  (`game.turn_block`) and dumps tiles and edges; `tests/test_spatial_sim.py`
  pins the text to the wasm engine byte for byte.

## Not done

* The native trainer: per-lifetime factual state and codec in C++, batched
  collection with GPU-resident maps, and the event critic and PPO loop in
  libtorch. Without it nothing trains at scale.
* The numpy deployment runtime and the judge-budget gate (spec 15).
* Any training run, ablation or promotion gate (spec 17-18).
