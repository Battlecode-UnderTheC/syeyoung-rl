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
snake_rl/env/        schema registry, private factual memory, prepare(), command codec (reference)
snake_rl/model/      spatial-memory actor, team-event critic, TorchScript export
cpp/agent/spatial/   the same factual memory and codec in C++, driven from the sim
cpp/agent/           the sim's exact spawn/turn blocks and full-state dumps, pybind
cpp/train/           GPU-resident state, scripted-model wrapper, arena groups, the trainer
tools/               build scripts, export, map generation
```

Training runs natively:

```sh
PYTHONPATH=.:cpp:dragons-env .venv/bin/python tools/export_spatial.py --out native
PYTHONPATH=.:cpp:dragons-env .venv/bin/python tools/gen_maps.py --out native/maps.txt --sizes 15 --per-size 64
PYTHON=.venv/bin/python ./tools/build_trainer.sh      # also runs the GPU-state self-test
./cpp/train/spatial_train --model native --maps native/maps.txt --threads 4 --arenas-per-thread 4096
```

`COMPETITIVE_RL_DESIGN.md` is the strategic analysis behind the spec;
`GAME_ANALYSIS.md` records the engine's rules as read out of the binary.

## What exists

* **Environment (Python reference)** -- a versioned schema (`env/schema.py`);
  a per-lifetime factual memory with structured per-tile storage, canonical
  north/west edges, portal pairing by identity, the own-body deque rebuilt from
  the dragon's own moves and a 32-command ring (`env/memory.py`); `prepare()`
  materialising the 42-channel map, crop, status, history, entity and portal
  tables (`env/observation.py`); the conditional MOVE/SPLIT codec with the
  private prefix simulator (`env/actions.py`).
* **Model** -- sparse learned writes into a 16-channel map, a multiscale ConvGRU
  reader with portal exchange, local and four global reads, the tactical GRU,
  the autoregressive move decoder, the split scorer, the optional sonar head
  and auxiliary heads; the team-event critic; TorchScript encoder, decoder
  step methods and critic (`model/export.py`).
* **Native** -- `cpp/agent/spatial` mirrors the Python memory and codec exactly
  (`tests/test_spatial_native.py` compares every tensor on real matches);
  `cpp/train` keeps each lifetime's factual store on the GPU as a compact raw
  tensor updated by the cells a turn changed, derives the 42 channels on
  device (`selftest` pins it to the CPU materialisation), steps thousands of
  arenas per thread in slot batches, samples full commands through the
  decoder step, records exact critic states, and runs event GAE and the
  joint-command PPO update by replaying chunks from snapshots on the GPU. A
  replay-parity metric is printed every iteration.

## Not done

* Frozen historical opponents in the native trainer (self-play only so far),
  the auxiliary losses, and the numpy/C++ deployment runtime with the
  judge-budget gate (spec 15).
* Any training run long enough to evaluate, ablations or promotion gates
  (spec 17-18).
