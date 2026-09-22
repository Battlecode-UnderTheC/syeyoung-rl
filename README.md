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

Phase 2 of the spec's curriculum: one dragon per side, no splitting, no
communication. Known gaps, so they are not mistaken for finished work:

* the native trainer uses a value head on the actor trunk, which is spec
  section 7's IPPO *ablation*, not section 4's team critic. The team critic is
  implemented and exported but not yet wired in;
* team GAE is therefore still per-dragon. Identical at one dragon per side,
  wrong as soon as splitting is enabled;
* no deployable bot exists yet. `underthec/` is still the starter bot, and the
  policy runs under torch, which the judge sandbox does not have. A numpy
  forward pass within the 100M point per turn budget is still to be written.
