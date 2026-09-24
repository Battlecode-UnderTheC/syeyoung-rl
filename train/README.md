# train — dragons competitive policy v1

An implementation of `../INDEPENDENT_MODEL_AND_TRAINING_DESIGN.md` (model and
training specification v1). Section numbers below refer to it.

| layer | where | knows |
|---|---|---|
| game | `../dragons-env` (C++) | the rules, incl. the 1.0.0 sonar (§2), held to the unswbc 1.0.0 wasm |
| facts | `native/observation.*`, `native/facts.*` | one lifetime's legal observations and its own commands (§3), encoded as exact integers |
| critic view | `native/critic.*` | the full state at an event, training only (§8 inputs) |
| collection / replay | `native/collector.*` | complete matches, per-event records, replay from birth (§9.1, §9.4) |
| features | `dragons_rl/features.py` | every §4 / §8 float formula, from the integer records |
| model | `dragons_rl/model.py` | actor (530,115), critic (83,457), init, action distributions, sampling (§5-8, §10.2) |
| learning | `dragons_rl/ppo.py` | returns, PPO losses, chunked replay, KL rule, Adam (§9.2-9.4) |
| run | `dragons_rl/{train,league,manifest,evaluate,fixtures}.py` | the reference loop, league, manifest, validation (§10-11) |

Build and test (macOS or Linux):

    uv venv -p 3.12 .venv && uv pip install --python .venv/bin/python numpy "jax[cpu]" pytest scikit-build-core pybind11 unswbc==1.0.0
    uv pip install --python .venv/bin/python --no-build-isolation -e ../dragons-env -e .
    .venv/bin/python -m pytest tests                      # §12 acceptance tests
    .venv/bin/python -m dragons_rl.fixtures fixtures.json # §2 conformance against the 1.0.0 wasm

A reference run (one training seed; all visible GPUs):

    .venv/bin/python -m dragons_rl.train --run runs/seed0 --seed 0 --fixtures fixtures.json

On poscatgpu2 it runs as `systemctl --user start dragons-rl@0` (`deploy/`),
resumes from its last checkpoint on restart, and `deploy/status.sh 0` summarises it.

Choices the spec leaves to the implementation:

* Maps: the 11 contest maps (train) plus seeded generated families
  (`manifest.FAMILIES`); validation and test use held-out families.
* Opponents for validation/test: two scripted adapters (`opponents/`,
  `greedy-v1`, `random-v1`) and the run's own S0. No external bots exist, so
  results are internal, not evidence of beating other bots.
* Batching: padding to board-size buckets (masked, so outputs are unchanged);
  several 4-chunk microbatches per device call; gradients summed over devices.
