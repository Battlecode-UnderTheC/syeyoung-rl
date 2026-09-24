"""The programs the C++ trainer runs: exactly these functions are exported,
and exactly these are tested. Argument names here are the names in the
exported signature (train/programs.hpp looks them up by name).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax

from .inputs import FeatureLayout
from .model import ModelSpec, Params


@dataclass(frozen=True)
class ProgramConfig:
    rows: int = 2048            # dragons per act call (envs per collecting device)
    slots: int = 1 << 20        # GRU slots per collecting device (2 KB each)
    gather_rows: int = 4096
    chunk_turns: int = 32
    minibatch: int = 1024
    learners: int = 4
    model: ModelSpec = ModelSpec()


class ActOut(NamedTuple):
    action: jax.Array           # s32 [rows]
    log_prob: jax.Array         # f32 [rows]
    value: jax.Array            # f32 [rows]
    pool_h: jax.Array           # f32 [slots, hidden], donated in and out
    key: jax.Array              # u32 [2]


class GradOut(NamedTuple):
    grads: Params
    losses: jax.Array           # f32 [6]: policy, value, entropy, approx_kl, clip_fraction, samples


class ApplyOut(NamedTuple):
    params: Params
    adam: object
    grad_norms: jax.Array       # f32 [2]; -1 when the step was skipped


@dataclass(frozen=True)
class Programs:
    act: Callable[..., ActOut]
    value: Callable[..., jax.Array]
    gather_states: Callable[..., jax.Array]
    grad_step: Callable[..., GradOut]
    apply: Callable[..., ApplyOut]


def build(config: ProgramConfig, features: FeatureLayout, critic: FeatureLayout) -> Programs:
    """act(params, features, critic, slots, fresh, pool_h, key) -> ActOut
    value(params, features, critic, slots, fresh, pool_h) -> f32 [rows]
    gather_states(pool_h, slots, fresh) -> f32 [gather_rows, hidden]
    grad_step(params, features, critic, valid, action, log_prob, advantage, target, start_h,
              clip, entropy_weight, value_weight) -> GradOut
    apply(params, adam, grads: tuple[Params, ...learners], learning_rate, max_grad_norm) -> ApplyOut
    """
    ...
