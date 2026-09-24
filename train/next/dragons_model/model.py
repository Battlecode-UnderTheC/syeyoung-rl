"""The actor (ships in the bot) and the critic (training only).

The only learned memory is the GRU: the map is kept by code and arrives
through `Inputs`. Parameters are pytrees built by `init`; the forward
functions are pure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax

from .inputs import CriticInputs, Inputs


@dataclass(frozen=True)
class ModelSpec:
    hidden: int = 512
    compute_dtype: str = "float16"     # matmuls and convolutions; accumulation and the rest in float32


class ActorParams(NamedTuple):
    window: dict
    near: dict
    far: dict
    scalars: dict
    fuse: dict
    gru: dict
    policy: dict


class CriticParams(NamedTuple):
    grid: dict
    scalars: dict
    hidden: dict
    head: dict


class Params(NamedTuple):
    actor: ActorParams
    critic: CriticParams


def init(spec: ModelSpec, key: jax.Array) -> Params:
    ...


def actor_step(spec: ModelSpec, params: ActorParams, inputs: Inputs, h: jax.Array) -> tuple[jax.Array, jax.Array]:
    """One turn for B dragons. h: f32 [B, hidden]. -> (masked log-probabilities [B, 24], h')"""
    ...


def critic_value(spec: ModelSpec, params: CriticParams, inputs: CriticInputs, h: jax.Array) -> jax.Array:
    """The acting team's value, f32 [B]; no gradient reaches `h`."""
    ...


def entropy(log_probs: jax.Array, mask: jax.Array) -> jax.Array:
    ...
