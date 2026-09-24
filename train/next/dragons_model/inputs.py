"""The model's inputs, read from the byte records agent::EncodeFeatures and
train::EncodeCritic write. Casting and reshaping only: every channel is
computed in C++ (agent/features.hpp), once, for the trainer and the bot alike.
"""

from __future__ import annotations

from typing import NamedTuple

import jax


class FeatureLayout(NamedTuple):
    """Parsed from the layout description C++ writes; fields by name."""
    bytes: int
    fields: dict            # name -> (dtype, shape, offset)

    @staticmethod
    def parse(description: str) -> "FeatureLayout": ...


class Inputs(NamedTuple):
    window: jax.Array       # [B, 49, C] f32
    portal_ids: jax.Array   # [B, 49, 4] i32
    scalars: jax.Array      # [B, S] f32
    mask: jax.Array         # [B, 24] bool
    near: jax.Array         # [B, Cn, 13, 13] f32   (memory, full resolution)
    far: jax.Array          # [B, Cf, 13, 13] f32   (memory, coarse)
    distances: jax.Array    # [B, Cd, 13, 13] f32   (distance fields sampled on the near view)


class CriticInputs(NamedTuple):
    grid: jax.Array         # [B, 8, 16, 16] f32
    scalars: jax.Array      # [B, 12] f32


def decode(features: jax.Array, layout: FeatureLayout) -> Inputs:
    """features: u8 [B, layout.bytes]."""
    ...


def decode_critic(critic: jax.Array, layout: FeatureLayout) -> CriticInputs:
    ...
