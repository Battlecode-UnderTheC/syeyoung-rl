"""Scripted evaluation baselines, working straight from a turn block.

They are opponents for held-out evaluation (spec 11, 17), never teachers: no
learner trajectory against them enters the training buffer.
"""

from __future__ import annotations

import random

import numpy as np

from dragons_env.protocol import DIR_INDEX, DIRS, OFFSET, TurnBlock, VISION, VISION_RADIUS

CENTER = VISION_RADIUS * VISION + VISION_RADIUS


class KeepHeadingPolicy:
    """Steps straight ahead, turning only when the tile ahead is blocked."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def __call__(self, dragon_id: int, block: TurnBlock) -> str:
        return "MOVE " + _safe_direction(block, self.rng)


class GreedyPearlPolicy:
    """Single steps toward the nearest visible pearl, avoiding known deaths."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def __call__(self, dragon_id: int, block: TurnBlock) -> str:
        options = _open_directions(block)
        if not options:
            return "MOVE " + block.facing
        pearls = np.flatnonzero(block.pearl)
        if pearls.size == 0:
            return "MOVE " + self.rng.choice(options)
        rows, cols = pearls // VISION, pearls % VISION
        best, score = None, None
        for d in options:
            dx, dy = OFFSET[d]
            r, c = VISION_RADIUS + dy, VISION_RADIUS + dx
            dist = (np.abs(rows - r) + np.abs(cols - c)).min()
            if score is None or dist < score:
                best, score = d, dist
        return "MOVE " + best


def _open_directions(block: TurnBlock) -> list[str]:
    out = []
    for d in DIRS:
        if block.kelp[CENTER, DIR_INDEX[d]]:
            continue
        dx, dy = OFFSET[d]
        i = (VISION_RADIUS + dy) * VISION + (VISION_RADIUS + dx)
        if block.occupied[i]:
            continue
        out.append(d)
    return out


def _safe_direction(block: TurnBlock, rng: random.Random) -> str:
    options = _open_directions(block)
    if block.facing in options:
        return block.facing
    return rng.choice(options) if options else block.facing
