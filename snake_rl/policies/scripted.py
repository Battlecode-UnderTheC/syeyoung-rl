"""Scripted baselines: an evaluation opponent and a sanity-check random walker.

Both only ever pick actions the mask allows, so they die to enemies and to being
boxed in, never to a wall or their own body.
"""

from __future__ import annotations

import random

import numpy as np

from ..env.actions import ActionSpace
from ..env.match import Action, Obs
from dragons_env.protocol import VISION, VISION_RADIUS

_ROW = np.arange(VISION * VISION) // VISION
_COL = np.arange(VISION * VISION) % VISION

def _neighbour_table() -> tuple[np.ndarray, np.ndarray]:
    """[49, 4] window neighbour of each tile in N E S W order, and whether it
    lies inside the window at all."""
    idx = np.zeros((VISION * VISION, 4), dtype=np.intp)
    inside = np.zeros((VISION * VISION, 4), dtype=bool)
    for i in range(VISION * VISION):
        row, col = divmod(i, VISION)
        for d, (dr, dc) in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
            r, c = row + dr, col + dc
            if 0 <= r < VISION and 0 <= c < VISION:
                idx[i, d] = r * VISION + c
                inside[i, d] = True
    return idx, inside

_NEIGHBOUR, _NEIGHBOUR_INSIDE = _neighbour_table()


class RandomMaskedPolicy:
    def __init__(self, space: ActionSpace | None = None, seed: int = 0,
                 split_probability: float = 0.0, split_when_trapped: bool = False):
        self.space = space or ActionSpace()
        self.rng = random.Random(seed)
        self.split_probability = split_probability
        self.split_when_trapped = split_when_trapped

    def __call__(self, obs: Obs) -> Action:
        mask = obs.action_mask.copy()
        if self.rng.random() >= self.split_probability:
            mask[self.space.n_moves:] = False
        legal = np.flatnonzero(mask)
        if legal.size == 0 and self.split_when_trapped:
            legal = np.flatnonzero(obs.action_mask)
        if legal.size == 0:
            return Action(0)
        return Action(int(self.rng.choice(list(legal))))


class GreedyPearlPolicy:
    """Walks towards the nearest visible pearl, eating what it can on the way.

    Sprinting costs length, so a longer path only wins if it collects enough
    pearls to pay for itself.

    It does not split. Splitting is tempting when boxed in -- a split replaces
    the move, so it survives a turn no move could -- but it also halves the
    longest dragon, which is the number that decides the match, and a trapped
    dragon usually just stalls into the same trap. `split_when_trapped` turns
    that stall on for comparison.
    """

    def __init__(self, space: ActionSpace | None = None, seed: int = 0,
                 sprint_cost: float = 1.5, pearl_value: float = 2.0,
                 room_value: float = 1.5, split_when_trapped: bool = False):
        self.space = space or ActionSpace()
        self.rng = random.Random(seed)
        self.sprint_cost = sprint_cost
        self.pearl_value = pearl_value
        self.room_value = room_value
        self.split_when_trapped = split_when_trapped
        self._steps = self.space.path_steps

    def __call__(self, obs: Obs) -> Action:
        space, block = self.space, obs.block
        legal = obs.action_mask[: space.n_moves]
        if not legal.any():
            if self.split_when_trapped:
                splits = np.flatnonzero(obs.action_mask[space.n_moves:])
                if splits.size:
                    return Action(space.n_moves + int(splits[0]))
            return Action(0)

        cells = space.path_cells(block.facing)
        real = space.path_real(block.facing)
        pearls = np.asarray(block.pearl, dtype=np.float32)

        eaten = (pearls[cells] * real).sum(axis=1)
        landing = space.landing_cell(block.facing)

        pearl_tiles = np.flatnonzero(block.pearl)
        if pearl_tiles.size:
            # straight-line distance inside the window is enough: it is only a
            # 7x7 neighbourhood and the mask already rules out the walls
            dist = (np.abs(_ROW[landing][:, None] - _ROW[pearl_tiles][None, :])
                    + np.abs(_COL[landing][:, None] - _COL[pearl_tiles][None, :]))
            closeness = dist.min(axis=1).astype(np.float32)
        else:
            centre = VISION_RADIUS
            closeness = -(np.abs(_ROW[landing] - centre)
                          + np.abs(_COL[landing] - centre)).astype(np.float32)

        # a pearl chaser with no sense of space walks into its own body, so
        # penalise landing somewhere with few ways out
        free = self._exits(block)[landing].astype(np.float32)

        score = (closeness - self.pearl_value * eaten
                 + self.sprint_cost * (self._steps - 1)
                 - self.room_value * free)
        score = np.where(legal, score, np.inf)
        best = np.flatnonzero(score == score.min())
        return Action(int(self.rng.choice(list(best))))

    @staticmethod
    def _exits(block) -> np.ndarray:
        """[49] how many of each tile's four sides can actually be stepped through."""
        blocked = block.occupied[_NEIGHBOUR] & _NEIGHBOUR_INSIDE
        return (~block.kelp & ~blocked).sum(axis=1)
