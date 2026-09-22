"""The action space, and exact legality masks built from the 7x7 window.

One action per turn: either a move (a path of 1..`max_sprint` steps) or a split.
A sprint of k steps costs k-1 length and the engine checks every intermediate
tile, so a path is only legal if the whole path is.

Steps are relative to the current heading -- F/L/R. Reversing is never useful:
step 0 backwards is your own neck, and after any step the tile behind you is
your own body, so a reverse step is always fatal. Leaving it out of the space
loses nothing and removes the commonest way to die.

With `max_sprint <= 3` every tile of every path lies inside the window, so the
mask is exact for everything except an enemy moving onto you (head-to-head) and
paths that cross a portal, whose far side is not observable.
"""

from __future__ import annotations

import itertools

import numpy as np

from dragons_env.protocol import (CENTER, DIR_INDEX, KELP, LEFT, MIN_SIZE, RIGHT, TurnBlock,
                       VISION)

REL = ("F", "L", "R")
# window step: moving this way changes (row, col) by ...
STEP = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}


def turn(facing: str, rel: str) -> str:
    if rel == "F":
        return facing
    return LEFT[facing] if rel == "L" else RIGHT[facing]


class ActionSpace:
    """Flat discrete space: move paths first, then split sizes."""

    def __init__(self, max_sprint: int = 3, split_sizes: tuple[int, ...] = tuple(range(2, 10)),
                 allow_portal_steps: bool = True):
        if max_sprint < 1:
            raise ValueError("max_sprint must be >= 1")
        self.max_sprint = max_sprint
        # A step through a portal lands on a tile no window can show, so the mask
        # cannot promise it is safe. Turning this off makes the mask exact.
        self.allow_portal_steps = allow_portal_steps
        self.paths: list[tuple[str, ...]] = []
        for k in range(1, max_sprint + 1):
            self.paths.extend(itertools.product(REL, repeat=k))
        self.split_sizes = tuple(split_sizes)
        self.n_moves = len(self.paths)
        self.n = self.n_moves + len(self.split_sizes)
        self.verifiable = max_sprint <= 3
        self._split_sizes = np.asarray(self.split_sizes, dtype=np.int32)
        # k steps cost k-1 segments and the remainder must stay >= MIN_SIZE
        self._min_length = np.array([len(p) - 1 + MIN_SIZE for p in self.paths],
                                    dtype=np.int32)
        self._build_tables()

    def _build_tables(self) -> None:
        """Per heading, the window tiles and edges every move path touches.

        The window is a fixed 7x7 grid around the head, so for a given heading
        each path always visits the same cells. Precomputing them turns masking
        into a handful of gathers instead of a Python walk per action.
        """
        steps = self.max_sprint
        shape = (self.n_moves, steps)
        self._edge_tile = np.zeros((4,) + shape, dtype=np.intp)
        self._edge_dir = np.zeros((4,) + shape, dtype=np.intp)
        self._cell = np.zeros((4,) + shape, dtype=np.intp)
        self._real = np.zeros((4,) + shape, dtype=bool)
        self._inside = np.ones((4, self.n_moves), dtype=bool)

        for f, start in enumerate(("N", "E", "S", "W")):
            for a, path in enumerate(self.paths):
                at, facing = CENTER, start
                for j, rel in enumerate(path):
                    facing = turn(facing, rel)
                    self._edge_tile[f, a, j] = at
                    self._edge_dir[f, a, j] = DIR_INDEX[facing]
                    dr, dc = STEP[facing]
                    row, col = divmod(at, VISION)
                    row, col = row + dr, col + dc
                    if not (0 <= row < VISION and 0 <= col < VISION):
                        self._inside[f, a] = False
                        break
                    at = row * VISION + col
                    self._cell[f, a, j] = at
                    self._real[f, a, j] = True

    # -- encoding ---------------------------------------------------------

    def is_split(self, index: int) -> bool:
        return index >= self.n_moves

    def command(self, index: int, block: TurnBlock) -> str:
        """The engine command line for this action, given the dragon's heading."""
        if self.is_split(index):
            return f"SPLIT {self.split_sizes[index - self.n_moves]}"
        facing = block.facing
        dirs = []
        for rel in self.paths[index]:
            facing = turn(facing, rel)
            dirs.append(facing)
        return "MOVE " + "".join(dirs)

    def describe(self, index: int) -> str:
        if self.is_split(index):
            return f"split {self.split_sizes[index - self.n_moves]}"
        return "move " + "".join(self.paths[index])

    # -- masking ----------------------------------------------------------

    def mask(self, block: TurnBlock, unit_limit: int) -> np.ndarray:
        """Boolean mask of actions that cannot kill this dragon by its own doing.

        Window occupancy is enough even for a dragon far longer than its
        window: a path is at most `max_sprint` (<= 3) steps, so every tile it
        can reach lies within the 7x7 view. An unseen tail is therefore never a
        candidate destination, and self-collision is only ever possible when
        nothing at all is legal -- that is being boxed in, not a blind spot.

        Exact, except that a step through a portal lands where no window can
        see (see `allow_portal_steps`) and that nothing can rule out an enemy
        moving onto your head later in the round.
        """
        f = DIR_INDEX[block.facing]
        real = self._real[f]
        tiles, dirs, cells = self._edge_tile[f], self._edge_dir[f], self._cell[f]

        kelp = block.kelp[tiles, dirs] & real
        portal = block.portal[tiles, dirs] & real
        blocked = block.occupied[cells] & real

        has_portal = portal.any(axis=1)
        # a portal is only offered as a lone first step, and then its landing
        # tile goes unchecked
        lone_portal = portal[:, 0] & (real.sum(axis=1) == 1) if self.allow_portal_steps \
            else np.zeros(self.n_moves, dtype=bool)
        clear = self._inside[f] & ~kelp.any(axis=1) & ~blocked.any(axis=1)

        mask = np.empty(self.n, dtype=bool)
        mask[: self.n_moves] = ((block.length >= self._min_length)
                                & ((~has_portal & clear) | lone_portal))
        mask[self.n_moves:] = (block.unit_count < unit_limit) & (
            self._split_sizes >= MIN_SIZE) & (block.length - self._split_sizes >= MIN_SIZE)
        return mask


    def path_cells(self, facing: str) -> np.ndarray:
        """[n_moves, max_sprint] window tiles each move path visits."""
        return self._cell[DIR_INDEX[facing]]

    def path_real(self, facing: str) -> np.ndarray:
        """[n_moves, max_sprint] which of those entries are real steps."""
        return self._real[DIR_INDEX[facing]]

    def landing_cell(self, facing: str) -> np.ndarray:
        """[n_moves] the window tile each move path ends on."""
        real = self._real[DIR_INDEX[facing]]
        cells = self._cell[DIR_INDEX[facing]]
        last = np.maximum(real.sum(axis=1) - 1, 0)
        return cells[np.arange(self.n_moves), last]

    @property
    def path_steps(self) -> np.ndarray:
        """[n_moves] number of steps, so the sprint cost is this minus one."""
        return np.array([len(p) for p in self.paths], dtype=np.int32)

    def crosses_portal(self, index: int, block: TurnBlock) -> bool:
        """Whether this action steps through a portal, i.e. lands somewhere the
        window cannot show. Such a transition is unverifiable, not illegal."""
        if self.is_split(index):
            return False
        at = CENTER
        facing = block.facing
        for rel in self.paths[index]:
            facing = turn(facing, rel)
            edge = block.edge(at, facing)
            if edge not in (".", KELP):
                return True
            if edge == KELP:
                return False
            dr, dc = STEP[facing]
            row, col = divmod(at, VISION)
            row, col = row + dr, col + dc
            if not (0 <= row < VISION and 0 <= col < VISION):
                return False
            at = row * VISION + col
        return False


def encode_sonar(symbols: "np.ndarray | list[int]", bits: int = 8) -> int:
    """Symbols -> the uint32 the engine carries, one symbol per `bits`.

    The default packs four symbols into the four bytes of the word, so a vocab
    of 16 uses the low nibble of each byte and a vocab of 256 uses all of it.
    """
    mask = (1 << bits) - 1
    value = 0
    for i, sym in enumerate(symbols):
        value |= (int(sym) & mask) << (bits * i)
    return value & 0xFFFFFFFF


def decode_sonar(value: int, count: int = 4, bits: int = 8) -> list[int]:
    mask = (1 << bits) - 1
    return [(int(value) >> (bits * i)) & mask for i in range(count)]
