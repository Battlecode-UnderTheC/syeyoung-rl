"""Batched environment on the native sim.

The engine's turn order is sequential within a round, so what gets batched is a
*slot*: dragon 0 of every arena, then dragon 1 of every arena, and so on. Each
slot yields one block of observations to run the policy on, which is what turns
a per-dragon Python call into one call per few hundred dragons.

Arena count per instance should stay modest (~128); past that the working set
falls out of cache and per-core throughput drops sharply. Scale with more
instances on more threads instead.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass

import numpy as np

_CPP = pathlib.Path(__file__).resolve().parents[2] / "cpp"
if str(_CPP) not in sys.path:
    sys.path.insert(0, str(_CPP))

try:
    import _dragons
except ImportError as error:             # pragma: no cover
    raise ImportError(
        "the native sim is not built; run tools/build_native.sh") from error

from dragons_env.mapgen import MapSpec


@dataclass
class SlotBatch:
    """One block of dragons that all act at the same point in the round."""
    rows: int
    grid: np.ndarray          # [rows, C, 7, 7] float32
    scalars: np.ndarray       # [rows, D] float32
    mask: np.ndarray          # [rows, A] bool
    arena: np.ndarray         # [rows] int32
    dragon: np.ndarray        # [rows] int32

    def __len__(self) -> int:
        return self.rows


class NativeVecEnv:
    def __init__(self, specs: list[MapSpec], seed: int = 0, max_sprint: int = 3,
                 split_sizes: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9),
                 allow_portal_steps: bool = True, ego_rotate: bool = True):
        self.specs = list(specs)
        self.seed = seed
        self.space = _dragons.ActionSpace(max_sprint, list(split_sizes),
                                          allow_portal_steps)
        self.vec = _dragons.VecGame([s.to_text() for s in self.specs], seed,
                                    self.space, ego_rotate)
        self.n_actions = self.space.size

    @property
    def arenas(self) -> int:
        return self.vec.arenas

    def begin_round(self) -> None:
        self.vec.begin_round()

    def end_round(self) -> None:
        self.vec.end_round()

    def slot(self, index: int) -> SlotBatch:
        rows, grid, scalars, mask, arena, dragon = self.vec.prepare_slot(index)
        return SlotBatch(rows, grid, scalars, mask, arena, dragon)

    def submit(self, actions: np.ndarray, sonar: np.ndarray | None = None,
               payloads: np.ndarray | None = None) -> None:
        self.vec.submit_slot(np.ascontiguousarray(actions, dtype=np.int32),
                             None if sonar is None else
                             np.ascontiguousarray(sonar, dtype=np.uint8),
                             None if payloads is None else
                             np.ascontiguousarray(payloads, dtype=np.uint32))

    def round_slots(self):
        """Iterate the slots of one round, yielding each non-empty block."""
        self.begin_round()
        index = 0
        while True:
            batch = self.slot(index)
            if batch.rows == 0:
                break
            yield batch
            index += 1
        self.end_round()

    def results(self) -> list[str]:
        return list(self.vec.results())

    def team_scores(self) -> np.ndarray:
        """[arenas, 2, 2] of (max length, total length) per team."""
        out = np.zeros((self.arenas, 2, 2), dtype=np.int32)
        for a in range(self.arenas):
            game = self.vec.game(a)
            for team in (0, 1):
                max_len, total, _ = game.team_score(team)
                out[a, team] = (max_len, total)
        return out
