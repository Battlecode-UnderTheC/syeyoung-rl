"""Reconstruct full state for the training-only critic.

The engine hands out per-dragon windows and nothing global, so the learner has
to assemble the picture itself. Because a self-play worker drives *both* sides,
the union of every dragon's window each round covers every dragon's own head and
its own length exactly -- which is what the score depends on. Pearls away from
all dragons are only known as "last seen", carried with an age channel so the
critic can tell fresh observation from stale memory.

This is privileged relative to any actor and approximate relative to the engine.
It must never be fed into an actor.
"""

from __future__ import annotations

import numpy as np

from dragons_env.mapgen import MapSpec, edge_between
from dragons_env.protocol import TurnBlock

# map channels
C_PEARL = 0
C_PEARL_AGE = 1
C_KELP = 2
C_PORTAL = 3
C_A_HEAD = 4
C_A_BODY = 5
C_B_HEAD = 6
C_B_BODY = 7
C_SEEN = 8
N_MAP_CHANNELS = 9

# swapping these channel pairs turns the state around to the other team's view
TEAM_CHANNEL_PAIRS = ((C_A_HEAD, C_B_HEAD), (C_A_BODY, C_B_BODY))

N_DRAGON_FEATURES = 4


class GlobalStateTracker:
    def __init__(self, spec: MapSpec, max_dragons: int = 32):
        self.width, self.height = spec.width, spec.height
        self.max_dragons = max_dragons
        self.static = np.zeros((2, spec.height, spec.width), dtype=np.float32)
        kelp = set(spec.kelp)
        portals = {idx for idx, _ in spec.portals}
        for y in range(spec.height):
            for x in range(spec.width):
                for d in "NESW":
                    idx = edge_between(x, y, d, spec.width, spec.height)
                    if idx in kelp:
                        self.static[0, y, x] += 0.25
                    if idx in portals:
                        self.static[1, y, x] = 1.0
        self.pearl = np.zeros((spec.height, spec.width), dtype=np.float32)
        self.pearl_age = np.full((spec.height, spec.width), 1.0, dtype=np.float32)
        self.dragons: dict[int, dict] = {}
        self.round = 0
        self._teams = np.zeros((spec.height, spec.width, 4), dtype=np.float32)

    # -- updating ---------------------------------------------------------

    def begin_round(self, round_num: int) -> None:
        self.round = round_num
        self.pearl_age = np.minimum(self.pearl_age + 1.0 / 64.0, 1.0)
        self._teams.fill(0.0)

    def observe(self, dragon_id: int, team: str, block: TurnBlock) -> None:
        xs, ys = np.asarray(block.xs), np.asarray(block.ys)
        self.pearl[ys, xs] = block.pearl
        self.pearl_age[ys, xs] = 0.0

        base = 0 if team == "A" else 2
        for part in block.parts:
            channel = base + (0 if part.is_head else 1)
            if part.team != team:
                channel = (2 if base == 0 else 0) + (0 if part.is_head else 1)
            self._teams[part.y, part.x, channel] = 1.0

        record = self.dragons.setdefault(dragon_id, {"team": team, "born": self.round})
        record.update(length=block.length, team=team, seen=self.round, alive=True)

    def kill(self, dragon_id: int) -> None:
        if dragon_id in self.dragons:
            self.dragons[dragon_id]["alive"] = False

    # -- snapshot ---------------------------------------------------------

    def map_tensor(self) -> np.ndarray:
        out = np.zeros((N_MAP_CHANNELS, self.height, self.width), dtype=np.float32)
        out[C_PEARL] = self.pearl
        out[C_PEARL_AGE] = self.pearl_age
        out[C_KELP] = self.static[0]
        out[C_PORTAL] = self.static[1]
        out[C_A_HEAD] = self._teams[..., 0]
        out[C_A_BODY] = self._teams[..., 1]
        out[C_B_HEAD] = self._teams[..., 2]
        out[C_B_BODY] = self._teams[..., 3]
        out[C_SEEN] = 1.0 - self.pearl_age
        return out

    def dragon_tensor(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feats = np.zeros((self.max_dragons, N_DRAGON_FEATURES), dtype=np.float32)
        teams = np.zeros(self.max_dragons, dtype=np.int64)
        mask = np.zeros(self.max_dragons, dtype=bool)
        living = [(did, r) for did, r in self.dragons.items() if r.get("alive")]
        for slot, (did, record) in enumerate(living[: self.max_dragons]):
            length = record.get("length", 0)
            feats[slot] = (length / 20.0, np.log1p(length) / 4.0, 1.0,
                           (self.round - record["born"]) / 500.0)
            teams[slot] = 0 if record["team"] == "A" else 1
            mask[slot] = True
        return feats, teams, mask

    def team_score(self, team: str) -> tuple[int, int]:
        """(longest living dragon, total living length) -- the authoritative score."""
        lengths = [r.get("length", 0) for r in self.dragons.values()
                   if r.get("alive") and r["team"] == team]
        return (max(lengths) if lengths else 0, sum(lengths))
