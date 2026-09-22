"""Fixtures shared by the dragons-spatial-v1 tests: native games that emit the
engine's exact turn blocks, and lifetimes driven through them."""
from __future__ import annotations

import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "cpp"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "dragons-env"))

import numpy as np

from dragons_env.mapgen import MapSpec, edge_between, north, snake, west
from dragons_env.protocol import parse_spawn, parse_turn

from snake_rl.env.memory import LifetimeKey, LifetimeState
from snake_rl.env.prepare import prepare
from snake_rl.rollout.collector import portal_id_table

try:
    import _dragons
except ImportError:                      # pragma: no cover
    _dragons = None


def two_dragons(w=21, h=21, la=6, lb=6, ax=5, ay=10, bx=5, by=2, fa="E", fb="E",
                pearl_gap=(0, 0), kelp=(), portals=()) -> MapSpec:
    return MapSpec(w, h, [(0, snake(ax, ay, la, fa, w, h)), (1, snake(bx, by, lb, fb, w, h))],
                   kelp=list(kelp), portals=list(portals), pearl_gap=pearl_gap)


class NativeMatch:
    """A native game plus one LifetimeState per dragon id, stepped by hand."""

    def __init__(self, spec: MapSpec, seed: int = 0):
        self.spec = spec
        self.game = _dragons.Game.from_map_text(spec.to_text(), seed)
        self.ids = portal_id_table(spec)
        self.states: dict[int, LifetimeState] = {}
        self.blocks: dict[int, object] = {}
        self.prepared: dict[int, object] = {}
        self.known = 0
        self._spawn()

    def _spawn(self) -> None:
        dragons = self.game.dragons()
        while self.known < len(dragons):
            info = parse_spawn(self.game.spawn_block(self.known))
            self.states[info.dragon_id] = LifetimeState(
                LifetimeKey(0, info.team, info.dragon_id), info.width, info.height, info.unit_limit)
            self.known += 1

    def dragon(self, dragon_id: int) -> dict:
        for d in self.game.dragons():
            if d["id"] == dragon_id:
                return d
        raise KeyError(dragon_id)

    def index(self, dragon_id: int) -> int:
        for i, d in enumerate(self.game.dragons()):
            if d["id"] == dragon_id:
                return i
        raise KeyError(dragon_id)

    def observe(self, dragon_id: int):
        """The engine-exact block for this dragon now, run through prepare()."""
        idx = self.index(dragon_id)
        block = parse_turn(self.game.turn_block(idx, self.ids), self.spec.width, self.spec.height)
        self.blocks[dragon_id] = block
        obs = prepare(self.states[dragon_id], block)
        self.prepared[dragon_id] = obs
        return obs

    def act(self, dragon_id: int, text: str) -> None:
        """Commit `text` (e.g. "MOVE EE" or "SPLIT 3") for the dragon and execute it."""
        block = self.blocks.get(dragon_id)
        if block is None or block.round != self.game.round:
            self.observe(dragon_id)
            block = self.blocks[dragon_id]
        st = self.states[dragon_id]
        parts = text.split()
        if parts[0] == "MOVE":
            st.commit_command(block, "move", parts[1])
            action = _dragons.Action.move(parts[1])
        else:
            st.commit_command(block, "split", split_n=int(parts[1]))
            action = _dragons.Action.split(int(parts[1]))
        self.game.take_turn(self.index(dragon_id), action)
        self._spawn()

    def play_round(self, commands: dict[int, str] | None = None, default="keep") -> None:
        """One full round: every living dragon observes then acts, in id order,
        including children born during the round."""
        commands = commands or {}
        self.game.begin_round()
        idx = -1
        while idx + 1 < len(self.game.dragons()):
            idx += 1
            d = self.game.dragons()[idx]
            if not d["alive"]:
                continue
            self._spawn()
            obs = self.observe(d["id"])
            text = commands.get(d["id"])
            if text is None:
                text = "MOVE " + d["facing"] if default == "keep" else default
            self.act(d["id"], text)
        self.game.end_round()

    def truth_segments(self, dragon_id: int) -> list[tuple[int, int]]:
        return [tuple(p) for p in self.dragon(dragon_id)["segments"]]
