"""Helpers for driving the real engine with scripted dragons."""
from __future__ import annotations

import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "dragons-env"))

import pytest

from dragons_env import WasmEngine, mapgen, parse_spawn, parse_turn
from dragons_env.mapgen import MapSpec, edge_between, snake


@pytest.fixture(scope="session")
def engine() -> WasmEngine:
    return WasmEngine()


class Record:
    def __init__(self) -> None:
        self.turns: list[tuple[int, int, object]] = []   # (round, dragon_id, TurnBlock)
        self.deaths: list[tuple[int, int, str]] = []
        self.teams: dict[int, str] = {}
        self.result = None

    def by(self, dragon_id: int):
        return [(r, b) for r, did, b in self.turns if did == dragon_id]

    def at(self, round_num: int, dragon_id: int):
        for r, did, block in self.turns:
            if r == round_num and did == dragon_id:
                return block
        return None

    def death_of(self, dragon_id: int):
        return next((d for d in self.deaths if d[0] == dragon_id), None)


def run_scripted(engine: WasmEngine, spec: MapSpec, plan: dict | None = None) -> Record:
    """`plan` maps dragon_id -> list of command strings (by round), or
    (dragon_id, round) -> command. Anything unscripted keeps its heading."""
    plan = plan or {}
    rec = Record()

    def command(dragon_id: int, block) -> str:
        if (dragon_id, block.round) in plan:
            return plan[(dragon_id, block.round)]
        seq = plan.get(dragon_id)
        if seq is not None and block.round < len(seq) and seq[block.round] is not None:
            return seq[block.round]
        return "MOVE " + block.facing

    def reply(dragon_id: int, raw: bytes) -> bytes:
        block = parse_turn(raw.decode())
        rec.turns.append((block.round, dragon_id, block))
        return (command(dragon_id, block) + "\n").encode()

    def spawned(dragon_id: int, raw: bytes) -> None:
        rec.teams[dragon_id] = parse_spawn(raw.decode()).team

    def died(dragon_id: int, round_num: int, reason: str) -> None:
        rec.deaths.append((dragon_id, round_num, reason))

    rec.result = engine.run(spec.to_bytes(), reply, spawned, died)
    return rec


def two_dragons(w=21, h=21, la=6, lb=6, ax=5, ay=10, bx=5, by=2,
                fa="E", fb="E", pearl_gap=(0, 0)) -> MapSpec:
    return MapSpec(w, h, [(0, snake(ax, ay, la, fa, w, h)),
                          (1, snake(bx, by, lb, fb, w, h))], pearl_gap=pearl_gap)


