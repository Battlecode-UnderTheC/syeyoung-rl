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


# Shared with the environment repo's differential tests: a deterministic action
# script and a 7x7 board generator. Kept here because the encoder tests need
# them and the simulator's own tests live in the dragons-env submodule.
REL_TO_ABS = {
    "N": {"F": "N", "L": "W", "R": "E"}, "E": {"F": "E", "L": "N", "R": "S"},
    "S": {"F": "S", "L": "E", "R": "W"}, "W": {"F": "W", "L": "S", "R": "N"},
}


def scripted_action(seed, round_num, dragon_id, facing, length):
    """Depends only on (round, dragon id) and the dragon's own facing, so both
    engines can be driven through the identical sequence."""
    rng = random.Random(f"{seed}:{round_num}:{dragon_id}")
    if length >= 5 and rng.random() < 0.10:
        return ("split", rng.randint(2, length - 2), False, 0)
    steps = 1 if rng.random() < 0.65 else rng.randint(2, 3)
    dirs, at = [], facing
    for _ in range(steps):
        at = REL_TO_ABS[at][rng.choice("FLR")]
        dirs.append(at)
    sonar = rng.random() < 0.25
    return ("move", "".join(dirs), sonar, rng.randrange(2 ** 32) if sonar else 0)


def random_seven(rng, pearl_gap=(0, 0)) -> MapSpec:
    """A 7x7 board. `pearl_gap` with min == max keeps spawning deterministic, so
    pearls can be compared without reproducing the engine's own RNG seed."""
    w = h = 7
    kelp = set()
    for y in range(h):
        for x in range(w):
            for direction in ("N", "W"):
                if rng.random() < 0.05:
                    kelp.add(edge_between(x, y, direction, w, h))
    spec = MapSpec(w, h, [(0, snake(1, 1, 3, "E", w, h)),
                          (1, snake(5, 5, 3, "W", w, h))],
                   kelp=sorted(kelp), pearl_gap=pearl_gap)
    spawn = {p for _, segs in spec.dragons for p in segs}
    spec.kelp = [e for e in spec.kelp
                 if all(e != edge_between(x, y, d, w, h)
                        for x, y in spawn for d in "NESW")]
    return spec
