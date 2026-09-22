"""The native sim's turn blocks must equal the wasm engine's byte for byte.

The spatial actor is trained on blocks the native sim prints and deployed on
blocks the judge prints, so any difference in text is a train/serve skew. Both
engines are driven through the same scripted commands, including splits,
sprints, sonar after a split, kelp and portals.
"""
from __future__ import annotations

import random

import pytest

from spatial_helpers import _dragons
from dragons_env import WasmEngine, parse_turn
from dragons_env.mapgen import MapSpec, edge_between, north, snake, west
from snake_rl.rollout.collector import portal_id_table

pytestmark = pytest.mark.skipif(_dragons is None, reason="native sim not built")

REL = {"N": {"F": "N", "L": "W", "R": "E"}, "E": {"F": "E", "L": "N", "R": "S"},
       "S": {"F": "S", "L": "E", "R": "W"}, "W": {"F": "W", "L": "S", "R": "N"}}


def scripted(seed, round_num, dragon_id, facing, length):
    r = random.Random(f"{seed}:{round_num}:{dragon_id}")
    if length >= 6 and r.random() < 0.08:
        return ("split", r.randint(2, length - 2), True, r.randrange(2 ** 32))
    steps = 2 if (length >= 6 and r.random() < 0.15) else 1
    dirs, at = [], facing
    for _ in range(steps):
        at = REL[at]["F" if r.random() < 0.8 else r.choice("LR")]
        dirs.append(at)
    sonar = r.random() < 0.3
    return ("move", "".join(dirs), sonar, r.randrange(2 ** 32) if sonar else 0)


def board(seed: int) -> MapSpec:
    rng = random.Random(seed)
    w, h = rng.randrange(9, 14), rng.randrange(9, 14)
    spec = MapSpec(w, h, [(0, snake(2, 2, 5, "E", w, h)), (1, snake(w - 3, h - 3, 5, "W", w, h))],
                   pearl_gap=(3, 3), tile_gaps={(0, 0): (0, 0)})
    spawn = {p for _, segs in spec.dragons for p in segs}
    forbidden = {edge_between(x, y, d, w, h) for x, y in spawn for d in "NESW"}
    kelp = {edge_between(x, y, d, w, h) for y in range(h) for x in range(w) for d in "NW"
            if rng.random() < 0.04}
    spec.kelp = sorted(kelp - forbidden)
    used = set(spec.kelp) | forbidden
    for pid, horizontal in enumerate((True, False)):
        picked = []
        while len(picked) < 2:
            x, y = rng.randrange(w), rng.randrange(h)
            idx = north(x, y, w) if horizontal else west(x, y, w)
            if idx in used or idx in picked:
                continue
            picked.append(idx)
        used.update(picked)
        spec.portals += [(picked[0], pid), (picked[1], pid)]
    return spec


@pytest.mark.parametrize("seed", range(8))
def test_native_turn_blocks_match_the_wasm_engine(engine, seed):
    spec = board(seed)
    blocks, spawns = {}, {}

    def spawned(did, raw):
        spawns[did] = raw.decode()

    def reply(did, raw):
        blk = parse_turn(raw.decode())
        blocks[(blk.round, did)] = raw.decode()
        kind, arg, sonar, payload = scripted(seed, blk.round, did, blk.facing, blk.length)
        lines = [f"MOVE {arg}"] if kind == "move" else [f"SPLIT {arg}"]
        if sonar:
            lines.append(f"SONAR {payload}")
        return ("\n".join(lines) + "\n").encode()

    engine.run(spec.to_bytes(), reply, spawned, None, None)

    game = _dragons.Game.from_map_text(spec.to_text())
    ids = portal_id_table(spec)
    compared = 0
    for d in range(len(game.dragons())):
        assert game.spawn_block(d) == spawns[d]
    while game.round < 500 and not game.finished():
        game.begin_round()
        idx = -1
        while idx + 1 < len(game.dragons()):
            idx += 1
            dr = game.dragons()[idx]
            if not dr["alive"]:
                continue
            ref = blocks[(game.round, dr["id"])]
            assert game.turn_block(idx, ids) == ref
            compared += 1
            blk = parse_turn(ref)
            kind, arg, sonar, payload = scripted(seed, blk.round, dr["id"], blk.facing, blk.length)
            action = (_dragons.Action.move(arg, sonar, payload) if kind == "move"
                      else _dragons.Action.split(arg, sonar, payload))
            game.take_turn(idx, action)
        game.end_round()
    assert compared >= 4
