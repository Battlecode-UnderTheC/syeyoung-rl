"""The native lifetime state must match the Python reference tensor for tensor.

Both are driven through identical matches on the native sim: Python from the
engine-exact turn block text, C++ straight from the sim's window. Every
prepared tensor, the prefix traces and the split tables are compared on each
turn, across sprints, splits, kelp, portals, sonar and deaths.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from spatial_helpers import _dragons
from dragons_env import mapgen
from dragons_env.mapgen import MapSpec, edge_between, north, snake, west
from dragons_env.protocol import parse_spawn, parse_turn
from snake_rl.env.actions import CommandCodec
from snake_rl.env.memory import LifetimeKey, LifetimeState
from snake_rl.env.observation import portal_id_table, prepare
from snake_rl.env.schema import (N_ENTITY_FEATURES, N_HISTORY_FEATURES, N_PREFIX_FEATURES,
                                 N_STATUS, PF)

pytestmark = pytest.mark.skipif(_dragons is None, reason="native sim not built")
codec = CommandCodec(8)
TENSORS = ("local", "facts", "visible_xy", "visible_prev", "visible_age_before", "status",
           "history", "history_mask", "entities", "entity_mask", "portal_pairs", "portal_axis",
           "portal_unpaired")


def board(seed: int) -> MapSpec:
    rng = random.Random(seed)
    spec = mapgen.random_map(rng, min_size=9, max_size=15, dragons_per_team=rng.choice((1, 2)),
                             length=rng.choice((3, 5, 8)), kelp_density=0.05, portal_pairs=2,
                             pearl_gap_choices=((2, 5), (4, 9)))
    return spec


def choose(rng: random.Random, py_state: LifetimeState, obs, native) -> str:
    """A command that the private masks allow, sometimes fatal on purpose."""
    if obs.split_legal and rng.random() < 0.15:
        sizes, _ = codec.split_table(obs)
        return f"SPLIT {int(rng.choice(list(sizes)))}"
    sim = codec.begin_move(obs, py_state)
    dirs = ""
    for _ in range(rng.choice((1, 1, 2, 3, 4))):
        mask = sim.mask()
        allowed = [t for t in range(4) if mask[t]]
        if not allowed:
            break
        f = sim.features()
        safe = [t for t in allowed
                if not (f[PF.AHEAD_KELP_N + t] or f[PF.OCC_SELF_N + t] or f[PF.OCC_OTHER_N + t]
                        or f[PF.OCC_HEAD_N + t])]
        prefer = [t for t in safe if t != (obs.heading + 2) % 4] or safe
        if not prefer or rng.random() < 0.03:
            prefer = allowed                       # an occasional deliberate death
        tok = rng.choice(prefer)
        dirs += "NESW"[tok]
        sim.advance(tok)
        if not f[PF.LENGTH_KNOWN] or f[PF.LENGTH] * obs.width * obs.height <= 3:
            break
    return "MOVE " + (dirs or "NESW"[obs.heading])


@pytest.mark.parametrize("seed", range(6))
def test_native_prepare_matches_python(seed):
    spec = board(seed)
    game = _dragons.Game.from_map_text(spec.to_text(), seed)
    ids = portal_id_table(spec)
    rng = random.Random(seed)
    py: dict[int, LifetimeState] = {}
    cc: dict[int, object] = {}
    known = 0
    compared = 0
    while game.round < 60 and not game.finished():
        game.begin_round()
        idx = -1
        while idx + 1 < len(game.dragons()):
            idx += 1
            while known < len(game.dragons()):
                info = parse_spawn(game.spawn_block(known))
                py[info.dragon_id] = LifetimeState(LifetimeKey(0, info.team, info.dragon_id),
                                                   info.width, info.height, info.unit_limit)
                cc[info.dragon_id] = _dragons.Lifetime(info.dragon_id, 0 if info.team == "A" else 1,
                                                       info.width, info.height, info.unit_limit, 8)
                known += 1
            d = game.dragons()[idx]
            if not d["alive"]:
                continue
            did = d["id"]
            block = parse_turn(game.turn_block(idx, ids), spec.width, spec.height)
            obs = prepare(py[did], block)
            nat = cc[did].prepare(game, idx, ids)
            for name in TENSORS:
                a, b = getattr(obs, name), np.asarray(nat[name])
                assert a.shape == b.shape, (seed, game.round, did, name, a.shape, b.shape)
                assert np.allclose(a, b, atol=2e-6), (seed, game.round, did, name,
                                                       np.abs(a - b).max() if a.size else 0)
            assert list(nat["inbox"]) == list(block.messages)
            assert tuple(nat["head"]) == obs.head and nat["heading"] == obs.heading
            assert nat["length"] == obs.length and nat["split_legal"] == obs.split_legal
            assert [None if c is None else tuple(c) for c in py[did].body.cells] == \
                [None if c is None else tuple(c) for c in cc[did].body]
            sizes, feats = codec.split_table(obs)
            nsizes, nfeats = cc[did].split_table()
            assert np.array_equal(sizes, np.asarray(nsizes)) and np.allclose(feats, nfeats, atol=1e-6)
            # a redelivery changes nothing on either side
            obs2 = prepare(py[did], block)
            nat2 = cc[did].prepare(game, idx, ids)
            assert np.array_equal(obs2.facts, np.asarray(nat2["facts"]))
            assert np.array_equal(obs.visible_prev, np.asarray(nat2["visible_prev"]))
            cmd = choose(rng, py[did], obs, cc[did])
            if cmd.startswith("MOVE"):
                dirs = cmd.split()[1]
                tr = codec.trace_move(obs, py[did], dirs)
                tokens, active, masks, prefix = cc[did].trace_move(dirs)
                assert np.array_equal(tr.tokens, np.asarray(tokens))
                assert np.array_equal(tr.active, np.asarray(active))
                assert np.array_equal(tr.masks, np.asarray(masks))
                assert np.allclose(tr.prefix, np.asarray(prefix), atol=1e-6), (seed, dirs)
                py[did].commit_command(block, "move", dirs)
                cc[did].commit_move(dirs)
                action = _dragons.Action.move(dirs, rng.random() < 0.3, rng.randrange(2 ** 32))
            else:
                n = int(cmd.split()[1])
                py[did].commit_command(block, "split", split_n=n)
                cc[did].commit_split(n)
                action = _dragons.Action.split(n, rng.random() < 0.3, rng.randrange(2 ** 32))
            game.take_turn(idx, action)
            compared += 1
        game.end_round()
    assert compared >= 20, compared


def test_native_schema_sizes_match():
    assert N_STATUS == 36 and N_HISTORY_FEATURES == 30 and N_ENTITY_FEATURES == 16
    assert N_PREFIX_FEATURES == 49
