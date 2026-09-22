"""Tests for the RL-facing wrapper: action encoding, observations, masking."""
from __future__ import annotations

import random

import numpy as np
import pytest

from conftest import run_scripted, two_dragons
from snake_rl.env import Action, ActionSpace, MatchRunner, ObservationEncoder, mapgen
from snake_rl.env.actions import decode_sonar, encode_sonar
from dragons_env.mapgen import MapSpec, edge_between, snake
from snake_rl.env.observation import C_KELP, C_OWN_HEAD, C_PEARL
from dragons_env.protocol import parse_turn


def test_action_space_size_and_encoding():
    space = ActionSpace(max_sprint=3)
    assert space.n_moves == 3 + 9 + 27 and space.n == space.n_moves + 8

    block = parse_turn(_block(facing="N"))
    assert space.command(space.paths.index(("F",)), block) == "MOVE N"
    assert space.command(space.paths.index(("L",)), block) == "MOVE W"
    assert space.command(space.paths.index(("R",)), block) == "MOVE E"
    # relative turns compose: right then right is a U turn through east
    assert space.command(space.paths.index(("R", "R")), block) == "MOVE ES"
    assert space.command(space.n_moves + 0, block) == "SPLIT 2"


@pytest.mark.parametrize("vocab,bits", [(16, 8), (256, 8), (16, 4)])
def test_sonar_symbols_round_trip(vocab, bits):
    rng = random.Random(0)
    for _ in range(50):
        symbols = [rng.randrange(vocab) for _ in range(4)]
        assert decode_sonar(encode_sonar(symbols, bits), 4, bits) == symbols


def _block(facing: str = "N", length: int = 6) -> str:
    """A minimal synthetic turn block: empty 7x7 window, dragon alone at the centre."""
    lines = [f"ROUND 0", f"DIR {facing}", f"LENGTH {length}", "UNIT_COUNT 1", "NUM_MSGS 0"]
    for row in range(7):
        for col in range(7):
            lines.append(f"{col} {row} 0 5")
    lines.append("DRAGON_BODIES 1")
    lines.append(f"A 0 3 3 {facing} 1")
    lines += [" ".join(["."] * 7) for _ in range(8)]
    lines += [" ".join(["."] * 8) for _ in range(7)]
    return "\n".join(lines)


@pytest.mark.parametrize("facing", ["N", "E", "S", "W"])
def test_ego_rotation_puts_forward_up_for_every_heading(engine, facing):
    """A wall in front lands in the same ego cell and channel whatever the heading."""
    w = h = 15
    spec = MapSpec(w, h, [(0, snake(7, 7, 3, facing, w, h)),
                          (1, snake(1, 1, 3, "E", w, h))], pearl_gap=(0, 0))
    # kelp on the far side of the tile straight ahead
    dx, dy = mapgen.OFFSET[facing]
    ahead = ((7 + dx) % w, (7 + dy) % h)
    spec.kelp = [edge_between(ahead[0], ahead[1], facing, w, h)]

    encoder = ObservationEncoder(ego_rotate=True)
    grids = {}

    def reply(dragon_id, raw):
        block = parse_turn(raw.decode())
        if dragon_id == 0 and block.round == 0:
            grids["g"] = encoder.grid(block, 0, "A")
        return ("MOVE " + block.facing + "\n").encode()

    engine.run(spec.to_bytes(), reply, None, None, None)
    grid = grids["g"]
    assert grid[C_OWN_HEAD, 3, 3] == 1.0, "own head sits at the window centre"
    # ego channel C_KELP+0 is 'forward'; the wall is on the far side of (2,3)
    assert grid[C_KELP + 0, 2, 3] == 1.0
    assert grid[C_KELP + 0, 3, 3] == 0.0, "nothing blocking the first step"


def test_observation_shapes_and_message_encoding(engine):
    encoder = ObservationEncoder(max_messages=4)
    space = ActionSpace()
    runner = MatchRunner(engine, encoder, space, strict=True)
    seen = {}

    def policy(obs):
        seen.setdefault("obs", obs)
        legal = np.flatnonzero(obs.action_mask)
        return Action(int(legal[0]) if legal.size else 0, sonar=0xDEADBEEF)

    runner.run(two_dragons(la=6, lb=6).to_bytes(), policy)
    obs = seen["obs"]
    assert obs.grid.shape == (encoder.n_channels, 7, 7)
    assert obs.scalars.shape == (encoder.n_scalars,)
    assert obs.messages.shape == (4, 4) and obs.message_mask.shape == (4,)
    assert obs.action_mask.shape == (space.n,)


def test_a_masked_illegal_action_is_caught_in_strict_mode(engine):
    runner = MatchRunner(engine, strict=True)
    space = runner.space

    def policy(obs):
        # walking straight backwards is never in the space, so force a masked split
        illegal = int(np.flatnonzero(~obs.action_mask)[0])
        return Action(illegal)

    with pytest.raises(Exception):
        runner.run(two_dragons(la=3, lb=3).to_bytes(), policy)


# -- the important one -----------------------------------------------------

def _fuzz(engine, seed, matches, *, portals=0, dragons_per_team=1, allow_split=True,
          allow_portal_steps=True):
    """Play random masked actions and report how each dragon died."""
    rng = random.Random(seed)
    space = ActionSpace(allow_portal_steps=allow_portal_steps)
    runner = MatchRunner(engine, space=space)
    deaths: list[tuple[int, int, str]] = []
    trapped: set[int] = set()
    used_portal: set[int] = set()

    for _ in range(matches):
        spec = mapgen.random_map(rng, min_size=11, max_size=16,
                                 dragons_per_team=dragons_per_team,
                                 kelp_density=0.10, portal_pairs=portals)
        state = {"trapped": set(), "portal": set()}

        def policy(obs, state=state):
            legal = np.flatnonzero(obs.action_mask)
            if not allow_split:
                legal = legal[legal < space.n_moves]
            if legal.size == 0:
                state["trapped"].add(obs.dragon_id)
                return Action(0)
            choice = int(rng.choice(list(legal)))
            if space.crosses_portal(choice, obs.block):
                # the far side is unobservable, so the mask promises nothing there
                state["portal"].add(obs.dragon_id)
            return Action(choice)

        outcome = runner.run(spec.to_bytes(), policy)
        deaths += outcome.deaths
        trapped |= state["trapped"]
        used_portal |= state["portal"]
    return deaths, trapped, used_portal


def test_masking_removes_every_self_inflicted_death(engine):
    """With the mask applied, the only way to die is an enemy head-on -- or being
    genuinely boxed in with no legal action left."""
    deaths, trapped, _ = _fuzz(engine, seed=7, matches=6, portals=0)
    assert deaths, "the fuzz should actually kill somebody"
    bad = [d for d in deaths if d[2] != "H" and d[0] not in trapped]
    assert not bad, f"unexpected self-inflicted deaths: {bad}"
    assert not [d for d in deaths if d[2] == "A"], "an illegal action was let through"


def test_masking_holds_with_splitting_and_crowds(engine):
    deaths, trapped, _ = _fuzz(engine, seed=11, matches=4, dragons_per_team=3)
    bad = [d for d in deaths if d[2] != "H" and d[0] not in trapped]
    assert not bad, f"unexpected self-inflicted deaths: {bad}"


def test_portal_steps_are_the_only_hole_in_the_mask(engine):
    """Crossing a portal is offered but cannot be verified; nothing else leaks."""
    deaths, trapped, portal = _fuzz(engine, seed=3, matches=4, portals=2)
    bad = [d for d in deaths if d[2] != "H" and d[0] not in trapped and d[0] not in portal]
    assert not bad, f"unexpected self-inflicted deaths: {bad}"


def test_refusing_portal_steps_makes_the_mask_exact(engine):
    deaths, trapped, portal = _fuzz(engine, seed=3, matches=4, portals=2,
                                    allow_portal_steps=False)
    assert not portal, "no action should cross a portal when they are refused"
    bad = [d for d in deaths if d[2] != "H" and d[0] not in trapped]
    assert not bad, f"unexpected self-inflicted deaths: {bad}"
