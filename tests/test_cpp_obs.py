"""The native observation encoder must match the Python one exactly.

A policy is trained through one of these and deployed through the other, so any
difference is a silent train/serve skew. Both are driven from the same state,
with the real engine as the referee for that state.
"""
from __future__ import annotations

import pathlib
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "cpp"))
try:
    import _dragons
except ImportError:                      # pragma: no cover
    _dragons = None

from snake_rl.env import ActionSpace, ObservationEncoder
from dragons_env.protocol import parse_turn
from dragons_env import WasmEngine

from conftest import REL_TO_ABS, random_seven, scripted_action

pytestmark = pytest.mark.skipif(_dragons is None, reason="native sim not built")


@pytest.mark.parametrize("gap", [(0, 0), (1, 1), (4, 4)])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("ego", [True, False])
def test_native_and_python_encoders_agree(seed, gap, ego):
    spec = random_seven(random.Random(seed), pearl_gap=gap)
    py_space = ActionSpace()
    py_encoder = ObservationEncoder(ego_rotate=ego)
    cpp_space = _dragons.ActionSpace()
    assert cpp_space.size == py_space.n and cpp_space.n_moves == py_space.n_moves

    engine = WasmEngine()
    blocks: list[tuple] = []

    def reply(dragon_id: int, raw: bytes) -> bytes:
        block = parse_turn(raw.decode(), spec.width, spec.height)
        blocks.append((block.round, dragon_id, block))
        kind, arg, sonar, payload = scripted_action(
            seed, block.round, dragon_id, block.facing, block.length)
        lines = [f"MOVE {arg}"] if kind == "move" else [f"SPLIT {arg}"]
        if sonar:
            lines.append(f"SONAR {payload}")
        return ("\n".join(lines) + "\n").encode()

    engine.run(spec.to_bytes(), reply, None, None, None)

    game = _dragons.Game.from_map_text(spec.to_text())
    compared = 0
    cursor = 0
    while game.round < 500 and cursor < len(blocks):
        game.begin_round()
        index = 0
        while index < len(game.dragons()):
            dragon = game.dragons()[index]
            if not dragon["alive"]:
                index += 1
                continue
            assert cursor < len(blocks), "native sim outlived the engine"
            round_num, dragon_id, block = blocks[cursor]
            assert (round_num, dragon_id) == (game.round, dragon["id"])

            team = "A" if dragon["team"] == 0 else "B"
            py_grid = py_encoder.grid(block, dragon_id, team)
            py_scalars = py_encoder.scalars(block, 64)
            py_mask = py_space.mask(block, 64)

            cpp_grid, cpp_scalars = _dragons.encode(game, index, ego)
            cpp_mask = _dragons.legal_mask(game, index, cpp_space)

            assert np.allclose(cpp_grid, py_grid, atol=1e-6), (
                f"grid differs at round {round_num} dragon {dragon_id}; channels "
                f"{sorted(set(np.argwhere(~np.isclose(cpp_grid, py_grid))[:, 0].tolist()))}")
            assert np.allclose(cpp_scalars, py_scalars, atol=1e-6), (
                f"scalars differ: {cpp_scalars} vs {py_scalars}")
            assert np.array_equal(cpp_mask, py_mask), (
                f"mask differs at round {round_num} dragon {dragon_id}: "
                f"{np.flatnonzero(cpp_mask != py_mask).tolist()}")
            compared += 1

            kind, arg, sonar, payload = scripted_action(
                seed, game.round, dragon["id"], dragon["facing"], dragon["length"])
            action = (_dragons.Action.move(arg, sonar, payload) if kind == "move"
                      else _dragons.Action.split(arg, sonar, payload))
            game.take_turn(index, action)
            index += 1
            cursor += 1
        game.end_round()
        if game.finished():
            break
    assert compared > 0


def test_window_masking_is_enough_for_a_dragon_longer_than_its_window():
    """Why a 7x7 window is sufficient for masking, and why self-collision can
    only mean being boxed in.

    The Python mask reads window occupancy; the native one reads the whole
    board. They agree because a path is at most max_sprint steps and the window
    covers radius 3 -- so an unseen tail is never a reachable destination. This
    grows a dragon well past 7 tiles on a 15x15 board and checks they still
    agree, and that every path really does stay inside the window.
    """
    from dragons_env.mapgen import MapSpec, snake

    width = height = 15
    spec = MapSpec(width, height,
                   [(0, snake(7, 7, 3, "E", width, height)),
                    (1, snake(2, 12, 3, "E", width, height))],
                   pearl_gap=(1, 1))          # dense food so it grows fast
    space_py = ActionSpace()
    cpp_space = _dragons.ActionSpace()

    # every path a 3-step space can take stays within the window
    for path in space_py.paths:
        assert len(path) <= 3, "a longer path could leave the 7x7 view"

    game = _dragons.Game.from_map_text(spec.to_text())
    engine = WasmEngine()
    blocks: dict[tuple[int, int], object] = {}

    def reply(dragon_id, raw):
        block = parse_turn(raw.decode(), width, height)
        blocks[(block.round, dragon_id)] = block
        return ("MOVE " + block.facing + "\n").encode()

    engine.run(spec.to_bytes(), reply, None, None, None)

    longest_seen = 0
    compared = 0
    for _ in range(60):
        game.begin_round()
        index = 0
        while index < len(game.dragons()):
            dragon = game.dragons()[index]
            if not dragon["alive"]:
                index += 1
                continue
            block = blocks.get((game.round, dragon["id"]))
            if block is None:
                index += 1
                continue
            longest_seen = max(longest_seen, dragon["length"])
            assert np.array_equal(_dragons.legal_mask(game, index, cpp_space),
                                  space_py.mask(block, 64)), (
                f"masks differ at round {game.round} dragon {dragon['id']} "
                f"(length {dragon['length']})")
            compared += 1
            game.take_turn(index, _dragons.Action.move(dragon["facing"]))
            index += 1
        game.end_round()
        if game.finished():
            break

    assert compared > 0
    assert longest_seen > 7, (
        f"the dragon only reached {longest_seen}; it must exceed the 7-tile "
        "window for this to be testing anything")
