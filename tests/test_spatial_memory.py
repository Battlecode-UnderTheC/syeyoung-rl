"""Observation, privacy and memory acceptance tests (spec section 16).

Every block a lifetime sees here is the engine's exact protocol text, produced
by the native sim (tests/test_spatial_sim.py pins that text to the wasm
engine byte for byte), so the memory is tested on real observations.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from spatial_helpers import NativeMatch, _dragons, two_dragons
from dragons_env.mapgen import MapSpec, edge_between, north, snake, west
from dragons_env.protocol import parse_turn
from snake_rl.env.memory import UNKNOWN, LifetimeKey, LifetimeState
from snake_rl.env.prepare import ObservationError, prepare, validate_block
from snake_rl.env.schema import F, S, N_FACT_CHANNELS, Occupant, schema_hash, check_schema

pytestmark = pytest.mark.skipif(_dragons is None, reason="native sim not built")


# -- schema ---------------------------------------------------------------------

def test_schema_is_versioned_and_refuses_old_checkpoints():
    assert len(schema_hash()) == 16
    check_schema("dragons-spatial-v1", schema_hash())
    with pytest.raises(ValueError):
        check_schema("dragons-local-v0")
    with pytest.raises(ValueError):
        check_schema("dragons-spatial-v1", "0000000000000000")


# -- known empty vs unknown, ages, staleness --------------------------------------

def test_known_empty_differs_from_unknown_and_ages_are_computed_on_read():
    m = NativeMatch(two_dragons(w=21, h=21, la=4, lb=4, ax=5, ay=10, bx=15, by=10))
    m.game.begin_round()
    obs = m.observe(0)
    seen, visible = obs.facts[F.SEEN], obs.facts[F.VISIBLE]
    assert seen.sum() == 49 and visible.sum() == 49
    assert obs.facts[F.OCC + Occupant.EMPTY].sum() == 49 - 4, "observed empty tiles are recorded as empty"
    assert obs.facts[F.OCC + Occupant.EMPTY][0, 0] == 0 and seen[0, 0] == 0, "an unseen tile is not 'empty'"
    m.act(0, "MOVE E")
    m.act(1, "MOVE E")
    m.game.end_round()
    for _ in range(5):
        m.play_round()
    obs = m.observe(0)
    # the column the dragon left behind is stale: seen, not visible, aged
    x_old = 5 - 3
    assert obs.facts[F.SEEN, 10, x_old] == 1 and obs.facts[F.VISIBLE, 10, x_old] == 0
    assert obs.facts[F.AGE_SEEN, 10, x_old] == pytest.approx(6 / 500)


def test_a_last_seen_enemy_keeps_its_age_and_never_becomes_current_truth():
    """Dragon 1 is seen once, then walks out of view; its old cells stay
    recorded as enemy with growing age even after it has moved on."""
    m = NativeMatch(two_dragons(w=21, h=21, la=3, lb=3, ax=5, ay=10, bx=8, by=10, fa="W", fb="E"))
    m.game.begin_round()
    obs = m.observe(0)
    assert obs.facts[F.OCC + Occupant.ENEMY_HEAD, 10, 8] == 1
    m.act(0, "MOVE W")
    m.observe(1)
    m.act(1, "MOVE E")
    m.game.end_round()
    for _ in range(4):
        m.play_round()
    obs = m.observe(0)
    assert obs.facts[F.VISIBLE, 10, 8] == 0
    assert obs.facts[F.OCC + Occupant.ENEMY_HEAD, 10, 8] == 1, "the last sighting is retained"
    assert obs.facts[F.AGE_SEEN, 10, 8] > 0, "with its age, not as current truth"
    truth = m.truth_segments(1)
    assert (8, 10) not in truth, "the enemy really has moved on"


def test_an_empty_observation_clears_only_the_observed_cell():
    m = NativeMatch(two_dragons(w=21, h=21, la=3, lb=3, ax=5, ay=10, bx=8, by=10, fa="W", fb="E"))
    m.game.begin_round()
    m.observe(0)
    m.act(0, "MOVE W")
    m.observe(1)
    m.act(1, "MOVE E")
    m.game.end_round()
    st = m.states[0]
    assert st.facts.occupant_kind[10, 8] == Occupant.ENEMY_HEAD
    # dragon 0 turns back east so that (8,10) comes into view again, empty now
    for cmd in ("MOVE N", "MOVE E", "MOVE E", "MOVE E"):
        m.play_round({0: cmd, 1: "MOVE E"})
    m.game.begin_round()
    obs = m.observe(0)
    assert obs.facts[F.VISIBLE, 10, 8] == 1
    assert st.facts.occupant_kind[10, 8] == Occupant.EMPTY
    # a cell that never came back into view keeps its record
    far = [(y, x) for y in range(21) for x in range(21)
           if st.facts.seen[y, x] and not obs.facts[F.VISIBLE, y, x]]
    assert far, "some cells are remembered but out of view"


def test_reprocessing_a_turn_is_idempotent():
    m = NativeMatch(two_dragons(w=15, h=15, la=4, lb=4, ax=4, ay=7, bx=10, by=7))
    m.game.begin_round()
    obs1 = m.observe(0)
    st = m.states[0]
    before = st.snapshot()
    obs2 = m.observe(0)
    after = st.snapshot()
    for k in before["facts"]["arrays"]:
        assert np.array_equal(before["facts"]["arrays"][k], after["facts"]["arrays"][k]), k
    assert np.array_equal(obs1.facts, obs2.facts)
    assert np.array_equal(obs1.visible_prev, obs2.visible_prev)
    assert before["own_turns"] == after["own_turns"]


@pytest.mark.parametrize("facing", ["N", "E", "S", "W"])
def test_every_heading_writes_the_same_absolute_cells(facing):
    w = h = 15
    spec = MapSpec(w, h, [(0, snake(7, 7, 3, facing, w, h)), (1, snake(1, 1, 3, "E", w, h))],
                   kelp=[edge_between(9, 9, "N", w, h)], pearl_gap=(0, 0))
    m = NativeMatch(spec)
    m.game.begin_round()
    obs = m.observe(0)
    assert obs.facts[F.KELP + 0, 9, 9] == 1.0 and obs.facts[F.KELP + 2, 8, 9] == 1.0
    assert obs.facts[F.HEAD, 7, 7] == 1.0
    assert obs.local[F.HEAD, 3, 3] == 1.0, "the crop is north-up with the head at its centre"
    assert obs.status[S.HEADING_N + "NESW".index(facing)] == 1.0


# -- privacy ---------------------------------------------------------------------

def test_unseen_terrain_and_enemies_cannot_alter_prepared_tensors():
    """Two boards identical inside dragon 0's window but different far away
    produce identical actor tensors."""
    w = h = 25
    base = MapSpec(w, h, [(0, snake(5, 12, 4, "E", w, h)), (1, snake(20, 12, 4, "W", w, h))],
                   pearl_gap=(0, 0))
    other = MapSpec(w, h, [(0, snake(5, 12, 4, "E", w, h)), (1, snake(20, 3, 7, "S", w, h))],
                    kelp=[edge_between(20, 20, "N", w, h), edge_between(15, 15, "W", w, h)],
                    portals=[(north(18, 18, w), 3), (north(21, 2, w), 3)],
                    pearl_gap=(0, 0), tile_gaps={(22, 22): (1, 1)})
    a, b = NativeMatch(base), NativeMatch(other)
    a.game.begin_round(); b.game.begin_round()
    oa, ob = a.observe(0), b.observe(0)
    for name in ("local", "facts", "status", "history", "entities", "visible_prev"):
        assert np.array_equal(getattr(oa, name), getattr(ob, name)), name
    assert oa.portal_pairs.shape == ob.portal_pairs.shape == (0, 2, 2)


def test_the_child_starts_empty_and_the_parent_keeps_its_memory():
    m = NativeMatch(two_dragons(w=21, h=21, la=8, lb=3, ax=8, ay=10, bx=15, by=3))
    for _ in range(3):
        m.play_round()
    m.game.begin_round()
    m.observe(0)
    parent_seen = m.states[0].facts.seen.sum()
    parent_len = len(m.states[0].body)
    m.act(0, "SPLIT 3")
    child_id = max(m.states)
    assert child_id == 2
    child = m.states[child_id]
    assert child.body is None and child.facts.seen.sum() == 0 and child.own_turns == 0
    assert len(m.states[0].body) == parent_len - 3, "the parent's order is truncated"
    assert m.states[0].facts.seen.sum() == parent_seen, "the parent's facts are retained"
    assert m.states[0].history is not child.history
    m.observe(1); m.act(1, "MOVE E")
    obs = m.observe(child_id)
    assert child.body is not None and len(child.body) == 3
    assert child.body.cells[0] == tuple(m.truth_segments(child_id)[0])
    assert child.body.cells[1] is UNKNOWN, "the spawn body is not filled from the simulator"
    assert obs.status[S.FIRST_OWN_TURN] == 1.0


# -- own body order --------------------------------------------------------------

def _check_body(m: NativeMatch, dragon_id: int) -> tuple[int, int]:
    cells = m.states[dragon_id].body.cells
    truth = m.truth_segments(dragon_id)
    assert len(cells) == len(truth)
    known = 0
    for c, t in zip(cells, truth):
        if c is not UNKNOWN:
            assert c == t, (cells, truth)
            known += 1
    return known, len(truth)


def test_body_order_is_reconstructed_through_growth_sprints_and_splits():
    w = h = 21
    spec = MapSpec(w, h, [(0, snake(5, 10, 5, "E", w, h)), (1, snake(5, 2, 3, "E", w, h))],
                   pearl_gap=(0, 0), tile_gaps={(8, 10): (1, 1), (9, 10): (1, 1), (12, 10): (2, 2)})
    m = NativeMatch(spec)
    plans = ["MOVE E", "MOVE E", "MOVE EE", "MOVE E", "MOVE N", "MOVE N", "MOVE W", "MOVE WW", "MOVE W"]
    for cmd in plans:
        m.game.begin_round()
        m.observe(0)
        m.act(0, cmd)
        m.observe(1); m.act(1, "MOVE E")
        m.game.end_round()
        m.game.begin_round()
        m.observe(0)
        _check_body(m, 0)
        m.game.end_round()
    m.game.begin_round()
    m.observe(0)
    known, total = _check_body(m, 0)
    assert known == total, "after enough moves the whole body is known through the trail"
    assert m.states[0].body.tail_known
    m.act(0, "SPLIT 2")
    m.observe(1); m.act(1, "MOVE E")
    child = max(m.states)
    m.observe(child); m.act(child, "MOVE " + m.dragon(child)["facing"])
    m.game.end_round()
    m.game.begin_round()
    m.observe(0)
    known, total = _check_body(m, 0)
    assert known == total and total == len(m.truth_segments(0))


def test_body_order_through_a_known_portal_and_unknown_after_an_unseen_exit():
    w = h = 21
    # dragon 0 at (5,10) heading E; a vertical portal on the east edge of (7,10)
    # joined to the west edge of (15,3): the exit is far outside the window
    portals = [(west(8, 10, w), 4), (west(15, 3, w), 4)]
    spec = MapSpec(w, h, [(0, snake(5, 10, 4, "E", w, h)), (1, snake(2, 18, 3, "E", w, h))],
                   portals=portals, pearl_gap=(0, 0))
    m = NativeMatch(spec)
    m.play_round({0: "MOVE E"})
    m.play_round({0: "MOVE E"})       # head now at (7,10), portal ahead
    m.game.begin_round()
    obs = m.observe(0)
    assert obs.portal_unpaired.shape[0] == 1 and obs.portal_pairs.shape[0] == 0
    kind, landing = m.states[0].facts.step(7, 10, 1)
    assert kind == "portal" and landing is None, "the exit is not privately known"
    m.act(0, "MOVE E")
    m.observe(1); m.act(1, "MOVE E")
    m.game.end_round()
    m.game.begin_round()
    m.observe(0)
    assert m.truth_segments(0)[0] == (15, 3)
    cells = m.states[0].body.cells
    assert cells[0] == (15, 3) and cells[1] == (7, 10), "the observed head is known, the trail continues"
    _check_body(m, 0)
    assert m.states[0].history.last.intermediates_known, "a single step whose end is observed is fully known"
    # now both ends have been seen: the pair is confirmed and a return trip is traceable
    st = m.states[0]
    assert len(st.facts.confirmed_portal_pairs()) == 1
    kind, landing = st.facts.step(15, 3, 3)
    assert kind == "portal" and landing == (7, 10)


def test_contradicted_body_cells_are_invalidated_not_repaired():
    m = NativeMatch(two_dragons(w=21, h=21, la=4, lb=3, ax=5, ay=10, bx=5, by=2))
    m.game.begin_round()
    m.observe(0)
    st = m.states[0]
    st.body.cells[1] = (5, 9)          # a wrong hypothesis inside the window
    m.act(0, "MOVE E")
    m.observe(1); m.act(1, "MOVE E")
    m.game.end_round()
    m.game.begin_round()
    m.observe(0)
    assert (5, 9) not in st.body.cells
    _check_body(m, 0)


# -- raw contract ----------------------------------------------------------------

def test_validation_rejects_inconsistent_coordinates():
    m = NativeMatch(two_dragons(w=15, h=15, la=3, lb=3, ax=1, ay=7, bx=10, by=7))
    m.game.begin_round()
    block = parse_turn(m.game.turn_block(0, m.ids), 15, 15)
    validate_block(block, 15, 15)
    with pytest.raises(ObservationError):
        validate_block(block, 16, 15)       # the wrapped column no longer matches
    with pytest.raises(ObservationError):
        validate_block(block, 15, 9)        # the head is off a 9-row board


def test_snapshot_and_restore_round_trip():
    m = NativeMatch(two_dragons(w=17, h=13, la=5, lb=4, ax=4, ay=6, bx=12, by=6,
                                portals=[(north(3, 3, 17), 1), (north(9, 9, 17), 1)]))
    for cmd in ("MOVE E", "MOVE N", "MOVE NN", "MOVE W"):
        m.play_round({0: cmd})
    st = m.states[0]
    snap = st.snapshot()
    m.game.begin_round()
    obs_a = m.observe(0)
    st2 = LifetimeState.from_snapshot(snap)
    obs_b = prepare(st2, m.blocks[0])
    for name in ("local", "facts", "status", "history", "visible_prev", "visible_age_before"):
        assert np.array_equal(getattr(obs_a, name), getattr(obs_b, name)), name
    assert st2.body.cells == st.body.cells
