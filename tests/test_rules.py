"""Regression tests for the engine rules verified in GAME_ANALYSIS.md.

These pin the behaviour the training harness depends on. If the toolkit ever
ships a new engine and one of these fails, the analysis document is stale.
"""
from __future__ import annotations

import pytest

from conftest import run_scripted, two_dragons
from dragons_env.mapgen import MapSpec, edge_between, north, snake, west


# -- movement, sprinting ---------------------------------------------------

@pytest.mark.parametrize("length,steps,fatal", [
    (6, 1, False), (6, 5, False), (6, 6, True),
    (5, 4, False), (5, 5, True),
    (3, 2, False), (3, 3, True),
    (2, 1, False), (2, 2, True),
])
def test_sprint_costs_one_segment_per_extra_step(engine, length, steps, fatal):
    """k steps cost k-1 length and the remainder must stay >= 2; overspending kills."""
    spec = two_dragons(la=length, lb=3)
    rec = run_scripted(engine, spec, {0: ["MOVE " + "E" * steps]})
    death = rec.death_of(0)
    if fatal:
        assert death is not None and death[2] == "A"
    else:
        assert death is None or death[1] > 0
        after = rec.at(1, 0)
        assert after.length == length - (steps - 1)
        assert after.head == ((5 + steps) % 21, 10)


def test_reversing_into_own_neck_is_fatal(engine):
    rec = run_scripted(engine, two_dragons(la=4), {0: ["MOVE W"]})
    assert rec.death_of(0) == (0, 0, "S")


def test_tail_following_is_fatal(engine):
    """The tail does not vacate before the head arrives."""
    spec = two_dragons(la=4, lb=3)
    rec = run_scripted(engine, spec, {0: ["MOVE N", "MOVE W", "MOVE S"]})
    assert rec.death_of(0) == (0, 2, "S")


def test_sprint_checks_every_intermediate_tile(engine):
    """Landing beyond a body still dies if the path crosses it."""
    w = h = 21
    spec = MapSpec(w, h, [(0, snake(9, 10, 3, "N", w, h)),
                          (1, snake(6, 10, 6, "E", w, h))], pearl_gap=(0, 0))
    rec = run_scripted(engine, spec, {1: ["MOVE EEEEE"]})
    assert rec.death_of(1) == (1, 0, "O")


# -- collisions ------------------------------------------------------------

@pytest.mark.parametrize("la,lb", [(3, 3), (6, 3), (3, 6), (8, 2)])
def test_head_to_head_kills_both_whatever_the_lengths(engine, la, lb):
    w = h = 21
    spec = MapSpec(w, h, [(0, snake(8, 10, la, "E", w, h)),
                          (1, snake(11, 10, lb, "W", w, h))], pearl_gap=(0, 0))
    rec = run_scripted(engine, spec)
    assert {d[0] for d in rec.deaths} == {0, 1}
    assert all(d[2] == "H" for d in rec.deaths)
    assert rec.result.winner is None


# -- splitting -------------------------------------------------------------

def test_split_detaches_the_tail_reversed_and_keeps_the_parent_still(engine):
    spec = two_dragons(la=6, lb=3)
    rec = run_scripted(engine, spec, {0: ["SPLIT 3"]})
    parent_before = rec.at(0, 0)
    child = rec.at(0, 2)                      # the child acts later in the SAME round
    assert child is not None, "a split child takes its turn in the round it is born"
    assert child.length == 3
    assert child.head == (0, 10)              # the parent's old tail tip
    assert child.facing == "W"                # reversed relative to the parent
    parent_after = rec.at(1, 0)
    assert parent_after.length == 3
    assert parent_after.head == parent_before.head, "splitting replaces the move"
    assert parent_after.unit_count == 2


def test_split_is_an_action_so_a_move_in_the_same_turn_wins(engine):
    rec = run_scripted(engine, two_dragons(la=6, lb=3), {0: ["SPLIT 3\nMOVE N"]})
    after = rec.at(1, 0)
    assert after.unit_count == 1 and after.length == 6 and after.head == (5, 9)


def test_splitting_survives_a_turn_with_no_legal_move(engine):
    """A split replaces the move, so a boxed-in dragon can stall by splitting --
    until it is too short to split again, and then it dies of no valid action."""
    w = h = 15
    kelp = [edge_between(5, 5, "E", w, h), edge_between(5, 5, "N", w, h),
            edge_between(5, 5, "S", w, h)]
    spec = MapSpec(w, h, [(0, snake(5, 5, 4, "E", w, h)),
                          (1, snake(11, 11, 3, "E", w, h))],
                   kelp=kelp, pearl_gap=(0, 0))
    trapped = run_scripted(engine, spec, {0: ["MOVE E"]})
    assert trapped.death_of(0) == (0, 0, "W"), "every direction is a wall or itself"

    stalled = run_scripted(engine, spec, {0: ["SPLIT 2"] * 3})
    assert stalled.death_of(0) == (0, 1, "A"), "survives round 0, then has nothing left"


@pytest.mark.parametrize("size", [0, 1, 5, 100])
def test_illegal_split_sizes_kill(engine, size):
    rec = run_scripted(engine, two_dragons(la=6, lb=3), {0: [f"SPLIT {size}"]})
    assert rec.death_of(0) == (0, 0, "A")


def test_unit_limit_caps_splitting(engine):
    w = h = 21
    spec = MapSpec(w, h, [(0, snake(14, 10, 12, "E", w, h)),
                          (1, snake(5, 2, 3, "E", w, h))], pearl_gap=(0, 0), unit_limit=3)
    rec = run_scripted(engine, spec, {(0, 0): "SPLIT 2", (0, 1): "SPLIT 2", (0, 2): "SPLIT 2"})
    assert rec.death_of(0) == (0, 2, "A")


# -- illegal actions -------------------------------------------------------

@pytest.mark.parametrize("command,fatal", [
    ("MOVE E", False), ("MOVE X", True), ("MOVE", True), ("", True),
    ("move e", True), ("MOVE e", True),
    ("LOG hi\nMOVE E", False), ("FOO BAR\nMOVE E", False),
    ("MOVE E\nMOVE E", False),
])
def test_only_a_well_formed_action_keeps_you_alive(engine, command, fatal):
    rec = run_scripted(engine, two_dragons(la=6, lb=3), {(0, 1): command})
    death = rec.death_of(0)
    assert (death is not None and death == (0, 1, "A")) == fatal


# -- sonar -----------------------------------------------------------------

def test_sonar_reaches_the_first_body_in_front_even_an_enemy(engine):
    w, h = 40, 11
    spec = MapSpec(w, h, [(0, snake(4, 5, 3, "E", w, h)),
                          (1, snake(12, 5, 3, "E", w, h)),
                          (0, snake(20, 5, 3, "E", w, h))], pearl_gap=(0, 0))
    rec = run_scripted(engine, spec, {(0, 0): "SONAR 777\nMOVE E"})
    assert rec.at(0, 1).messages == [777], "the enemy in between intercepts it"
    assert rec.at(0, 2).messages == [], "the ally behind it hears nothing"


def test_sonar_is_blocked_by_kelp(engine):
    w, h = 40, 11
    spec = MapSpec(w, h, [(0, snake(4, 5, 3, "E", w, h)),
                          (1, snake(12, 5, 3, "E", w, h))],
                   kelp=[west(9, 5, w)], pearl_gap=(0, 0))
    rec = run_scripted(engine, spec, {(0, 0): "SONAR 777\nMOVE E"})
    assert rec.at(0, 1).messages == []


def test_sonar_wraps_the_torus_and_can_return_to_the_sender(engine):
    w, h = 40, 11
    spec = MapSpec(w, h, [(0, snake(4, 5, 3, "E", w, h)),
                          (1, snake(12, 4, 3, "E", w, h))], pearl_gap=(0, 0))
    rec = run_scripted(engine, spec, {(0, 0): "SONAR 777\nMOVE E"})
    assert rec.at(0, 1).messages == [], "nothing on that row"
    assert rec.at(1, 0).messages == [777], "the ray circled the row into its own body"


def test_sonar_travels_through_a_portal(engine):
    w = h = 21
    portals = [(west(9, 10, w), 0), (west(15, 15, w), 0)]
    spec = MapSpec(w, h, [(0, snake(5, 10, 3, "E", w, h)),
                          (1, snake(18, 15, 3, "E", w, h))],
                   portals=portals, pearl_gap=(0, 0))
    rec = run_scripted(engine, spec, {(0, 0): "SONAR 555\nMOVE E"})
    assert rec.at(0, 1).messages == [555]


def test_only_the_last_sonar_of_a_turn_is_sent(engine):
    w, h = 40, 11
    spec = MapSpec(w, h, [(0, snake(4, 5, 3, "E", w, h)),
                          (1, snake(12, 5, 3, "E", w, h))], pearl_gap=(0, 0))
    rec = run_scripted(engine, spec, {(0, 0): "SONAR 111\nSONAR 222\nMOVE E"})
    assert rec.at(0, 1).messages == [222]


# -- portals, pearls, turn order ------------------------------------------

def test_portal_teleports_the_head_and_keeps_direction(engine):
    w = h = 21
    portals = [(west(9, 10, w), 0), (west(15, 15, w), 0)]
    spec = MapSpec(w, h, [(0, snake(5, 10, 3, "E", w, h)),
                          (1, snake(5, 2, 3, "E", w, h))],
                   portals=portals, pearl_gap=(0, 0))
    rec = run_scripted(engine, spec)
    assert rec.at(3, 0).head == (8, 10)
    assert rec.at(4, 0).head == (15, 15) and rec.at(4, 0).facing == "E"


def test_eating_a_pearl_adds_one_segment(engine):
    spec = two_dragons(la=6, lb=6, pearl_gap=(1, 1))
    rec = run_scripted(engine, spec)
    lengths = [b.length for _, b in rec.by(0)[:4]]
    assert lengths == [6, 7, 8, 9]


def test_a_dead_dragon_drops_pearls_on_its_corpse(engine):
    """Half a corpse comes back as food -- kills feed the killer."""
    w = h = 21
    spec = MapSpec(w, h, [(0, snake(8, 10, 4, "E", w, h)),
                          (1, snake(8, 13, 3, "E", w, h))],
                   kelp=[west(10, 10, w)], pearl_gap=(0, 0))
    rec = run_scripted(engine, spec, {1: ["MOVE E", "MOVE S", "MOVE W", "MOVE N"] * 3})
    assert rec.death_of(0) == (0, 1, "W")
    watcher = rec.at(1, 1)
    pearls = {(watcher.xs[i], watcher.ys[i]) for i in range(49) if watcher.pearl[i]}
    assert pearls == {(7, 10), (9, 10)}, "alternating tiles, starting at the head"


def test_dragons_move_in_id_order_within_a_round(engine):
    """Dragon 1 sees dragon 0 already moved -- turns are sequential, not simultaneous."""
    w, h = 40, 11
    spec = MapSpec(w, h, [(0, snake(4, 5, 3, "E", w, h)),
                          (1, snake(8, 5, 3, "W", w, h))], pearl_gap=(0, 0))
    rec = run_scripted(engine, spec)
    seen = {(p.x, p.y) for p in rec.at(0, 1).parts if p.dragon_id == 0 and p.is_head}
    assert seen == {(5, 5)}, "id 1 observes id 0 at its post-move position"


# -- scoring ---------------------------------------------------------------

def _score_match(engine, dragons):
    w, h = 60, 21
    spec = MapSpec(w, h, [(t, snake(x, y, n, "E", w, h)) for t, x, y, n in dragons],
                   pearl_gap=(0, 0))
    return run_scripted(engine, spec).result


@pytest.mark.parametrize("dragons,winner", [
    # longest single dragon wins even when the other side has more total length
    ([(0, 20, 2, 10), (1, 20, 6, 7), (1, 40, 6, 7)], "A"),
    ([(0, 20, 2, 8), (0, 40, 2, 8), (1, 20, 6, 9)], "B"),
    # tie on longest -> total length decides
    ([(0, 20, 2, 10), (0, 40, 2, 3), (1, 20, 6, 10)], "A"),
    ([(0, 20, 2, 10), (0, 40, 2, 2), (1, 20, 6, 10), (1, 40, 6, 5)], "B"),
    # tie on both -> draw, whatever the dragon counts
    ([(0, 20, 2, 10), (1, 20, 6, 10)], None),
    ([(0, 20, 2, 10), (0, 35, 2, 2), (0, 45, 2, 2), (1, 20, 6, 10), (1, 40, 6, 4)], None),
])
def test_winner_is_longest_dragon_then_total_length(engine, dragons, winner):
    assert _score_match(engine, dragons).winner == winner
