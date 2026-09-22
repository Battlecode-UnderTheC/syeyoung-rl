"""Event-timeline collection on the native sim (spec sections 8 and 12).

One match is one global timeline of action events. Event e is the instant
before a living dragon acts, after any pearl tick. We store the exact native
state there for the critic, then run the acting lifetime's policy on the
byte-exact turn block the sim generates, execute the command, and advance
through dead-id skips and round boundaries until another dragon can act or
the match is terminal.

Actor data is stored losslessly and sparsely: the raw turn block text, the
command trace and the log probability per decision, plus a snapshot of the
lifetime's private state and its (M, h) every `snapshot_every` turns. Dense
maps are never copied per turn; PPO replays them from the snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dragons_env.mapgen import MapSpec

import pathlib as _pathlib
import sys as _sys

_CPP = _pathlib.Path(__file__).resolve().parents[2] / "cpp"
if str(_CPP) not in _sys.path:
    _sys.path.insert(0, str(_CPP))

from dragons_env.protocol import parse_spawn, parse_turn

from ..env.actions import Command
from ..env.memory import LifetimeKey
from ..env.schema import DIR_INDEX, MAX_ROUNDS, LAST_ROUND
from ..model.critic import (EV_C_A_BODY, EV_C_A_HEAD, EV_C_B_BODY, EV_C_B_HEAD,
                            EV_C_COUNTDOWN, EV_C_FACING, EV_C_KELP, EV_C_NEVER,
                            EV_C_NEXT, EV_C_PEARL, EV_C_PORTAL, EV_N_DRAGON_FEATURES,
                            EV_N_MAP_CHANNELS, EV_N_PHASE, EV_N_TEAM_FEATURES)
from ..model.policy import SpatialPolicy


def portal_id_table(spec: MapSpec) -> list[int]:
    """[2*W*H] portal id per canonical edge (north edges, then west), -1 elsewhere."""
    w, h = spec.width, spec.height
    ids = [-1] * (2 * w * h)
    stride = 2 * (w + 1)
    for idx, pid in spec.portals:
        y, off = divmod(idx, stride)
        vertical = off >= w + 1
        if vertical:
            off -= w + 1
        x, y = off % w, y % h
        ids[(w * h if vertical else 0) + y * w + x] = pid
    return ids


@dataclass
class Turn:
    event: int                 # index into the match's event timeline
    round: int
    block: str                 # raw turn block text (lossless)
    command: Command
    log_prob: float


@dataclass
class LifetimeRecord:
    key: LifetimeKey
    width: int
    height: int
    unit_limit: int
    spawn_block: str
    learner: bool
    turns: list[Turn] = field(default_factory=list)
    # turn index -> (LifetimeState snapshot, M numpy, h numpy), taken *before* that turn
    snapshots: dict[int, tuple] = field(default_factory=dict)
    death: tuple[int, str] | None = None      # (round, reason)


@dataclass
class EventRecord:
    """One team-event: the exact pre-action state and what followed (12.2)."""
    index: int
    round: int
    acting: LifetimeKey
    acting_team: int
    map_grid: np.ndarray       # [20, H, W] float16
    dragons: np.ndarray        # [N, 16] float32
    dragon_team: np.ndarray    # [N] int64
    team_feats: np.ndarray     # [2, 4] float32
    phase: np.ndarray          # [3] float32
    delta_rounds: int = 0      # round boundaries crossed before the next event
    terminal: bool = False
    reward: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))


@dataclass
class MatchRecord:
    spec: MapSpec
    seed: int
    lifetimes: dict[LifetimeKey, LifetimeRecord]
    events: list[EventRecord]
    result: str                # "A", "B" or "draw"
    reward: np.ndarray         # [2] terminal team reward
    learner_teams: set[str]
    opponent_ids: dict[str, int]
    info: dict = field(default_factory=dict)


class ScriptedAdapter:
    """Wraps a scripted baseline `(dragon_id, block) -> command text` with the
    lifecycle hooks the collector expects."""

    def __init__(self, policy):
        self.policy = policy
        self.states: dict[int, object] = {}

    def on_spawn(self, dragon_id: int, info) -> None:
        self.states[dragon_id] = info

    def on_death(self, dragon_id: int, round_num: int, reason: str) -> None:
        self.states.pop(dragon_id, None)

    def __call__(self, dragon_id: int, block) -> str:
        text = self.policy(dragon_id, block)
        return text if text.endswith("\n") else text + "\n"


def _event_state(game, edges, born: dict[int, int], next_index: int,
                 summaries: dict[int, tuple[float, float, float, float]],
                 opponent_ids: dict[str, int]) -> tuple:
    W, H = game.width, game.height
    area = float(W * H)
    pearl, countdown = game.tiles()
    grid = np.zeros((EV_N_MAP_CHANNELS, H, W), dtype=np.float32)
    grid[EV_C_PEARL] = pearl
    grid[EV_C_COUNTDOWN] = np.where(countdown >= 0, 1.0 / (1.0 + np.maximum(countdown, 0)), 0.0)
    grid[EV_C_NEVER] = countdown < 0
    north, west = edges
    dirs = (north, np.roll(west, -1, axis=1), np.roll(north, -1, axis=0), west)
    for d, kind in enumerate(dirs):
        grid[EV_C_KELP + d] = kind == 1
        grid[EV_C_PORTAL + d] = kind == 2
    dragons = game.dragons()
    alive = [(i, d) for i, d in enumerate(dragons) if d["alive"]]
    feats = np.zeros((max(len(alive), 1), EV_N_DRAGON_FEATURES), dtype=np.float32)
    teams = np.zeros(max(len(alive), 1), dtype=np.int64)
    team_feats = np.zeros((2, EV_N_TEAM_FEATURES), dtype=np.float32)
    limit = float(game.unit_limit)
    for slot, (i, d) in enumerate(alive):
        team = d["team"]
        hx, hy = d["segments"][0]
        head_ch = EV_C_A_HEAD if team == 0 else EV_C_B_HEAD
        body_ch = EV_C_A_BODY if team == 0 else EV_C_B_BODY
        grid[head_ch, hy, hx] = 1.0
        for (x, y) in d["segments"][1:]:
            grid[body_ch, y, x] = 1.0
        grid[EV_C_FACING + DIR_INDEX[d["facing"]], hy, hx] = 1.0
        if i == next_index:
            grid[EV_C_NEXT, hy, hx] = 1.0
        L = d["length"]
        cov, body_known, h_mean, h_std = summaries.get(d["id"], (0.0, 0.0, 0.0, 0.0))
        ax, ay = 2 * np.pi * hx / W, 2 * np.pi * hy / H
        row = feats[slot]
        row[0:4] = (np.sin(ax), np.cos(ax), np.sin(ay), np.cos(ay))
        row[4 + DIR_INDEX[d["facing"]]] = 1.0
        row[8] = L / area
        row[9] = np.log1p(L) / np.log1p(area)
        row[10] = i / max(len(dragons), 1)
        row[11] = float(i == next_index)
        row[12] = cov
        row[13] = body_known
        row[14] = h_mean
        row[15] = h_std
        teams[slot] = team
        team_feats[team, 0] = max(team_feats[team, 0], L / area)
        team_feats[team, 1] += L / area
    for team in (0, 1):
        team_feats[team, 2] = game.alive_unit_count(team) / limit
        team_feats[team, 3] = opponent_ids.get("AB"[team], 0) / 16.0
    next_team = dragons[next_index]["team"] if next_index >= 0 else 0
    phase = np.array([game.round / MAX_ROUNDS, max(LAST_ROUND - game.round, 0) / MAX_ROUNDS,
                      float(next_team == 0)], dtype=np.float32)
    mask_n = len(alive)
    return grid.astype(np.float16), feats, teams, mask_n, team_feats, phase


def play_event_match(spec: MapSpec, policies: dict[str, object], learner_teams: set[str],
                     seed: int = 0, snapshot_every: int = 16, match_id: int = 0,
                     opponent_ids: dict[str, int] | None = None,
                     max_rounds: int = MAX_ROUNDS) -> MatchRecord:
    """Play one complete match on the native sim, recording the event timeline.

    `policies` maps team letter -> SpatialPolicy or ScriptedAdapter. Every
    policy owns its own per-lifetime state; nothing crosses between them.
    """
    import _dragons
    game = _dragons.Game.from_map_text(spec.to_text(), seed)
    portal_ids = portal_id_table(spec)
    edges = game.edges()
    opponent_ids = opponent_ids or {}
    for p in policies.values():
        if isinstance(p, SpatialPolicy):
            p.new_match(match_id)

    lifetimes: dict[LifetimeKey, LifetimeRecord] = {}
    by_id: dict[int, LifetimeRecord] = {}
    born: dict[int, int] = {}
    events: list[EventRecord] = []
    known_dragons = 0
    known_deaths = 0

    def spawn_new() -> None:
        nonlocal known_dragons
        dragons = game.dragons()
        while known_dragons < len(dragons):
            i = known_dragons
            d = dragons[i]
            text = game.spawn_block(i)
            info = parse_spawn(text)
            team = "AB"[d["team"]]
            policy = policies[team]
            policy.on_spawn(d["id"], info)
            key = (policy.states[d["id"]].key if isinstance(policy, SpatialPolicy)
                   else LifetimeKey(match_id, team, d["id"], 0))
            rec = LifetimeRecord(key, info.width, info.height, info.unit_limit, text,
                                 learner=team in learner_teams)
            lifetimes[key] = rec
            by_id[d["id"]] = rec
            born[d["id"]] = game.round
            known_dragons += 1

    def reap() -> None:
        nonlocal known_deaths
        deaths = game.deaths()
        while known_deaths < len(deaths):
            did, rnd, reason = deaths[known_deaths]
            rec = by_id.get(did)
            if rec is not None:
                rec.death = (rnd, reason)
            team = rec.key.team if rec is not None else None
            if team is not None:
                policies[team].on_death(did, rnd, reason)
            known_deaths += 1

    def summaries() -> dict[int, tuple[float, float, float, float]]:
        out = {}
        for policy in policies.values():
            if not isinstance(policy, SpatialPolicy):
                continue
            for did, st in policy.states.items():
                if st.body is None:
                    out[did] = (0.0, 0.0, 0.0, 0.0)
                    continue
                M, h = policy.memory[did]
                hv = h.detach().abs()
                out[did] = (float(st.facts.seen.mean()), st.body.known_fraction(),
                            float(hv.mean()), float(hv.std()))
        return out

    spawn_new()
    while game.round < max_rounds and not game.finished():
        game.begin_round()
        idx = -1
        while idx + 1 < len(game.dragons()):
            idx += 1
            d = game.dragons()[idx]
            if not d["alive"]:
                continue
            spawn_new()
            rec = by_id[d["id"]]
            e = len(events)
            grid, feats, teams, n, team_feats, phase = _event_state(
                game, edges, born, idx, summaries(), opponent_ids)
            events.append(EventRecord(e, game.round, rec.key, d["team"], grid, feats, teams,
                                      team_feats, phase))
            policy = policies[rec.key.team]
            block = parse_turn(game.turn_block(idx, portal_ids), spec.width, spec.height)
            turn_index = len(rec.turns)
            if isinstance(policy, SpatialPolicy):
                if turn_index % snapshot_every == 0:
                    st = policy.states[d["id"]]
                    M, h = policy.memory[d["id"]]
                    rec.snapshots[turn_index] = (st.snapshot(), M.squeeze(0).cpu().numpy().copy(),
                                                 h.squeeze(0).cpu().numpy().copy())
                box = {}
                policy.on_transition = lambda t, box=box: box.update(t)
                cmd = policy.decide(d["id"], block)
                policy.on_transition = None
                rec.turns.append(Turn(e, game.round, block_text(block, game, idx, portal_ids),
                                      cmd, box.get("log_prob", 0.0)))
                action = _to_action(cmd)
            else:
                text = policy(d["id"], block)
                rec.turns.append(Turn(e, game.round, game.turn_block(idx, portal_ids),
                                      _parse_command(text), 0.0))
                action = _text_to_action(text)
            game.take_turn(idx, action)
            reap()
            spawn_new()
        game.end_round()
        spawn_new()
        reap()

    result = game.result()
    reward = np.zeros(2, dtype=np.float32)
    if result == "A":
        reward[:] = (1.0, -1.0)
    elif result == "B":
        reward[:] = (-1.0, 1.0)
    terminal = game.finished()
    for i, ev in enumerate(events):
        if i + 1 < len(events):
            ev.delta_rounds = events[i + 1].round - ev.round
        else:
            ev.terminal = terminal
            ev.delta_rounds = 1
            if terminal:
                ev.reward = reward.copy()
    deaths = {}
    for _, _, reason in game.deaths():
        deaths[reason] = deaths.get(reason, 0) + 1
    return MatchRecord(spec, seed, lifetimes, events, result, reward, set(learner_teams),
                       dict(opponent_ids),
                       info={"rounds": game.round, "events": len(events), "deaths": deaths,
                             "terminal": terminal,
                             "a_score": game.team_score(0), "b_score": game.team_score(1)})


def block_text(block, game, idx, portal_ids) -> str:
    return game.turn_block(idx, portal_ids)


def _to_action(cmd: Command):
    import _dragons
    sonar = cmd.sonar != 0
    payload = cmd.sonar - 1 if sonar else 0
    if cmd.top == 0:
        return _dragons.Action.move(cmd.dirs, sonar, payload)
    return _dragons.Action.split(cmd.split_n, sonar, payload)


def _parse_command(text: str) -> Command:
    cmd = Command(0)
    for line in text.strip().splitlines():
        parts = line.split()
        if parts[0] == "MOVE":
            cmd.top, cmd.dirs = 0, parts[1]
        elif parts[0] == "SPLIT":
            cmd.top, cmd.split_n = 1, int(parts[1])
        elif parts[0] == "SONAR":
            cmd.sonar = (int(parts[1]) & 0xF) + 1
    return cmd


def _text_to_action(text: str):
    import _dragons
    action = None
    sonar, payload = False, 0
    for line in text.strip().splitlines():
        parts = line.split()
        if parts[0] == "MOVE":
            action = ("move", parts[1])
        elif parts[0] == "SPLIT":
            action = ("split", int(parts[1]))
        elif parts[0] == "SONAR":
            sonar, payload = True, int(parts[1]) & 0xFFFFFFFF
    if action is None:
        return _dragons.Action.none()
    if action[0] == "move":
        return _dragons.Action.move(action[1], sonar, payload)
    return _dragons.Action.split(action[1], sonar, payload)
