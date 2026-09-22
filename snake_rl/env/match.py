"""Run one match, with a policy callback per dragon-turn.

The engine drives the loop, so this is not a gym `step()` API: you give `run` a
policy and it returns once the match is over. A rollout worker keeps its own
per-dragon recurrent state in the closure, keyed by `dragon_id`.

Dragon ids are assigned in spawn order and are never reused, including for the
children of a split, so they serve as lifetime ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .actions import ActionSpace
from dragons_env.engine import DEBUG_NONE, MatchResult, WasmEngine
from .observation import ObservationEncoder
from dragons_env.protocol import SpawnInfo, TurnBlock, parse_spawn, parse_turn


@dataclass
class Obs:
    dragon_id: int
    team: str
    round: int
    grid: np.ndarray            # [C, 7, 7] float32
    scalars: np.ndarray         # [D] float32
    messages: np.ndarray        # [K, 4] uint8
    message_mask: np.ndarray    # [K] bool
    action_mask: np.ndarray     # [A] bool
    block: TurnBlock            # raw parse, for debugging and state tracking


@dataclass
class Action:
    index: int
    sonar: int | None = None    # uint32 payload, or None to stay quiet


@dataclass
class MatchOutcome:
    result: MatchResult
    deaths: list[tuple[int, int, str]] = field(default_factory=list)   # (id, round, reason)
    teams: dict[int, str] = field(default_factory=dict)
    final_lengths: dict[int, int] = field(default_factory=dict)
    turns: int = 0
    rounds: int = 0

    def death_reasons(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, _, reason in self.deaths:
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    def team_reward(self, team: str) -> float:
        if self.result.winner is None:
            return 0.0
        return 1.0 if self.result.winner == team else -1.0


class IllegalAction(RuntimeError):
    pass


class MatchRunner:
    def __init__(self, engine: WasmEngine, encoder: ObservationEncoder | None = None,
                 space: ActionSpace | None = None, strict: bool = False):
        self.engine = engine
        self.space = space or ActionSpace()
        self.encoder = encoder or ObservationEncoder(max_sprint=self.space.max_sprint)
        self.strict = strict

    def run(self, map_bytes: bytes, policy: Callable[[Obs], "Action | int"],
            on_spawn: Callable[[int, SpawnInfo], None] | None = None,
            on_death: Callable[[int, int, str], None] | None = None,
            debug: int = DEBUG_NONE) -> MatchOutcome:
        spawns: dict[int, SpawnInfo] = {}
        outcome = MatchOutcome(result=None)  # type: ignore[arg-type]

        def spawned(dragon_id: int, block: bytes) -> None:
            info = parse_spawn(block.decode())
            spawns[dragon_id] = info
            outcome.teams[dragon_id] = info.team
            if on_spawn is not None:
                on_spawn(dragon_id, info)

        def died(dragon_id: int, round_num: int, reason: str) -> None:
            outcome.deaths.append((dragon_id, round_num, reason))
            outcome.final_lengths.pop(dragon_id, None)
            if on_death is not None:
                on_death(dragon_id, round_num, reason)

        def reply(dragon_id: int, raw: bytes) -> bytes:
            info = spawns[dragon_id]
            block = parse_turn(raw.decode(), info.width, info.height)
            outcome.turns += 1
            outcome.rounds = max(outcome.rounds, block.round)
            outcome.final_lengths[dragon_id] = block.length

            messages, message_mask = self.encoder.messages(block)
            obs = Obs(
                dragon_id=dragon_id,
                team=info.team,
                round=block.round,
                grid=self.encoder.grid(block, dragon_id, info.team),
                scalars=self.encoder.scalars(block, info.unit_limit),
                messages=messages,
                message_mask=message_mask,
                action_mask=self.space.mask(block, info.unit_limit),
                block=block,
            )
            action = policy(obs)
            if isinstance(action, int):
                action = Action(action)
            if (self.strict and obs.action_mask.any()
                    and not obs.action_mask[action.index]):
                # a boxed-in dragon has nothing legal and must still send
                # something, so only an avoidable illegal choice is a bug
                raise IllegalAction(
                    f"dragon {dragon_id} round {block.round}: "
                    f"{self.space.describe(action.index)} is masked out")
            lines = [self.space.command(action.index, block)]
            if action.sonar is not None:
                lines.append(f"SONAR {int(action.sonar) & 0xFFFFFFFF}")
            return ("\n".join(lines) + "\n").encode()

        outcome.result = self.engine.run(map_bytes, reply, spawned, died, None, debug)
        return outcome


def team_score(lengths: dict[int, int], teams: dict[int, str], team: str) -> tuple[int, int]:
    """The authoritative score: (longest living dragon, total living length).

    Compared lexicographically; ties on both are a draw. Verified against the
    engine -- one dragon of 10 beats two of 7.
    """
    mine = [n for did, n in lengths.items() if teams.get(did) == team]
    return (max(mine) if mine else 0, sum(mine))
