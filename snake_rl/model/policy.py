"""Per-lifetime runtime for the spatial actor (spec sections 3 and 13)."""

from __future__ import annotations

from typing import Callable

import torch

from ..env.actions import Command, CommandCodec
from ..env.memory import LifetimeKey, LifetimeState
from ..env.observation import prepare
from .actor import SpatialActor, SpatialInputs


class SpatialPolicy:
    """Drives a `SpatialActor` one decision at a time, owning one private
    `LifetimeState` and one (M, h) pair per lifetime key.

    State is committed exactly once per turn, after the command is chosen. A
    split child arrives as a new dragon id and therefore a new lifetime with
    empty facts and zero M/h; the parent's state is left untouched. A dead
    lifetime's state is dropped once `on_death` is called.
    """

    def __init__(self, actor: SpatialActor, codec: CommandCodec | None = None,
                 device: str | torch.device = "cpu", deterministic: bool = False,
                 match_id: int = 0, on_transition: Callable[[dict], None] | None = None):
        self.actor = actor.to(device).eval()
        self.codec = codec or CommandCodec(actor.K)
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.match_id = match_id
        self.on_transition = on_transition
        self.states: dict[int, LifetimeState] = {}
        self.memory: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.generation: dict[int, int] = {}

    # -- lifecycle ------------------------------------------------------------

    def new_match(self, match_id: int) -> None:
        self.match_id = match_id
        self.states.clear()
        self.memory.clear()
        self.generation.clear()

    def on_spawn(self, dragon_id: int, info) -> None:
        gen = self.generation.get(dragon_id, -1) + 1
        self.generation[dragon_id] = gen
        key = LifetimeKey(self.match_id, info.team, dragon_id, gen)
        self.states[dragon_id] = LifetimeState(key, info.width, info.height, info.unit_limit)
        self.memory[dragon_id] = self.actor.initial_state(1, info.height, info.width, self.device)

    def on_death(self, dragon_id: int, round_num: int, reason: str) -> None:
        self.states.pop(dragon_id, None)
        self.memory.pop(dragon_id, None)

    # -- one decision ---------------------------------------------------------

    def decide(self, dragon_id: int, block) -> Command:
        state = self.states[dragon_id]
        obs = prepare(state, block, self.codec.max_steps)
        x = SpatialInputs.from_prepared([obs], self.codec, self.device)
        M_old, h_old = self.memory[dragon_id]
        with torch.no_grad():
            ctx, M_new, h_new = self.actor.encode_and_update(x, M_old, h_old)
            cmds, logp, stored = self.actor.decode_sample(
                ctx, x, [obs], [state], self.codec, self.deterministic)
        cmd = cmds[0]
        # commit exactly once
        self.memory[dragon_id] = (M_new, h_new)
        if cmd.top == 0:
            state.commit_command(block, "move", cmd.dirs)
        else:
            state.commit_command(block, "split", split_n=cmd.split_n)
        if self.on_transition is not None:
            self.on_transition({
                "dragon_id": dragon_id, "key": state.key, "round": block.round,
                "obs": obs, "command": cmd, "log_prob": float(logp[0]),
                "M_old": M_old, "h_old": h_old, "M_new": M_new, "h_new": h_new,
                "stored": stored,
            })
        return cmd

    def __call__(self, dragon_id: int, block) -> str:
        """Protocol text for one turn, for engine callbacks."""
        return self.decide(dragon_id, block).text + "\n"
