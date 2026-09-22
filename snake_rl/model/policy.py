"""Glue: drive `MatchRunner` with a torch actor, one private GRU state per dragon.

This is the decentralised execution path. Each dragon gets its own hidden state
keyed by its lifetime id; a split child starts from zeros because it is a new
id, and a dead dragon's state is dropped and never reused. Nothing is shared
between dragons except the weights.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from ..env.match import Action as EnvAction
from ..env.match import Obs as EnvObs
from .actor import Actor, ActorObs, JointAction


class TorchPolicy:
    """Wraps an `Actor` so it can be handed straight to `MatchRunner.run`."""

    def __init__(self, actor: Actor, device: str | torch.device = "cpu",
                 deterministic: bool = False, transmit: bool = True,
                 on_transition: Callable[[dict], None] | None = None,
                 record_entropy: bool = False):
        self.actor = actor.to(device).eval()
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.transmit = transmit
        self.on_transition = on_transition
        self.record_entropy = record_entropy
        self.hidden: dict[int, torch.Tensor] = {}

    # -- lifecycle hooks --------------------------------------------------

    def on_spawn(self, dragon_id: int, info) -> None:
        self.hidden[dragon_id] = self.actor.initial_state(1, self.device)

    def on_death(self, dragon_id: int, round_num: int, reason: str) -> None:
        self.hidden.pop(dragon_id, None)

    def reset(self) -> None:
        self.hidden.clear()

    # -- acting -----------------------------------------------------------

    def to_tensors(self, obs: EnvObs) -> ActorObs:
        t = lambda x, dtype: torch.as_tensor(np.ascontiguousarray(x), dtype=dtype,
                                             device=self.device).unsqueeze(0)
        return ActorObs(
            grid=t(obs.grid, torch.float32),
            scalars=t(obs.scalars, torch.float32),
            messages=self.actor.symbols_from_messages(t(obs.messages, torch.long)),
            message_mask=t(obs.message_mask, torch.bool),
            action_mask=t(obs.action_mask, torch.bool),
        )

    def __call__(self, obs: EnvObs) -> EnvAction:
        state = self.hidden.get(obs.dragon_id)
        if state is None:
            state = self.actor.initial_state(1, self.device)
        tensors = self.to_tensors(obs)
        joint, log_prob, entropy, new_state = self.actor.act(
            tensors, state, deterministic=self.deterministic,
            with_entropy=self.on_transition is not None and self.record_entropy)
        self.hidden[obs.dragon_id] = new_state

        send = bool(joint.transmit[0]) and self.transmit
        payload = int(self.actor.payload(joint)[0]) if send else None

        if self.on_transition is not None:
            self.on_transition({
                "dragon_id": obs.dragon_id, "team": obs.team, "round": obs.round,
                "obs": tensors, "hidden": state, "joint": joint,
                "log_prob": float(log_prob[0]),
                "entropy": None if entropy is None else float(entropy[0]),
            })
        return EnvAction(int(joint.action[0]), sonar=payload)
