"""Play one match and record everything the learner needs.

The engine owns the loop, so collection happens inside the policy callback.
Per dragon-step we keep the observation it acted on, the hidden state it acted
*from*, the action, and the log-prob under the policy that chose it. Per round
we keep one critic state and one reward entry per team -- never one per dragon.

Observations are stored as uint8 (every channel is a flag except the pearl
timer, which is quantised) to keep a match's worth of data in single-digit MB.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..env.actions import ActionSpace
from ..env.global_state import GlobalStateTracker, N_DRAGON_FEATURES, N_MAP_CHANNELS
from dragons_env.mapgen import MapSpec
from ..env.match import Action as EnvAction
from ..env.match import MatchRunner
from ..model.actor import Actor, ActorObs

TEAM_INDEX = {"A": 0, "B": 1}


@dataclass
class Trajectory:
    grid: np.ndarray
    scalars: np.ndarray
    messages: np.ndarray
    message_mask: np.ndarray
    action_mask: np.ndarray
    hidden: np.ndarray
    action: np.ndarray
    transmit: np.ndarray
    symbols: np.ndarray
    log_prob: np.ndarray
    dragon: np.ndarray
    team: np.ndarray
    round: np.ndarray
    learner: np.ndarray
    map_grid: np.ndarray
    dragon_feats: np.ndarray
    dragon_team: np.ndarray
    dragon_mask: np.ndarray
    rounds: np.ndarray
    reward: np.ndarray
    info: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.action)


class _Recorder:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, **row) -> None:
        self.rows.append(row)

    def stack(self, key, dtype):
        if not self.rows:
            return np.zeros((0,), dtype=dtype)
        return np.stack([r[key] for r in self.rows]).astype(dtype, copy=False)

    def column(self, key, dtype):
        return np.array([r[key] for r in self.rows], dtype=dtype)


def play_match(runner: MatchRunner, spec: MapSpec, actors: dict[str, object],
               learner_teams: set[str], device: str = "cpu",
               max_dragons: int = 32, deterministic: bool = False) -> Trajectory:
    """Run one match. `actors` maps team -> Actor or scripted callable."""
    tracker = GlobalStateTracker(spec, max_dragons=max_dragons)
    recorder = _Recorder()
    hidden: dict[int, torch.Tensor] = {}
    rounds: list[dict] = []
    current_round = -1

    def snapshot() -> None:
        feats, teams, mask = tracker.dragon_tensor()
        rounds.append({"map": tracker.map_tensor().astype(np.float16),
                       "feats": feats, "teams": teams, "mask": mask,
                       "round": tracker.round})

    def policy(obs):
        nonlocal current_round
        if obs.round != current_round:
            if current_round >= 0:
                snapshot()
            current_round = obs.round
            tracker.begin_round(obs.round)
        tracker.observe(obs.dragon_id, obs.team, obs.block)

        agent = actors[obs.team]
        if not isinstance(agent, Actor):
            return agent(obs)                       # a scripted opponent

        state = hidden.get(obs.dragon_id)
        if state is None:
            state = agent.initial_state(1, device)
        tensors = ActorObs(
            grid=torch.as_tensor(obs.grid, device=device).unsqueeze(0),
            scalars=torch.as_tensor(obs.scalars, device=device).unsqueeze(0),
            messages=agent.symbols_from_messages(
                torch.as_tensor(obs.messages.astype(np.int64), device=device).unsqueeze(0)),
            message_mask=torch.as_tensor(obs.message_mask, device=device).unsqueeze(0),
            action_mask=torch.as_tensor(obs.action_mask, device=device).unsqueeze(0),
        )
        joint, log_prob, _, new_state = agent.act(tensors, state,
                                                  deterministic=deterministic)
        if obs.team in learner_teams:
            recorder.add(
                grid=np.clip(obs.grid * 255.0, 0, 255).astype(np.uint8),
                scalars=obs.scalars,
                messages=obs.messages,
                message_mask=obs.message_mask,
                action_mask=obs.action_mask,
                hidden=state.squeeze(0).cpu().numpy().astype(np.float16),
                action=int(joint.action[0]),
                transmit=bool(joint.transmit[0]),
                symbols=joint.symbols[0].cpu().numpy().astype(np.int64),
                log_prob=float(log_prob[0]),
                dragon=obs.dragon_id,
                team=TEAM_INDEX[obs.team],
                round=obs.round,
            )
        hidden[obs.dragon_id] = new_state
        send = bool(joint.transmit[0])
        payload = int(agent.payload(joint)[0]) if send else None
        return EnvAction(int(joint.action[0]), sonar=payload)

    def died(dragon_id: int, round_num: int, reason: str) -> None:
        tracker.kill(dragon_id)
        hidden.pop(dragon_id, None)

    outcome = runner.run(spec.to_bytes(), policy, on_death=died)
    snapshot()

    winner = outcome.result.winner
    reward = np.zeros(2, dtype=np.float32)
    if winner is not None:
        reward[TEAM_INDEX[winner]] = 1.0
        reward[1 - TEAM_INDEX[winner]] = -1.0

    # against a scripted opponent the outcome is an actual skill signal; in
    # self-play it is not, since both sides are the same weights
    self_play = len(learner_teams) == 2
    learner_result = None
    if not self_play:
        team = next(iter(learner_teams))
        learner_result = 0.0 if winner is None else (1.0 if winner == team else -1.0)

    return Trajectory(
        grid=recorder.stack("grid", np.uint8),
        scalars=recorder.stack("scalars", np.float32),
        messages=recorder.stack("messages", np.uint8),
        message_mask=recorder.stack("message_mask", bool),
        action_mask=recorder.stack("action_mask", bool),
        hidden=recorder.stack("hidden", np.float16),
        action=recorder.column("action", np.int64),
        transmit=recorder.column("transmit", bool),
        symbols=recorder.stack("symbols", np.int64),
        log_prob=recorder.column("log_prob", np.float32),
        dragon=recorder.column("dragon", np.int64),
        team=recorder.column("team", np.int64),
        round=recorder.column("round", np.int64),
        learner=np.ones(len(recorder.rows), dtype=bool),
        map_grid=np.stack([r["map"] for r in rounds]),
        dragon_feats=np.stack([r["feats"] for r in rounds]),
        dragon_team=np.stack([r["teams"] for r in rounds]),
        dragon_mask=np.stack([r["mask"] for r in rounds]),
        rounds=np.array([r["round"] for r in rounds], dtype=np.int64),
        reward=reward,
        info={
            "winner": winner,
            "rounds": outcome.result.rounds + 1,
            "turns": outcome.turns,
            "deaths": outcome.death_reasons(),
            "a_length": outcome.result.a_length,
            "b_length": outcome.result.b_length,
            "a_dragons": outcome.result.a_dragons,
            "b_dragons": outcome.result.b_dragons,
            "end_reason": outcome.result.end_reason,
            "self_play": self_play,
            "learner_result": learner_result,
        },
    )
