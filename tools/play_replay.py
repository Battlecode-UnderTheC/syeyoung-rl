"""Play a trained policy through the official engine and write a .replay file.

The policy is the TorchScript module the native trainer saves, run here by the
same observation encoder the trainer used, against the real wasm engine. The
replay is produced by the engine itself, so it opens in the official viewer.
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from snake_rl.env import ActionSpace, MatchRunner, ObservationEncoder, WasmEngine, mapgen
from snake_rl.env.engine import DEBUG_ALL
from snake_rl.env.match import Action
from snake_rl.training.config import TrainConfig
from snake_rl.training.train import sample_map


class ScriptedPolicy:
    """Wraps the exported TorchScript actor, one hidden state per dragon."""

    def __init__(self, path: str, n_actions: int, hidden: int = 256,
                 deterministic: bool = False, seed: int = 0):
        self.module = torch.jit.load(path, map_location="cpu")
        self.module.eval()
        self.n_actions = n_actions
        self.hidden_size = hidden
        self.deterministic = deterministic
        self.generator = torch.Generator().manual_seed(seed)
        self.hidden: dict[int, torch.Tensor] = {}

    def on_death(self, dragon_id: int, round_num: int, reason: str) -> None:
        self.hidden.pop(dragon_id, None)

    @torch.no_grad()
    def __call__(self, obs) -> Action:
        state = self.hidden.get(obs.dragon_id)
        if state is None:
            state = torch.zeros(1, self.hidden_size)
        logits, _transmit, _symbols, new_state, _value = self.module(
            torch.from_numpy(obs.grid).unsqueeze(0),
            torch.from_numpy(obs.scalars).unsqueeze(0),
            torch.zeros(1, 8, 4, dtype=torch.long),
            torch.zeros(1, 8, dtype=torch.bool),
            torch.from_numpy(obs.action_mask).unsqueeze(0),
            state,
        )
        self.hidden[obs.dragon_id] = new_state
        if self.deterministic:
            action = int(logits.argmax(-1)[0])
        else:                                   # gumbel-max, as in training
            noise = torch.rand(logits.shape, generator=self.generator).clamp_min(1e-20)
            action = int((logits - torch.log(-torch.log(noise))).argmax(-1)[0])
        return Action(action)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/actor-latest.ts")
    parser.add_argument("--map", default="", help="a .map file; omitted means a generated board")
    parser.add_argument("--board", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    if args.map:
        map_bytes = pathlib.Path(args.map).read_bytes()
        label = pathlib.Path(args.map).stem
    else:
        # the same distribution training draws from, so a replay shows what the
        # policy actually practised on
        spec = sample_map(TrainConfig(), random.Random(args.seed))
        map_bytes = spec.to_bytes()
        label = f"gen{spec.width}x{spec.height}-seed{args.seed}"

    space = ActionSpace(max_sprint=3, split_sizes=())
    runner = MatchRunner(WasmEngine(), encoder=ObservationEncoder(ego_rotate=True),
                         space=space)
    policy = ScriptedPolicy(args.model, space.n, deterministic=args.deterministic,
                            seed=args.seed)

    started = time.perf_counter()
    outcome = runner.run(map_bytes, policy, on_death=policy.on_death, debug=DEBUG_ALL)
    elapsed = time.perf_counter() - started

    score = {team: (max((n for d, n in outcome.final_lengths.items()
                         if outcome.teams.get(d) == team), default=0),
                    sum(n for d, n in outcome.final_lengths.items()
                        if outcome.teams.get(d) == team))
             for team in ("A", "B")}
    print(f"map {label} | {outcome.result.rounds + 1} rounds | {outcome.turns} dragon-turns "
          f"| {elapsed:.1f}s")
    print(f"winner: {outcome.result.winner or 'draw'} | "
          f"A (max,total)={score['A']}  B (max,total)={score['B']}")
    print(f"deaths: {outcome.death_reasons() or 'none'}")

    out = pathlib.Path(args.out) if args.out else pathlib.Path(
        f"trained-vs-itself-on-{label}-{time.strftime('%Y-%m-%d-%H%M%S')}.replay")
    out.write_bytes(runner.engine.replay("trained", "trained"))
    print(f"wrote replay: {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
