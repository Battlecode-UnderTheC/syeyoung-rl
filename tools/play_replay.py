"""Play a trained policy against itself on the official engine and write a
replay the toolkit's viewer opens.

    python tools/play_replay.py --checkpoint runs/native-v12 --map maps/default_small.map \
        --out runs/replays/v12-default_small.replay

The checkpoint is the trainer's TorchScript export (spatial_encoder.ts,
spatial_decoder.ts); its weights are loaded into the reference Python actor,
which runs one private lifetime state per dragon exactly as in training and
answers every turn the engine asks for. Both teams use the same policy.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dragons-env"))

from dragons_env import WasmEngine, parse_spawn, parse_turn  # noqa: E402
from dragons_env.engine import DEBUG_ALL  # noqa: E402
from snake_rl.env.actions import TOP_MOVE, CommandCodec  # noqa: E402
from snake_rl.env.memory import LifetimeKey, LifetimeState  # noqa: E402
from snake_rl.env.observation import prepare  # noqa: E402
from snake_rl.model import SpatialActor, SpatialActorConfig  # noqa: E402
from snake_rl.model.actor import SpatialInputs  # noqa: E402


def load_actor(checkpoint: pathlib.Path, sonar: bool) -> SpatialActor:
    actor = SpatialActor(SpatialActorConfig(sonar=sonar))
    state = {}
    for name in ("spatial_encoder.ts", "spatial_decoder.ts"):
        module = torch.jit.load(str(checkpoint / name), map_location="cpu")
        state.update(module.state_dict())
    missing, unexpected = actor.load_state_dict(state, strict=False)
    # auxiliary heads train the representation only and are not exported
    missing = [k for k in missing if not k.startswith(("inbox_enc", "sonar_head", "aux_"))]
    unexpected = [k for k in unexpected if not k.startswith(("inbox_enc", "sonar_head"))]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint does not match the actor: missing {missing}, unexpected {unexpected}")
    actor.eval()
    return actor


class Policy:
    """One policy instance serving every dragon of a match."""

    def __init__(self, actor: SpatialActor, codec: CommandCodec, deterministic: bool):
        self.actor = actor
        self.codec = codec
        self.deterministic = deterministic
        self.states: dict[int, LifetimeState] = {}
        self.memory: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.turns = 0

    def spawn(self, dragon_id: int, block: bytes) -> None:
        info = parse_spawn(block.decode())
        self.states[dragon_id] = LifetimeState(LifetimeKey(0, info.team, info.dragon_id),
                                               info.width, info.height, info.unit_limit)
        self.memory[dragon_id] = self.actor.initial_state(1, info.height, info.width)

    @torch.no_grad()
    def reply(self, dragon_id: int, block: bytes) -> bytes:
        st = self.states[dragon_id]
        turn = parse_turn(block.decode(), st.width, st.height)
        obs = prepare(st, turn, self.codec.max_steps)
        x = SpatialInputs.from_prepared([obs], self.codec)
        M, h = self.memory[dragon_id]
        ctx, M_new, h_new = self.actor.encode_and_update(x, M, h)
        cmds, _, _ = self.actor.decode_sample(ctx, x, [obs], [st], self.codec, self.deterministic)
        cmd = cmds[0]
        if cmd.top == TOP_MOVE:
            st.commit_command(turn, "move", cmd.dirs)
        else:
            st.commit_command(turn, "split", split_n=cmd.split_n)
        self.memory[dragon_id] = (M_new, h_new)
        self.turns += 1
        return (cmd.text + "\n").encode()   # the engine needs the terminator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="directory with spatial_encoder.ts and spatial_decoder.ts")
    parser.add_argument("--map", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", default="policy")
    parser.add_argument("--sonar", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    actor = load_actor(pathlib.Path(args.checkpoint), args.sonar)
    policy = Policy(actor, CommandCodec(max_steps=8), args.deterministic)
    map_bytes = pathlib.Path(args.map).read_bytes()
    deaths: list[tuple[int, int, str]] = []
    engine = WasmEngine()
    started = time.time()
    result = engine.run(map_bytes, policy.reply, on_spawn=policy.spawn,
                        on_death=lambda dragon_id, round_num, reason: deaths.append((dragon_id, round_num, reason)),
                        debug=DEBUG_ALL)
    elapsed = time.time() - started
    blob = engine.replay(args.name, args.name)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    print(f"{args.map}: {result.rounds} rounds, winner {result.winner}, end reason {result.end_reason} "
          f"(0 elimination, 1 on length), A {result.a_dragons} dragons/{result.a_length} length, "
          f"B {result.b_dragons}/{result.b_length}, {len(deaths)} deaths, {policy.turns} turns in {elapsed:.0f}s")
    print(f"wrote {out} ({len(blob)} bytes)")


if __name__ == "__main__":
    main()
