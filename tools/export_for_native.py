"""Write the TorchScript actor and a maps file the native rollout driver reads."""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from snake_rl.env import ActionSpace
from snake_rl.model import Actor, export
from snake_rl.training.config import TrainConfig
from snake_rl.training.train import sample_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="native")
    parser.add_argument("--arenas", type=int, default=512)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = TrainConfig()
    space = ActionSpace(max_sprint=config.max_sprint, split_sizes=config.split_sizes)

    actor = Actor(space.n, hidden=config.hidden, enable_sonar=config.enable_sonar)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        actor.load_state_dict(state["actor"])
        print(f"loaded actor from {args.checkpoint}")
    export(actor, out / "actor.ts")

    # The native driver must build the identical action space, or the mask and
    # the policy head disagree.
    (out / "space.txt").write_text(
        f"max_sprint {config.max_sprint}\n"
        f"split_sizes {','.join(str(n) for n in config.split_sizes)}\n"
        f"hidden {config.hidden}\n"
        f"actions {space.n}\n")

    rng = random.Random(args.seed)
    maps = [sample_map(config, rng).to_text() for _ in range(args.arenas)]
    (out / "maps.txt").write_text("---\n".join(maps))
    print(f"wrote {out/'actor.ts'}, {args.arenas} maps, and space.txt "
          f"({space.n} actions, hidden {config.hidden})")


if __name__ == "__main__":
    main()
