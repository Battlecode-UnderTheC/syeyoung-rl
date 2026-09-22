"""Export the spatial actor's TorchScript entry points for the native trainer."""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from snake_rl.env.schema import check_schema
from snake_rl.model import EventCritic, SpatialActor, SpatialActorConfig, export_spatial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="native")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--sonar", action="store_true")
    args = parser.parse_args()
    actor = SpatialActor(SpatialActorConfig(sonar=args.sonar))
    critic = EventCritic()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        check_schema(state["schema_version"], state.get("schema_hash"))
        actor.load_state_dict(state["actor"])
        critic.load_state_dict(state["critic"])
    paths = export_spatial(actor, args.out, critic)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
