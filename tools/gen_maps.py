"""Write a maps file for the native trainer: map texts separated by '---'.

Every thread of the trainer takes one board size, so maps are generated per
size. UNIT_LIMIT is capped so two teams can never exceed the trainer's slot
count; everything else follows the spec 11 curriculum: varied kelp, portals,
starting lengths, dragons per team and pearl gaps.
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "dragons-env"))

from dragons_env import mapgen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="native/maps.txt")
    parser.add_argument("--sizes", default="11,13,15,19")
    parser.add_argument("--per-size", type=int, default=64)
    parser.add_argument("--max-slots", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    texts = []
    for size in map(int, args.sizes.split(",")):
        for _ in range(args.per_size):
            spec = mapgen.random_map(
                rng, min_size=size, max_size=size,
                dragons_per_team=rng.choice((1, 1, 2, 3)),
                length=rng.choice((3, 4, 6, 8)),
                kelp_density=rng.uniform(0.0, 0.08),
                portal_pairs=rng.choice((0, 0, 1, 2)),
                pearl_gap_choices=((4, 12), (8, 20), (15, 40), (20, 60)))
            spec.unit_limit = args.max_slots // 2
            texts.append(spec.to_text())
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("---\n".join(texts))
    print(f"wrote {len(texts)} maps to {out}")


if __name__ == "__main__":
    main()
