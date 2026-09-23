"""Ship the native trainer's log to wandb.

The trainer prints one `iter N: ...` line per iteration plus `learn:` and
`outcomes:` detail lines, and never blocks on the network; this tails the log
so a logging hiccup can never stall training.

    python tools/wandb_sidecar.py --log runs/native-v1.log --name native-v1
"""
from __future__ import annotations

import argparse
import pathlib
import re
import time

ITER = re.compile(r"^iter (\d+): (\d+) decisions, collect ([\d.]+)s \(([\d.]+)/s\), learn ([\d.]+)s "
                  r"\((\d+) scored, (\d+) chunks, ([\d.]+) team-rounds\), kl ([\d.]+) clip ([\d.]+) "
                  r"policy ([-\d.]+) value ([-\d.]+) entropy ([-\d.]+) epochs (\d+) replay-parity ([\d.e+-]+)")
LEARN = re.compile(r"learn: (\d+) microbatches on \d+ gpus, (\d+) steps; prep [\d.]+s, per-learner pack ([\d.]+)s gpu ([\d.]+)s value ([\d.]+)s backward\+step ([\d.]+)s")
OUTCOMES = re.compile(r"outcomes: (\d+) events, (\d+) terminal \(A (\d+), B (\d+), draw (\d+)\), mean \|return\| ([\d.]+), "
                      r"split fraction ([\d.]+), mean move steps ([\d.]+)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--project", default="snake-marl")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--name", default="native-spatial-v1")
    parser.add_argument("--id", default=None, help="resume this wandb run id")
    args = parser.parse_args()

    import wandb
    run = wandb.init(project=args.project, entity=args.entity, name=args.name, id=args.id, resume="allow" if args.id else None)
    path = pathlib.Path(args.log)
    while not path.exists():
        time.sleep(2.0)
    row: dict = {}
    step = 0
    total = 0
    started = time.time()
    with path.open() as stream:
        while True:
            line = stream.readline()
            if not line:
                time.sleep(2.0)
                continue
            m = ITER.match(line)
            if m:
                if row:
                    run.log(row, step=step)
                    step += 1
                v = m.groups()
                total += int(v[1])
                row = {"iteration": int(v[0]), "decisions": int(v[1]), "time/collect_s": float(v[2]),
                       "throughput/collect_per_s": float(v[3]), "time/learn_s": float(v[4]),
                       "scored": int(v[5]), "chunks": int(v[6]), "team_rounds": float(v[7]),
                       "ppo/kl": float(v[8]), "ppo/clip_fraction": float(v[9]), "loss/policy": float(v[10]),
                       "loss/value": float(v[11]), "loss/entropy": float(v[12]), "ppo/epochs": int(v[13]),
                       "ppo/replay_parity": float(v[14]), "decisions_total": total,
                       "throughput/end_to_end_per_s": int(v[1]) / max(float(v[2]) + float(v[4]), 1e-6),
                       "time/elapsed_min": (time.time() - started) / 60.0}
                continue
            m = LEARN.search(line)
            if m and row:
                v = m.groups()
                row.update({"learn/microbatches": int(v[0]), "learn/steps": int(v[1]), "learn/pack_s": float(v[2]),
                            "learn/gpu_s": float(v[3]), "learn/value_s": float(v[4]), "learn/backward_s": float(v[5])})
                continue
            m = OUTCOMES.search(line)
            if m and row:
                v = m.groups()
                terminal = int(v[1])
                row.update({"outcome/events": int(v[0]), "outcome/terminal": terminal,
                            "outcome/win_a": int(v[2]) / max(terminal, 1), "outcome/win_b": int(v[3]) / max(terminal, 1),
                            "outcome/draw": int(v[4]) / max(terminal, 1), "outcome/mean_abs_return": float(v[5]),
                            "behaviour/split_fraction": float(v[6]), "behaviour/mean_move_steps": float(v[7]),
                            "outcome/events_per_terminal": int(v[0]) / max(terminal, 1)})


if __name__ == "__main__":
    main()
