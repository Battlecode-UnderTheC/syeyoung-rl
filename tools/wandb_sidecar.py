"""Ship the native trainer's metrics.jsonl to wandb.

The trainer writes one JSON object per iteration and never blocks on the
network; this tails that file so a logging hiccup can never stall training.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--project", default="snake-marl")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--name", default="native-phase2")
    parser.add_argument("--config", default="{}")
    args = parser.parse_args()

    import wandb
    run = wandb.init(project=args.project, entity=args.entity, name=args.name,
                     config=json.loads(args.config))

    path = pathlib.Path(args.metrics)
    while not path.exists():
        time.sleep(2.0)

    # The trainer's iteration counter restarts from zero every time the
    # supervisor restarts it. wandb drops any step that is not greater than the
    # last one, so using it as the step silently discards everything after a
    # restart -- the chart freezes while training carries on fine. Count rows
    # instead, and keep the trainer's own counter as an ordinary metric.
    step = 0
    total_turns = 0
    restarts = 0
    previous_iteration = -1
    with path.open() as stream:
        while True:
            line = stream.readline()
            if not line:
                time.sleep(2.0)
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            iteration = int(row.get("iteration", 0))
            if iteration < previous_iteration:
                restarts += 1
            previous_iteration = iteration
            total_turns += int(row.get("turns", 0))
            row["turns_total"] = total_turns
            row["trainer_iteration"] = iteration
            row["restarts"] = restarts
            run.log(row, step=step)
            step += 1


if __name__ == "__main__":
    main()
