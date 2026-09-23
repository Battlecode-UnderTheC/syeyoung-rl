"""Ship a run's metrics.jsonl to wandb.

The trainer appends one JSON object per iteration and never touches the
network; this tails the file (from the start, so it backfills) and logs each
row at step = iteration, so a wandb hiccup can never stall training. The wandb
run id is kept in <run>/wandb_id, so restarting the sidecar continues the
same wandb run.

    python tools/wandb_sidecar.py --run runs/v2-selfplay [--project snake-marl]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import secrets
import time

# metrics.jsonl key -> wandb key (anything not listed goes under misc/)
GROUPS = {
    "turns": "throughput/turns_total", "elapsed": "time/elapsed_s",
    "rollout_seconds": "time/rollout_s", "update_seconds": "time/update_s",
    "games": "games/count", "win_a": "games/win_a", "win_b": "games/win_b", "draw": "games/draw",
    "game_rounds": "games/rounds", "eliminations": "games/elimination_rate",
    "splits": "behaviour/splits", "sprints": "behaviour/sprints",
    "death_wall": "deaths/wall", "death_self": "deaths/self", "death_body": "deaths/body",
    "death_head": "deaths/head_to_head", "death_action": "deaths/no_valid_action",
    "policy_loss": "loss/policy", "value_loss": "loss/value", "entropy": "loss/entropy",
    "approx_kl": "ppo/approx_kl", "clip_fraction": "ppo/clip_fraction",
    "actor_grad_norm": "ppo/actor_grad_norm", "critic_grad_norm": "ppo/critic_grad_norm",
    "explained_variance": "ppo/explained_variance", "replay_kl": "ppo/replay_kl",
    "replay_clip": "ppo/replay_clip", "epochs": "ppo/epochs",
    "max_slots": "pool/slots_used", "overflow": "pool/overflow_turns",
}


def row_for(d: dict, turns_per_iteration: float) -> dict:
    row = {GROUPS.get(k, "misc/" + k): v for k, v in d.items() if k != "iteration"}
    seconds = d.get("rollout_seconds", 0) + d.get("update_seconds", 0)
    if seconds > 0:
        row["throughput/turns_per_s"] = turns_per_iteration / seconds
    if turns_per_iteration > 0:
        row["pool/overflow_fraction"] = d.get("overflow", 0) / turns_per_iteration
    deaths = sum(d.get(k, 0) for k in ("death_wall", "death_self", "death_body", "death_head", "death_action"))
    for k in ("death_wall", "death_self", "death_body", "death_head", "death_action"):
        row[GROUPS[k] + "_share"] = d.get(k, 0) / max(deaths, 1)
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="the trainer's --out directory")
    p.add_argument("--project", default="snake-marl")
    p.add_argument("--entity", default=None)
    p.add_argument("--name", default=None, help="default: the run directory's name")
    args = p.parse_args()

    import wandb

    run_dir = pathlib.Path(args.run)
    id_file = run_dir / "wandb_id"
    run_id = id_file.read_text().strip() if id_file.exists() else secrets.token_hex(4)
    id_file.write_text(run_id + "\n")
    commit = (run_dir / "COMMIT").read_text().strip() if (run_dir / "COMMIT").exists() else None
    run = wandb.init(project=args.project, entity=args.entity, name=args.name or run_dir.name, id=run_id,
                     resume="allow", config={"run_dir": str(run_dir.resolve()), "commit": commit})
    last = run.summary.get("iteration", -1)     # already logged (the sidecar restarted)
    path = run_dir / "metrics.jsonl"
    while not path.exists():
        time.sleep(5.0)
    prev_turns = None
    with path.open() as stream:
        while True:
            line = stream.readline()
            if not line.endswith("\n"):          # nothing new, or a line still being written
                if line:
                    stream.seek(stream.tell() - len(line))
                time.sleep(5.0)
                continue
            d = json.loads(line)
            it = d["iteration"]
            per_it = d["turns"] - prev_turns if prev_turns is not None and d["turns"] > prev_turns else d["turns"] / (it + 1)
            prev_turns = d["turns"]
            if it <= last:
                continue
            row = row_for(d, per_it)
            row["iteration"] = it
            run.log(row, step=it)
            last = it


if __name__ == "__main__":
    main()
