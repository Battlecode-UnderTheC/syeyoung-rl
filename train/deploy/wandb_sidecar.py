"""Streams a run's metrics.jsonl to Weights & Biases (a separate process, so the
trainer never depends on the network). Resumes the same wandb run by id.

    python deploy/wandb_sidecar.py runs/seed0
"""

import json
import pathlib
import sys
import time

import wandb

run_dir = pathlib.Path(sys.argv[1])
id_file = run_dir / "wandb_id"
if not id_file.exists():
    id_file.write_text(wandb.util.generate_id() if hasattr(wandb.util, "generate_id") else __import__("secrets").token_hex(4))
run = wandb.init(project="dragons-competitive-v1", name=run_dir.name, id=id_file.read_text().strip(), resume="allow")
seen = int(run.summary.get("update", -1)) if run.summary else -1
metrics = run_dir / "metrics.jsonl"
while not metrics.exists():
    time.sleep(10)
with open(metrics) as f:
    while True:
        at = f.tell()
        line = f.readline()
        if not line.endswith("\n"):          # nothing new, or a row still being written
            f.seek(at)
            time.sleep(10)
            continue
        r = json.loads(line)
        if r["update"] <= seen:
            continue
        flat = {}
        for k, v in r.items():
            if isinstance(v, list):
                for i, x in enumerate(v):
                    flat[f"{k}/{i}"] = x
            elif isinstance(v, dict):
                for kk, x in v.items():
                    if isinstance(x, (int, float, bool)):
                        flat[f"{k}/{kk}"] = float(x)
            elif isinstance(v, (int, float, bool)):
                flat[k] = float(v)
        flat["score"] = (r["win"] + 0.5 * r["draw"]) / max(1, r["win"] + r["draw"] + r["loss"])
        run.log(flat, step=r["update"], commit=True)   # commit now, not when the next update arrives
        seen = r["update"]
