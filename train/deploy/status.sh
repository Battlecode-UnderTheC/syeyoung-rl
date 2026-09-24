#!/bin/sh
# One-screen status of a run: service state, last log lines, last updates.
SEED=${1:-0}
cd "$(dirname "$0")/.."
systemctl --user --no-pager -o cat status dragons-rl@$SEED | sed -n 1,4p
grep -v "^W0\|WARNING\|cuda_timer" runs/seed$SEED.log | tail -5
.venv/bin/python - "$SEED" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(f"runs/seed{sys.argv[1]}/metrics.jsonl")] if True else []
for r in rows[-5:]:
    print(f"u{r['update']:5d} W/D/L {r['win']:2d}/{r['draw']:2d}/{r['loss']:2d} rounds {r['mean_rounds']:6.1f} "
          f"dec {r['decisions']:6d} kl {max(r['kl'] or [0]):.4f} Hp {r.get('h_phys',0):.3f} "
          f"vmse {r.get('value_mse_2',0):.3f} {r['collect_s']:.0f}+{r['train_s']:.0f}s")
PY
