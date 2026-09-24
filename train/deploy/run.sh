#!/bin/sh
# Started by dragons-rl@<seed>.service: the reference run for one training
# seed, resumed from its latest completed-update checkpoint on every start.
set -e
cd "$(dirname "$0")/.."
SEED=${1:?seed}
RUN=runs/seed$SEED
echo "=== start $(date -Is) seed $SEED commit $(git rev-parse --short HEAD 2>/dev/null || cat COMMIT 2>/dev/null)"
exec .venv/bin/python -u -m dragons_rl.train --run "$RUN" --seed "$SEED" --fixtures ../fixtures.json --threads 24
