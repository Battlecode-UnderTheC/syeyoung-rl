#!/bin/sh
# Started by dragons-rl@<name>.service: the throughput trainer, resumed from its
# latest checkpoint on every start.
set -e
cd "$(dirname "$0")/.."
NAME=${1:?run name}
echo "=== start $(date -Is) run $NAME commit $(cat COMMIT 2>/dev/null)"
exec .venv/bin/python -u -m dragons_rl.train --run "runs/$NAME" --slots 512 --threads 24
