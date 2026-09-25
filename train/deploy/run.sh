#!/bin/sh
# Started by dragons-rl@<name>.service: the throughput trainer, resumed from its
# latest checkpoint on every start.
set -e
cd "$(dirname "$0")/.."
NAME=${1:?run name}
echo "=== start $(date -Is) run $NAME commit $(cat COMMIT 2>/dev/null)"
# no GPU kernel autotuning: under memory pressure it picked unvalidated kernels
# that produced NaNs in blocks of rows (and it slows every compile)
# many threads allocate and free: without a cap glibc keeps freed memory in per-thread arenas
export MALLOC_ARENA_MAX=4
export XLA_FLAGS="--xla_gpu_autotune_level=0 ${XLA_FLAGS:-}"
exec .venv/bin/python -u -m dragons_rl.train --run "runs/$NAME" --slots 512 --threads 24
