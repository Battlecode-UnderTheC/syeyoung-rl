#!/bin/sh
# Started by deploy/dragons-train.service: trains into $RUN, resuming from its
# newest checkpoint. RUN and ARTIFACTS may be overridden in the environment.
set -e
cd "$(dirname "$0")/.."
RUN=${RUN:-runs/v2-selfplay}
ARTIFACTS=${ARTIFACTS:-artifacts/v3}
PLUGIN=.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda13/xla_cuda_plugin.so
mkdir -p "$RUN"
LATEST=$(ls "$RUN/ckpt" 2>/dev/null | sed -n 's/\.bin$//p' | sort -n | tail -1)
RESUME=""
[ -n "$LATEST" ] && RESUME="--resume $RUN/ckpt/$LATEST.bin"
echo "=== start $(date -Is) commit $(cat "$RUN/COMMIT" 2>/dev/null) resume '$LATEST'"
exec ./build/trainer --artifacts "$ARTIFACTS" --plugin "$PLUGIN" --out "$RUN" \
  --rollout-device 0 --learner-devices 1,2,3 --steps 256 --threads 20 --save-every 10 $RESUME
