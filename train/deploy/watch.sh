#!/bin/bash
# Event stream for watching dragons-train from afar (e.g. `ssh host ~/rl2/train/deploy/watch.sh`):
# every 10th iteration line, trainer errors and crashes as they are logged, service
# state changes and restarts, and low host memory (checked every minute).
LOG=${LOG:-$HOME/rl2/train/runs/train.log}
tail -n0 -F "$LOG" 2>/dev/null | grep --line-buffered -aE '^it [0-9]*0 |^error|=== start|RESOURCE_EXHAUSTED|Segmentation|Aborted|terminate called|core dumped|Killed' \
  | cut -c1-520 &
prev_state=""; prev_restarts=""; low=0
while true; do
  state=$(systemctl --user show dragons-train -p ActiveState --value)
  restarts=$(systemctl --user show dragons-train -p NRestarts --value)
  [ "$state" != "$prev_state" ] && echo "SERVICE STATE: $state (restarts $restarts)"
  [ -n "$prev_restarts" ] && [ "$restarts" != "$prev_restarts" ] && echo "SERVICE RESTARTED: NRestarts $prev_restarts -> $restarts"
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  if [ "$avail" -lt 10 ]; then [ $low -eq 0 ] && echo "LOW HOST MEMORY: ${avail} GiB available"; low=1; else low=0; fi
  prev_state=$state; prev_restarts=$restarts
  sleep 60
done
