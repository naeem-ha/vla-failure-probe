#!/bin/bash
# Run one probe definition over N episodes (all variants by default) and write a CSV summary.
# Usage (inside WSL): bash ~/spais-demo/work/polaris_wsl/probes/run_probe_set.sh <probe.yaml> [episodes] [extra batch args...]
#   bash run_probe_set.sh food_bussing_battery.yaml 5
#   bash run_probe_set.sh food_bussing_battery.yaml 10 --variant stock      (one variant only)
# Needs the policy server on :8000 (bash ~/spais-demo/work/serve_droid.sh > ~/spais-demo/server.log 2>&1 &).
probe=$1; n=${2:-5}
shift; [ $# -gt 0 ] && shift
here=$(cd "$(dirname "$0")" && pwd)
case "$probe" in
  /*) ;;
  *) probe="$here/$probe" ;;
esac
all="--all-variants"
for a in "$@"; do [ "$a" = "--variant" ] && all=""; done
source ~/spais-demo/work/polaris_env.sh
mkdir -p ~/spais-demo/logs
log=~/spais-demo/logs/probe_$(basename "$probe" .yaml)_$(date +%m%d_%H%M).log
timeout "${PROBE_TIMEOUT:-14400}" uv run python "$here/../probe_polaris.py" batch \
  --probe "$probe" --episodes "$n" $all "$@" > "$log" 2>&1
echo "exit: $?"
grep -E "^\[probe\]|Traceback|Error:" "$log" | tail -40
echo "log: $log"
