#!/bin/bash
# Repeat one probe over several seeds and print a one-line result per seed.
# Usage: bash run_seeds.sh <tag> <seed>... -- [probe.py args...]
#   bash run_seeds.sh stop15 1 2 3 4 5 -- --switch-prompt stop --switch-step 15 --continue-after-done
tag=$1; shift
seeds=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do seeds+=("$1"); shift; done
shift
for s in "${seeds[@]}"; do
  bash ~/spais-demo/work/run_probe.sh "$@" --seed "$s" --tag "${tag}_seed$s" > /dev/null
  python3 - "$tag" "$s" <<'EOF'
import json, sys
tag, s = sys.argv[1], sys.argv[2]
d = json.load(open(f"runs/libero_spatial_t0_e0_{tag}_seed{s}.json"))["summary"]
print(f"seed {s}: completed={d['task_completed']} at step {d['task_completed_step']}, "
      f"moved {d.get('eef_path_after_switch_m', 0):.2f} m after switch")
EOF
done
