#!/bin/bash
# Usage: bash polaris_eval.sh <Environment> <run_folder> [rollouts]
source ~/spais-demo/work/polaris_env.sh
uv run scripts/eval.py --environment "$1" --policy.port 8000 --run-folder "$2" --rollouts "${3:-1}" 2>&1 \
  | grep -v -E "^\s*$|\[Warning\]|warn" | tail -40
