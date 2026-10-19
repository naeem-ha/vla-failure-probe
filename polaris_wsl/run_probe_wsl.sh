#!/bin/bash
# Run one PolaRiS probe episode fully in WSL: splat-only env, all objects as surfels.
# Start the policy server first (bash ~/spais-demo/work/serve_droid.sh) and wait for port 8000.
# Usage: bash run_probe_wsl.sh <probe.yaml> <variant> [ic] [extra probe_polaris.py run args...]
source ~/spais-demo/work/polaris_env.sh
export POLARIS_DATA_PATH=$HOME/spais-demo/polaris/PolaRiS-Hub-wsl
export PATH=$HOME/spais-demo/polaris/.venv/bin:$PATH   # ninja for the splat extension JIT
probe=$1; variant=$2; ic=${3:-0}; shift 3 2>/dev/null || shift $#
mkdir -p ~/spais-demo/logs
log=~/spais-demo/logs/probe_$(basename "$probe" .yaml)_${variant}_ic${ic}.log
timeout 1800 uv run python ~/spais-demo/work/polaris_wsl/probe_polaris.py run \
  --probe ~/spais-demo/work/polaris_wsl/probes/"$probe" --variant "$variant" --ic "$ic" "$@" > "$log" 2>&1
echo "EXIT=$?  log=$log"
grep -v -E "omni\.|neuraylib|libXt|libGLU|Warning|PhysX warning|^\s*$" "$log" | tail -25
