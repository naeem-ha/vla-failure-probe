#!/bin/bash
# Usage: bash run_probe.sh [probe.py args...]
source ~/spais-demo/work/client_env.sh
python probe.py "$@" 2>&1 | grep -v -i -E "warn|np.MachAr|^\s*$"
