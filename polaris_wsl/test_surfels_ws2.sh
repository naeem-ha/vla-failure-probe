#!/usr/bin/env bash
# Workstream 2: render generated object splats with PolaRiS's SplatRenderer vs a
# software mesh rasterizer; writes ~/spais-demo/work/runs/ws2/*.png. Small GPU use.
#   bash ~/spais-demo/work/polaris_wsl/test_surfels_ws2.sh [--envs food_bussing] [--scene-only|--no-scene]
set -eo pipefail
source ~/spais-demo/work/polaris_env.sh >/dev/null
U=$(ls -d ~/spais-demo/polaris/.venv/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-* | head -1)
PYLIB=$(dirname "$(readlink -f ~/spais-demo/polaris/.venv/bin/python)")/../lib
export PATH=$HOME/spais-demo/polaris/.venv/bin:$PATH
export PYTHONPATH=$U:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$U/bin:$PYLIB:${LD_LIBRARY_PATH:-}
mkdir -p ~/spais-demo/logs ~/spais-demo/work/runs/ws2
timeout 1800 .venv/bin/python ~/spais-demo/work/polaris_wsl/test_surfels_ws2.py "$@" 2>&1 | grep -v "^Warning\|Coding Error" | tee ~/spais-demo/logs/test_surfels_ws2.log
