#!/usr/bin/env bash
# Workstream 2: (re)generate surfel splats for mesh-only rigid objects.
#   bash ~/spais-demo/work/polaris_wsl/mesh_to_surfels_ws2.sh [--envs food_bussing ...] [--objects bowl ...]
# Builds the PolaRiS-Hub-wsl mirror first if needed. CPU only (no GPU use).
set -eo pipefail
source ~/spais-demo/work/polaris_env.sh >/dev/null
[ -f ~/spais-demo/polaris/PolaRiS-Hub-wsl/food_bussing/scene.usda ] || bash ~/spais-demo/work/polaris_wsl/make_hub_wsl_ws2.sh
# Isaac Sim ships pxr as an extension; expose it without launching Kit.
U=$(ls -d ~/spais-demo/polaris/.venv/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-* | head -1)
PYLIB=$(dirname "$(readlink -f ~/spais-demo/polaris/.venv/bin/python)")/../lib
export PATH=$HOME/spais-demo/polaris/.venv/bin:$PATH
export PYTHONPATH=$U:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$U/bin:$PYLIB:${LD_LIBRARY_PATH:-}
mkdir -p ~/spais-demo/logs
ulimit -v 8000000   # 8 GB address-space cap: WSL RAM is shared and tight; fail, don't OOM the VM
.venv/bin/python ~/spais-demo/work/polaris_wsl/mesh_to_surfels.py "$@" 2>&1 | grep -v "^Warning\|Coding Error" | tee ~/spais-demo/logs/mesh_to_surfels_ws2.log
