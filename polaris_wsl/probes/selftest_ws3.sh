#!/bin/bash
# Offline self-test of probe_polaris.py (no Isaac, no server) + CLI parse check.
source ~/spais-demo/work/polaris_env.sh
W=~/spais-demo/work/polaris_wsl
uv run python $W/probes/test_probe_ws3.py 2>&1 | grep -v "tool.uv.dev-dependencies"
uv run python $W/probe_polaris.py run --help 2>&1 | head -60
uv run python $W/probe_polaris.py batch --help 2>&1 | grep -E "episodes|fixed-ic|all-variants"
