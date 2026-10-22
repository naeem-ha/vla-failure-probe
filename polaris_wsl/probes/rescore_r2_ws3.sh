#!/bin/bash
# Rescore a run set with the current checks + probe variant selection, re-caption, and rebuild its CSV.
# Usage: rescore_r2_ws3.sh <probe.yaml> <pname> <tag>
source ~/spais-demo/work/polaris_env.sh
R=~/spais-demo/work/runs/polaris; P=~/spais-demo/work/polaris_wsl/probe_polaris.py
J=$(ls $R/${2}_*_${3}/${2}_*_${3}.json)
uv run python $P rescore --probe ~/spais-demo/work/polaris_wsl/probes/$1 --force-recaption --jsons $J 2>&1 | grep -v dev-dep
uv run python $P aggregate --out-csv $R/${2}_${3}_summary.csv --jsons $J 2>&1 | grep -v dev-dep | tr -d '\n '; echo
for f in $J; do python3 -c "import json,sys; s=json.load(open(sys.argv[1]))['summary']; print(sys.argv[1].split('/')[-1][:-5], {k:v for k,v in s['first_violation'].items() if v is not None}, 'phys', s.get('physics_error_step'), 'prog', round(s['rubric_max_progress'],2))" $f; done
