#!/bin/bash
# Re-evaluate the b12 batch with the current checks (adds the max_above rule) and rebuild the CSV.
source ~/spais-demo/work/polaris_env.sh
R=~/spais-demo/work/runs/polaris
P=~/spais-demo/work/polaris_wsl/probe_polaris.py
J=$(ls $R/food_battery_*_b12/food_battery_*_b12.json)
uv run python $P rescore --probe ~/spais-demo/work/polaris_wsl/probes/food_bussing_battery.yaml --jsons $J 2>&1 | grep -v dev-dep
uv run python $P aggregate --out-csv $R/food_battery_b12_summary.csv --jsons $J 2>&1 | grep -v dev-dep
cat $R/food_battery_b12_summary.csv | cut -d, -f1-12
for f in $J; do echo "$(basename $f .json): $(python3 -c "import json,sys; s=json.load(open(sys.argv[1]))['summary']; print({k:v for k,v in s['first_violation'].items() if v is not None}, 'progress', s['progress_checks_first'], 'phys', s.get('physics_error_step'))" $f)"; done
