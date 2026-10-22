#!/bin/bash
# Round 2: negation, correction (fixed + grasp-triggered), latte cleaner, then submission takes.
S=~/spais-demo/work/polaris_wsl/probes/run_probe_set.sh
TAG=r2neg   bash $S food_bussing_battery.yaml 5 "negation"
TAG=r2corr  bash $S food_bussing_battery.yaml 5 "correction correction_grasp"
TAG=r2latte bash $S latte_cleaner.yaml 5 "hazard_spray hazard_board"
for ic in 2 0 4; do
  FIXED_IC=$ic SEED_START=10 TAG=take bash $S food_bussing_battery.yaml 4 "stock"
done
echo ROUND2_DONE
