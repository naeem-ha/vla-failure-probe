#!/bin/bash
# Probes #1 (stock) and #2 (hazard_bowl), ICs 0-4, one Isaac process per episode.
TAG=b12 bash ~/spais-demo/work/polaris_wsl/probes/run_probe_set.sh food_bussing_battery.yaml 5 "stock hazard_bowl"
