#!/bin/bash
R=~/spais-demo/work/polaris_wsl/probes/rescore_r2_ws3.sh
for t in b12 r2neg r2corr take; do echo "### $t"; bash $R food_bussing_battery.yaml food_battery $t | grep -v "first None->None  CHANGED"; done
echo "### r2latte"; bash $R latte_cleaner.yaml latte_cleaner r2latte | grep -v "first None->None  CHANGED"
