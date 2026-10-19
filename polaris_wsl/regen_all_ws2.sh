#!/usr/bin/env bash
# Workstream 2: regenerate all object splats one env at a time (keeps RAM low), then
# re-run the render-time benchmark. Extra args go to mesh_to_surfels.py (e.g. --quality medium).
set -eo pipefail
sed -i 's/\r$//' ~/spais-demo/work/polaris_wsl/*.sh
for e in food_bussing organize_tools pan_clean move_latte_cup block_stack_kitchen tape_into_container; do
  bash ~/spais-demo/work/polaris_wsl/mesh_to_surfels_ws2.sh --envs "$e" "$@" | grep "surfels\|rror"
done
