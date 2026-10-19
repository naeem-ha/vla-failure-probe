#!/usr/bin/env bash
# Workstream 2: build ~/spais-demo/polaris/PolaRiS-Hub-wsl as a mirror of the stock
# PolaRiS-Hub. Directories are real; large binary files are symlinked to the stock
# hub; small text files (scene.usda, initial_conditions.json, config.yaml, ...) are
# copied so they can be edited and so Path.resolve() on scene.usda stays inside the
# -wsl tree. Generated object splats (assets/<rigid prim name>/splat.ply) are then
# written by mesh_to_surfels.py. Idempotent; never modifies the stock hub.
set -euo pipefail
SRC=$HOME/spais-demo/polaris/PolaRiS-Hub
DST=$HOME/spais-demo/polaris/PolaRiS-Hub-wsl
mkdir -p "$DST"
cd "$SRC"
find . -path ./.cache -prune -o -path ./.git -prune -o -type d -print | while read -r d; do
  mkdir -p "$DST/$d"
done
find . -path ./.cache -prune -o -path ./.git -prune -o -type f -print | while read -r f; do
  case "$f" in
    *.usda|*.json|*.yaml|*.yml|*.md|*.txt|*/.asset_hash|./.gitattributes)
      [ -e "$DST/$f" ] || cp -p "$f" "$DST/$f" ;;
    *)
      [ -e "$DST/$f" ] || ln -s "$SRC/${f#./}" "$DST/$f" ;;
  esac
done
echo "mirror ready: $DST"
du -sh --apparent-size "$DST" 2>/dev/null | tail -1
