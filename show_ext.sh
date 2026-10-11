#!/bin/bash
cd ~/spais-demo/polaris/src
for d in diff-surfel-rasterization simple-knn; do
  echo "== $d"; ls "$d"; cat "$d/pyproject.toml" 2>/dev/null
done
