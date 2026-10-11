#!/bin/bash
# JIT-build the 2DGS rasterizer + simple-knn outside Isaac Sim.
# torch >= 2.10 headers need C++20, but both loaders hard-code -std=c++17.
source ~/spais-demo/work/polaris_env.sh
sed -i 's/-std=c++17/-std=c++20/' src/diff-surfel-rasterization/diff_surfel_rasterization/__init__.py \
                                  src/simple-knn/simple_knn/__init__.py
uv run python -c "import diff_surfel_rasterization as d, simple_knn as k; print('SPLAT_OK')" > ~/spais-demo/logs/splat_build.log 2>&1
grep -n -E "error|SPLAT_OK" ~/spais-demo/logs/splat_build.log | head -15
