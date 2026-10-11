#!/bin/bash
# Build fixes for PolaRiS on this machine:
#  * flatdict 4.0.1 (via isaaclab) imports pkg_resources at build time; setuptools>=81 dropped it.
#  * the splat CUDA extensions (setup.py only, no pyproject) import torch at build time;
#    give their isolated build env the same torch as the runtime env.
cd ~/spais-demo/polaris
grep -q 'build-constraint-dependencies' pyproject.toml || \
  sed -i 's/^override-dependencies = \[/build-constraint-dependencies = ["setuptools<81"]\noverride-dependencies = [/' pyproject.toml
grep -q 'extra-build-dependencies' pyproject.toml || cat >> pyproject.toml <<'EOF'

[tool.uv.extra-build-dependencies]
diff-surfel-rasterization = [{ requirement = "torch", match-runtime = true }, "setuptools<81", "wheel"]
simple-knn = [{ requirement = "torch", match-runtime = true }, "setuptools<81", "wheel"]

[tool.uv.extra-build-variables]
diff-surfel-rasterization = { TORCH_CUDA_ARCH_LIST = "12.0" }
simple-knn = { TORCH_CUDA_ARCH_LIST = "12.0" }
EOF
sed -n '/\[tool.uv\]/,$p' pyproject.toml | grep -v '^#' | head -40
