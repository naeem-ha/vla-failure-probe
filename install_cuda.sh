#!/bin/bash
# Toolkit-only CUDA 13.0 into ~/cuda-13.0 without root and without NVIDIA's installer UI
# (cuda-installer needs libxml2.so.2, which Ubuntu 26.04 no longer ships).
# Unpack the makeself payload and merge the toolkit component trees by hand.
set -e
X=~/spais-demo/tmp/cuda-x
rm -rf "$X" && mkdir -p "$X"
sh ~/spais-demo/dl/cuda_13.0.1_linux.run --noexec --target "$X" > /dev/null
ls "$X/builds"
DEST=~/cuda-13.0
mkdir -p "$DEST"
for comp in "$X"/builds/cuda_* "$X"/builds/lib*; do
  [ -d "$comp" ] || continue
  cp -a "$comp"/. "$DEST"/
done
# Some components ship lib64 vs targets layout; make sure lib64 exists.
[ -e "$DEST/lib64" ] || ln -s targets/x86_64-linux/lib "$DEST/lib64"
"$DEST/bin/nvcc" --version | tail -2
gcc --version | head -1
