#!/bin/bash
# Parallel downloads for the PolaRiS setup. Each writes its own log in ~/spais-demo/logs.
mkdir -p ~/spais-demo/logs ~/spais-demo/dl
cd ~/spais-demo

( wget -q -c -O dl/cuda_13.0.1_linux.run \
    https://developer.download.nvidia.com/compute/cuda/13.0.1/local_installers/cuda_13.0.1_580.82.07_linux.run \
  && echo CUDA_DL_OK || echo CUDA_DL_FAIL ) > logs/cuda_dl.log 2>&1 &

( python3 ~/spais-demo/work/download_ckpt.py checkpoints/polaris/pi05_droid_jointpos_polaris \
  && echo CKPT_DL_OK || echo CKPT_DL_FAIL ) > logs/ckpt_dl.log 2>&1 &

( cd polaris && ~/.local/bin/uvx hf download owhan/PolaRiS-Hub --repo-type=dataset --local-dir ./PolaRiS-Hub \
  && echo HUB_DL_OK || echo HUB_DL_FAIL ) > logs/hub_dl.log 2>&1 &

wait
tail -n 2 logs/cuda_dl.log logs/ckpt_dl.log logs/hub_dl.log
ls -la dl/; du -sh ~/.cache/openpi/openpi-assets/checkpoints/polaris polaris/PolaRiS-Hub
