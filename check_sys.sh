#!/bin/bash
which nvcc ffmpeg gcc g++ 2>&1
ls -d /usr/local/cuda* 2>/dev/null
nvidia-smi | head -4 | tail -2
sudo -n true 2>/dev/null && echo SUDO_NOPASS || echo SUDO_NEEDS_PASSWORD
cat /etc/os-release | grep PRETTY
