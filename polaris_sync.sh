#!/bin/bash
source ~/spais-demo/work/polaris_env.sh
uv sync 2>&1 | grep -v -E '^\s+\+ ' | tail -40
echo "SYNC_EXIT=${PIPESTATUS[0]}"
uv run python -c "import torch, isaacsim; print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
