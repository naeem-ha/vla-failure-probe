#!/bin/bash
# Usage: bash ~/spais-demo/work/polaris_wsl/test_env_ws1.sh [extra args for test_env_ws1.py]
source ~/spais-demo/work/polaris_env.sh
mkdir -p ~/spais-demo/logs
timeout 600 uv run python ~/spais-demo/work/polaris_wsl/test_env_ws1.py "$@" 2>&1 | tee ~/spais-demo/logs/test_env_ws1.log | grep -v -E "libXt|libGLU|neuraylib|GPU Foundation|^\s*$" | grep -E "STAGE|cam |bodies|USD-vs|instruction|obs keys|policy:|info:|TIMING|arm after|gripper|not drawn|Error|error|Traceback|File \"|WS1_TEST_OK|world pts|^\w+Error" | grep -v "omni\.\|carb\." | head -120
pkill -f '[t]est_env_ws1.py' || true
