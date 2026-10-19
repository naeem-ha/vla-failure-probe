#!/bin/bash
source ~/spais-demo/work/polaris_env.sh
timeout 540 uv run python ~/spais-demo/work/physics_only_test.py > ~/spais-demo/logs/physics_test.log 2>&1
echo "EXIT=$?"
grep -E "STAGE|objects:|arm_joint|  [a-z_]+: pos|PHYSICS_ONLY_OK|Error|Traceback|rror:" ~/spais-demo/logs/physics_test.log \
  | grep -v -E "omni.ext.plugin|neuraylib|libXt|libGLU" | tail -30
