#!/bin/bash
# Usage: bash try_render.sh <egl|glx|osmesa>
source ~/spais-demo/work/client_env.sh
export MUJOCO_GL=$1
[ "$1" = osmesa ] && export PYOPENGL_PLATFORM=osmesa
echo "DISPLAY=$DISPLAY WAYLAND_DISPLAY=$WAYLAND_DISPLAY MUJOCO_GL=$MUJOCO_GL"
timeout 300 python render_test.py 2>&1 | grep -v -i warn | tail -6
ls -la runs/ 2>/dev/null
