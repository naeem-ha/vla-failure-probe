# Source this inside WSL to get the LIBERO client environment.
#   source ~/spais-demo/work/client_env.sh
OPENPI=~/spais-demo/openpi
source "$OPENPI/examples/libero/.venv/bin/activate"
export PYTHONPATH=$PYTHONPATH:$OPENPI/third_party/libero
# EGL fails under WSL2; GLX through WSLg works.
export MUJOCO_GL=${MUJOCO_GL:-glx}
cd ~/spais-demo/work
