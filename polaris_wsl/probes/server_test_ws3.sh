#!/bin/bash
# Start the pi0.5-DROID server, send one dummy observation through ProbePolicy, then stop the server.
export PATH=$HOME/.local/bin:$PATH
mkdir -p ~/spais-demo/logs
: > ~/spais-demo/server.log
nohup bash ~/spais-demo/work/serve_droid.sh > ~/spais-demo/server.log 2>&1 &
# watchdog: the WSL VM has ~15 GB RAM; kill the server before the VM OOMs (it crashed once)
( while sleep 2; do
    pgrep -f '[s]erve_policy' > /dev/null || exit 0
    avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
    if [ "$avail" -lt 1500 ]; then echo "WATCHDOG: MemAvailable ${avail} MB, killing server"; pkill -9 -f '[s]erve_policy'; exit 0; fi
  done ) &
bash ~/spais-demo/work/wait_server.sh || { pkill -f '[s]erve_policy'; exit 1; }
source ~/spais-demo/work/polaris_env.sh
timeout 300 uv run python - <<'PY'
import sys, time, numpy as np
sys.path.insert(0, "/home/hamza/spais-demo/work/polaris_wsl")
import probe_polaris as pp
pol = pp.ProbePolicy("127.0.0.1", 8000, 8)
obs = {"splat": {"external_cam": np.random.randint(0, 255, (720, 1280, 3), np.uint8),
                 "wrist_cam": np.random.randint(0, 255, (720, 1280, 3), np.uint8)},
       "policy": {"arm_joint_pos": np.array([[0, -0.6, 0, -2.2, 0, 1.6, 0.8]]), "gripper_pos": np.array([[0.0]])}}
ext, wr = pol.model_images(obs)
req = pol.build_request(obs, "Put all the foods in the bowl", ext, wr)
print("request:", {k: (getattr(v, "shape", v), getattr(v, "dtype", "")) for k, v in req.items()})
t = time.time(); r = pol.client.infer(req); dt = time.time() - t
print("response keys:", list(r.keys()))
a = np.asarray(r["actions"]); print("actions shape", a.shape, a.dtype, "infer s", round(dt, 2))
print("first action", a[0].round(3)); print("gripper col", a[:, -1].round(2))
t = time.time(); r = pol.client.infer(req); print("2nd infer s", round(time.time() - t, 2))
act, inferred = pol.infer(obs, "stop", ext, wr); print("ProbePolicy action", act.round(3), inferred)
print("SERVER_TEST_OK")
PY
pkill -f '[s]erve_policy'; sleep 3; pkill -9 -f '[s]erve_policy'
nvidia-smi --query-gpu=memory.used --format=csv
