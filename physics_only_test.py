"""Feasibility test for option C: can Isaac Lab run PolaRiS physics under WSL with no RTX cameras?

Builds DROID-FoodBussing with the camera sensors removed, steps zero actions, and prints
robot joint state and object poses. Run from ~/spais-demo/polaris with `uv run python`.
"""
import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = False
args_cli.headless = True
app = AppLauncher(args_cli).app

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import polaris.environments  # noqa: E402,F401  (registers envs)
from polaris.utils import DATA_PATH, load_eval_initial_conditions  # noqa: E402

import os  # noqa: E402
SIM_DEVICE = os.environ.get("SIM_DEVICE", "cpu")  # GPU PhysX fails under WSL and the cuda path deadlocks
usd = str(DATA_PATH / "food_bussing/scene.usda")
cfg = parse_env_cfg("DROID-FoodBussing", device=SIM_DEVICE, num_envs=1, use_fabric=False)
cfg.dynamic_setup(usd)
for name in ("wrist_cam", "external_cam"):
    if hasattr(cfg.scene, name):
        setattr(cfg.scene, name, None)
print("STAGE: cfg ready", flush=True)

t0 = time.time()
env = ManagerBasedRLEnv(cfg=cfg)
print(f"STAGE: env built in {time.time() - t0:.1f}s", flush=True)

instruction, poses = load_eval_initial_conditions(usd)
obs, _ = env.reset()
print("STAGE: reset ok; instruction:", instruction, flush=True)
print("objects:", list(env.scene.rigid_objects.keys()), flush=True)

t0 = time.time()
action = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
for i in range(60):
    obs, *_ = env.step(action)
dt = (time.time() - t0) / 60
print(f"STAGE: 60 steps ok, {dt * 1000:.1f} ms/step", flush=True)
print("arm_joint_pos:", obs["policy"]["arm_joint_pos"][0].cpu().numpy().round(3))
for name in env.scene.rigid_objects:
    print(f"  {name}: pos {env.scene[name].data.root_state_w[0, :3].cpu().numpy().round(3)}")
print("PHYSICS_ONLY_OK", flush=True)
env.close()
app.close()
