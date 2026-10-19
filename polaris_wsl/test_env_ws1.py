"""WS1 validation: build DROID-FoodBussing as a WSLSplatEnv, reset with IC 0, hold, then move
the arm, saving splat renders of both cameras to spais-demo/runs/ws1/.

Run (inside WSL): bash ~/spais-demo/work/polaris_wsl/test_env_ws1.sh [ENV_NAME]
"""
import argparse
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="DROID-FoodBussing")
parser.add_argument("--usd", default=None)
parser.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "runs" / "ws1"))
parser.add_argument("--move_steps", type=int, default=50)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = False
args_cli.headless = True
app = AppLauncher(args_cli).app

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wsl_env import make_wsl_env  # noqa: E402
from polaris.utils import load_eval_initial_conditions  # noqa: E402

out = Path(args_cli.out)
out.mkdir(parents=True, exist_ok=True)
tag = args_cli.env.replace("DROID-", "").lower()


def save(obs, name):
    for cam, img in obs["splat"].items():
        cv2.imwrite(str(out / f"{tag}_{name}_{cam}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    both = np.concatenate([obs["splat"]["external_cam"], obs["splat"]["wrist_cam"]], axis=1)
    cv2.imwrite(str(out / f"{tag}_{name}_both.png"), cv2.cvtColor(cv2.resize(both, (1280, 360)), cv2.COLOR_RGB2BGR))


t0 = time.time()
env = make_wsl_env(args_cli.env, args_cli.usd)
print(f"STAGE: env built in {time.time() - t0:.1f}s", flush=True)
print("bodies:", env.robot_.body_names, flush=True)
for n, s in env.cam_rig.specs.items():
    print(f"cam {n}: res {s.res} fovx {np.degrees(s.fovx):.2f} fovy {np.degrees(s.fovy):.2f} body {s.body_idx}", flush=True)


def project(cam, pts):
    """Project world points with the same pinhole model the splat renderer uses."""
    s = env.cam_rig.specs[cam]
    e = env.cam_rig.extrinsics()[cam]
    H, W = s.res
    fx, fy = W / (2 * np.tan(s.fovx / 2)), H / (2 * np.tan(s.fovy / 2))
    d = (np.asarray(pts) - e["pos"]) @ e["rot"]  # world-convention cam frame: x fwd, y left, z up
    return [(int(W / 2 - fx * q[1] / q[0]), int(H / 2 - fy * q[2] / q[0])) if q[0] > 0 else None for q in d]


def annotate(obs, name):
    labels, pts = [], []
    for n in env.scene.rigid_objects:
        labels.append(n)
        pts.append(env.scene[n].data.root_state_w[0, :3].numpy())
    names, pos, _ = env.robot_geom_poses()
    lt = env.cam_rig.link_transforms()
    for b in ("panda_link0", "base_link"):
        labels.append(b)
        pts.append(lt[env.robot_.body_names.index(b), :3].numpy())
    for cam in ("external_cam", "wrist_cam"):
        img = cv2.cvtColor(obs["splat"][cam], cv2.COLOR_RGB2BGR).copy()
        for lab, uv in zip(labels, project(cam, pts)):
            if uv is None:
                continue
            cv2.circle(img, uv, 8, (0, 0, 255), 2)
            cv2.putText(img, lab, (uv[0] + 10, uv[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imwrite(str(out / f"{tag}_{name}_{cam}_annot.png"), img)
    print("world pts:", {l: np.round(p, 3).tolist() for l, p in zip(labels, pts)}, flush=True)


instruction, ics = load_eval_initial_conditions(env.usd_file)
print("instruction:", instruction, flush=True)
obs, info = env.reset(object_positions=ics[0])
print("obs keys:", list(obs.keys()), {k: v.shape for k, v in obs["splat"].items()}, flush=True)
print("policy:", {k: v.cpu().numpy().round(3).tolist() for k, v in obs["policy"].items()}, flush=True)
print("info:", info, flush=True)
save(obs, "reset")
annotate(obs, "reset")

# hold current joints
q0 = obs["policy"]["arm_joint_pos"].clone()
hold = torch.cat([q0, torch.zeros(1, 1)], dim=1)
for i in range(5):
    obs, rew, term, trunc, info = env.step(hold)
save(obs, "hold")

# move: rotate base +0.6 rad, lift shoulder, close gripper
target = q0.clone()
target[0, 0] += 0.6
target[0, 1] += 0.3
target[0, 3] += 0.4
act = torch.cat([target, torch.ones(1, 1)], dim=1)
env.timing = {"physics": 0.0, "render": 0.0, "n": 0}
for i in range(args_cli.move_steps):
    obs, rew, term, trunc, info = env.step(act)
    if i == args_cli.move_steps // 2:
        save(obs, "move_mid")
save(obs, "moved")
annotate(obs, "moved")
print("arm after move:", obs["policy"]["arm_joint_pos"][0].numpy().round(3), "target", target[0].numpy().round(3))
print("gripper:", obs["policy"]["gripper_pos"][0].numpy().round(3))
t = env.timing
print(f"TIMING over {t['n']} steps: physics {1000 * t['physics'] / t['n']:.1f} ms/step, "
      f"render {1000 * t['render'] / t['n']:.1f} ms/step", flush=True)
# cheap step timing
t1 = time.perf_counter()
for i in range(10):
    env.step(act, expensive=False)
print(f"TIMING expensive=False: {100 * (time.perf_counter() - t1):.1f} ms/step", flush=True)
print("info:", info["rubric"], flush=True)
print("WS1_TEST_OK", flush=True)
env.close()
app.close()
