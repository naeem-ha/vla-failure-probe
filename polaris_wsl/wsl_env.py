"""Splat-only PolaRiS env for WSL2 (no Isaac camera sensors, no RTX).

WSLSplatEnv subclasses ManagerBasedRLSplatEnv and keeps its interface:
    obs, info = env.reset(object_positions=ic)
    obs, rew, term, trunc, info = env.step(action, expensive=True)
    obs["policy"]["arm_joint_pos"], obs["policy"]["gripper_pos"]   (torch, (1, 7) / (1, 1))
    obs["splat"]["external_cam"], obs["splat"]["wrist_cam"]         (np.uint8, 720x1280x3)
    info["rubric"] = {"success", "progress", "metrics"}

Everything visible is a splat: background/static splats and any rigid object with
assets/<name>/splat.ply (scene dir), plus the per-link robot splats in
<data>/nvidia_droid/SEGMENTED. Rigid objects without a splat are simply not drawn.
Physics runs on CPU with fabric disabled (GPU PhysX is broken under WSL).

Data path: polaris.utils.DATA_PATH, i.e. $POLARIS_DATA_PATH (read at import time),
so set POLARIS_DATA_PATH before importing polaris to point at PolaRiS-Hub-wsl.

Must be imported after the Isaac Sim app is launched (AppLauncher, enable_cameras=False).
"""
from __future__ import annotations

import copy
import os
import sys
import time
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.sensors import CameraCfg

import polaris.environments  # noqa: F401  (registers DROID-* envs)
from polaris.environments.manager_based_rl_splat_environment import ManagerBasedRLSplatEnv
from polaris.splat_renderer import SplatRenderer
from polaris.utils import DATA_PATH

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cameras import CameraRig, LinkBinding, resolve_prim_path  # noqa: E402


def strip_camera_sensors(cfg) -> dict:
    """Remove every CameraCfg from cfg.scene; return {name: deep copy of its cfg}."""
    cams = {}
    for name in list(vars(cfg.scene).keys()):
        val = getattr(cfg.scene, name)
        if isinstance(val, CameraCfg):
            cams[name] = copy.deepcopy(val)
            setattr(cfg.scene, name, None)
    return cams


class WSLSplatEnv(ManagerBasedRLSplatEnv):
    """ManagerBasedRLSplatEnv that renders both cameras purely from splats.

    Args:
        cfg: DROID EnvCfg on which dynamic_setup(usd_file) has NOT yet been called
             (it is called here), or has been called already (pass usd_file anyway).
        usd_file: scene.usda path.
        rubric: polaris Rubric (optional).
        splat_device: torch device for the splat renderer (physics stays on cfg.sim.device).
        half_res_blur: replicate stock's downsample-by-2 then upsample (default True).
    """

    def __init__(
        self,
        cfg,
        *args,
        rubric=None,
        usd_file: str | None = None,
        splat_device: str = "cuda:0",
        half_res_blur: bool = True,
        **kwargs,
    ):
        if usd_file is None:
            raise ValueError("WSLSplatEnv needs usd_file")
        self.usd_file = str(Path(usd_file).resolve())
        if not getattr(cfg, "_wsl_dynamic_setup_done", False):
            cfg.dynamic_setup(self.usd_file)
        self._cam_cfgs = strip_camera_sensors(cfg)
        self._splat_device = splat_device
        self._half_res_blur = half_res_blur
        self._last_rgb: dict = {}
        self.timing = {"physics": 0.0, "render": 0.0, "n": 0}
        # parent: usd_file=None so it does not redo dynamic_setup; it calls our
        # overridden setup_splat_world_and_robot_views / setup_splat_robot.
        super().__init__(cfg, *args, rubric=rubric, usd_file=None, **kwargs)

    # ---------------------------------------------------------------- setup
    def _object_splat_path(self, name: str) -> Path:
        return Path(self.usd_file).parent / "assets" / name / "splat.ply"

    def setup_splat_world_and_robot_views(self):
        self.views = {}
        splats = {}
        self.objects_without_splat = []
        for name in self.scene.rigid_objects:
            p = self._object_splat_path(name)
            if p.exists():
                splats[name] = p
            else:
                self.objects_without_splat.append(name)
        if self.objects_without_splat:
            print(f"[WSLSplatEnv] not drawn (no splat.ply): {self.objects_without_splat}", flush=True)
        self.robot_ = self.scene["robot"]
        self.cam_rig = CameraRig(self.robot_, self._cam_cfgs)
        self.splat_renderer = SplatRenderer(splats=splats, device=self._splat_device)
        self.splat_renderer.init_cameras(self.cam_rig.renderer_camera_cfg())

    def setup_splat_robot(self):
        more_splats = {}
        robot_asset_path = Path(self.cfg.scene.robot.spawn.usd_path).parent
        robot_root = resolve_prim_path(self.robot_.cfg.prim_path)
        for ply in sorted(robot_asset_path.glob("SEGMENTED/*.ply")):
            sim_path = ply.stem.replace("-", "/")
            self.views[ply.stem] = LinkBinding(self.robot_, f"{robot_root}/{sim_path}")
            more_splats[ply.stem] = ply
        self.splat_renderer.add_splats(more_splats)
        names = list(self.views)
        self._bind_idx = torch.tensor([self.views[n].body_idx for n in names])
        self._bind_pos = torch.stack([self.views[n].rel_pos for n in names])
        self._bind_quat = torch.stack([self.views[n].rel_quat for n in names])
        self._bind_names = names

    # --------------------------------------------------------------- render
    def robot_geom_poses(self, link_tf: torch.Tensor | None = None):
        """World poses of each SEGMENTED geometry prim: (names, pos (N,3), quat (N,4))."""
        if link_tf is None:
            link_tf = self.cam_rig.link_transforms()
        pb = link_tf[self._bind_idx, :3]
        qb = link_tf[self._bind_idx, 3:]
        pos = pb + math_utils.quat_apply(qb, self._bind_pos)
        quat = math_utils.quat_mul(qb, self._bind_quat)
        return self._bind_names, pos, quat

    def transform_sim_to_splat(self, transform_static=False, link_tf=None):
        all_transforms = {}
        for name in self.scene.rigid_objects:
            if ("static" not in name or transform_static) and self._object_splat_path(name).exists():
                st = self.scene[name].data.root_state_w[0]
                all_transforms[name] = (st[:3], st[3:7])
        names, pos, quat = self.robot_geom_poses(link_tf)
        for i, n in enumerate(names):
            all_transforms[n] = (pos[i], quat[i])
        if all_transforms:
            self.splat_renderer.transform_many(all_transforms)

    def render_splat(self, link_tf=None):
        extr = self.cam_rig.extrinsics(link_tf)
        rgb = self.splat_renderer.render(extr)
        out = {}
        for k, v in rgb.items():
            img = (np.clip(v.detach().cpu().numpy(), 0, 1) * 255).astype(np.uint8)
            if self._half_res_blur:  # same as stock render_splat
                img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
                img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
            out[k] = img
        return out

    def custom_render(self, expensive: bool, transform_static: bool = False):
        if expensive or not self._last_rgb:
            with torch.no_grad():
                link_tf = self.cam_rig.link_transforms()
                self.transform_sim_to_splat(transform_static=transform_static, link_tf=link_tf)
                self._last_rgb = self.render_splat(link_tf)
        # stock returns raw sim camera images when expensive=False; we have none, so
        # return the last splat render (the policy client ignores them in that case).
        return {k: v.copy() for k, v in self._last_rgb.items()}

    # ------------------------------------------------- timing instrumentation
    def step(self, action, expensive=True):
        t0 = time.perf_counter()
        obs, rew, done, trunc, info = super(ManagerBasedRLSplatEnv, self).step(action)
        t1 = time.perf_counter()
        obs["splat"] = self.custom_render(expensive)
        if self._splat_device != "cpu":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        info.update(self._evaluate_rubric())
        self.timing["physics"] += t1 - t0
        self.timing["render"] += t2 - t1
        self.timing["n"] += 1
        self.last_step_timing = (t1 - t0, t2 - t1)
        return obs, rew, done, trunc, info


def make_wsl_env(
    env_name: str,
    usd_path_override: str | None = None,
    splat_device: str = "cuda:0",
    sim_device: str = "cpu",
    **kwargs,
) -> WSLSplatEnv:
    """Build a registered DROID-* env as a WSLSplatEnv (rubric and usd from the gym registry).

    usd_path_override: alternate scene.usda (e.g. a PolaRiS-Hub-wsl copy with surfel splats).
    If not given, the registry's usd path is rebased onto the current DATA_PATH.
    """
    spec = gym.spec(env_name)
    kw = spec.kwargs
    cfg_cls = kw["env_cfg_entry_point"]
    cfg = cfg_cls() if isinstance(cfg_cls, type) else copy.deepcopy(cfg_cls)
    if usd_path_override is not None:
        usd = Path(usd_path_override)
    else:
        reg = Path(kw["usd_file"])
        usd = DATA_PATH / reg.parent.name / reg.name
    cfg.sim.device = sim_device
    cfg.sim.use_fabric = False
    cfg.scene.num_envs = 1
    return WSLSplatEnv(cfg, rubric=kw.get("rubric"), usd_file=str(usd), splat_device=splat_device, **kwargs)
