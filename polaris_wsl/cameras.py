"""Camera intrinsics and world poses for PolaRiS without Isaac camera sensors.

Stock PolaRiS feeds the splat renderer with, per camera:
    init_cameras({name: {"res": (H, W), "fovx": rad, "fovy": rad}})
    render({name: {"pos": pos_w (3,), "rot": matrix_from_quat(quat_w_world) (3, 3)}})
where pos_w / quat_w_world come from the Isaac Lab Camera sensor: the USD camera prim's
world pose (OpenGL convention: -Z forward, +Y up) converted to the "world" convention
(+X forward, +Z up). SplatRenderer.render() then right-multiplies by its own permutation
to get a COLMAP-style camera. This module reproduces those exact quantities:

* intrinsics: fovx = 2 atan(h_ap / 2f), fovy = 2 atan(v_ap / 2f) from the spawn cfg
  (vertical aperture defaulted to square pixels exactly like isaaclab Camera.__init__) or,
  for spawn=None cameras (the scene's own USD camera, e.g. external_cam), from the prim.
* poses:
  - camera whose parent is an articulation link (wrist_cam on Robotiq base_link):
    link pose from the PhysX articulation view, composed with the cfg offset
    (offset rot converted from cfg.offset.convention to opengl, as Camera.__init__ does).
  - any other camera: static; world transform read once from the USD stage
    (spawn=None -> the prim itself; spawned -> parent prim world xform * offset).

Must be imported after the Isaac Sim app is launched.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Usd, UsdGeom, UsdPhysics

ENV_NS = "/World/envs/env_0"


def resolve_prim_path(prim_path: str) -> str:
    return prim_path.replace("{ENV_REGEX_NS}", ENV_NS).replace("/World/envs/env_.*", ENV_NS)


def usd_world_pose(prim) -> tuple[np.ndarray, np.ndarray]:
    """World (pos, quat wxyz) of a USD prim, scale removed (same as Fabric decomposition)."""
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return matrix_to_pose(m)


def matrix_to_pose(m: Gf.Matrix4d) -> tuple[np.ndarray, np.ndarray]:
    t = Gf.Transform(m)
    p = t.GetTranslation()
    q = t.GetRotation().GetQuat()
    im = q.GetImaginary()
    return (
        np.array([p[0], p[1], p[2]], dtype=np.float64),
        np.array([q.GetReal(), im[0], im[1], im[2]], dtype=np.float64),
    )


def compose(p_a, q_a, p_b, q_b):
    """(p_a, q_a) * (p_b, q_b) for torch tensors (..., 3) / (..., 4) wxyz."""
    return p_a + math_utils.quat_apply(q_a, p_b), math_utils.quat_mul(q_a, q_b)


def to_world_convention(quat_gl: torch.Tensor) -> torch.Tensor:
    return math_utils.convert_camera_frame_orientation_convention(
        quat_gl.reshape(-1, 4).float(), origin="opengl", target="world"
    )


@dataclass
class CamSpec:
    name: str
    res: tuple[int, int]  # (H, W)
    fovx: float
    fovy: float
    # static cameras: fixed world pose (world convention)
    static_pos: np.ndarray | None = None
    static_rot: np.ndarray | None = None
    # link-attached cameras
    body_idx: int | None = None
    offset_pos: torch.Tensor | None = None  # (3,)
    offset_quat_gl: torch.Tensor | None = None  # (4,) wxyz, opengl convention


class CameraRig:
    """Computes splat-renderer camera dicts from physics state, no Isaac camera sensors.

    Args:
        robot: the isaaclab Articulation (env.scene["robot"]); must be initialized.
        cam_cfgs: {name: CameraCfg} captured from the env cfg before the sensors were removed.
    """

    def __init__(self, robot, cam_cfgs: dict):
        self.robot = robot
        self.robot_root = resolve_prim_path(robot.cfg.prim_path)
        self.stage = get_current_stage()
        self.specs: dict[str, CamSpec] = {}
        for name, cfg in cam_cfgs.items():
            self.specs[name] = self._build(name, cfg)

    # ------------------------------------------------------------------ setup
    def _build(self, name, cfg) -> CamSpec:
        H, W = int(cfg.height), int(cfg.width)
        prim_path = resolve_prim_path(cfg.prim_path)
        if cfg.spawn is None:
            prim = self.stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise RuntimeError(f"camera prim {prim_path} not found for {name}")
            cam = UsdGeom.Camera(prim)
            f = cam.GetFocalLengthAttr().Get()
            h_ap = cam.GetHorizontalApertureAttr().Get()
            v_ap = cam.GetVerticalApertureAttr().Get()
        else:
            f = cfg.spawn.focal_length
            h_ap = cfg.spawn.horizontal_aperture
            v_ap = cfg.spawn.vertical_aperture
            if v_ap is None:  # isaaclab Camera.__init__ default (square pixels)
                v_ap = h_ap * H / W
        spec = CamSpec(
            name=name,
            res=(H, W),
            fovx=float(2 * np.arctan(h_ap / (2 * f))),
            fovy=float(2 * np.arctan(v_ap / (2 * f))),
        )

        if cfg.spawn is None:
            # existing prim, offset is NOT applied by isaaclab in this case
            parent_path, local = None, None
        else:
            parent_path = prim_path.rsplit("/", 1)[0]
            rot = torch.tensor(cfg.offset.rot, dtype=torch.float32).unsqueeze(0)
            rot_gl = math_utils.convert_camera_frame_orientation_convention(
                rot, origin=cfg.offset.convention, target="opengl"
            )[0]
            local = (torch.tensor(cfg.offset.pos, dtype=torch.float32), rot_gl)

        body_idx = None
        if parent_path is not None and parent_path.startswith(self.robot_root + "/"):
            body_name = parent_path.rsplit("/", 1)[1]
            if body_name in self.robot.body_names:
                body_idx = self.robot.body_names.index(body_name)

        if body_idx is not None:
            spec.body_idx = body_idx
            spec.offset_pos, spec.offset_quat_gl = local
        else:
            if parent_path is None:
                p, q = usd_world_pose(self.stage.GetPrimAtPath(prim_path))
                p, q = torch.tensor(p, dtype=torch.float32), torch.tensor(q, dtype=torch.float32)
            else:
                pp, pq = usd_world_pose(self.stage.GetPrimAtPath(parent_path))
                p, q = compose(
                    torch.tensor(pp, dtype=torch.float32),
                    torch.tensor(pq, dtype=torch.float32),
                    local[0],
                    local[1],
                )
            q_world = to_world_convention(q)
            spec.static_pos = p.numpy().astype(np.float32)
            spec.static_rot = math_utils.matrix_from_quat(q_world)[0].numpy()
        return spec

    # ------------------------------------------------------------- interface
    def renderer_camera_cfg(self) -> dict:
        """Argument for SplatRenderer.init_cameras."""
        return {n: {"res": s.res, "fovx": s.fovx, "fovy": s.fovy} for n, s in self.specs.items()}

    def link_transforms(self) -> torch.Tensor:
        """(num_bodies, 7) link poses [pos, quat wxyz] straight from the PhysX view (always fresh)."""
        t = self.robot.root_physx_view.get_link_transforms()[0].clone()  # pos + quat xyzw
        return torch.cat([t[:, :3], t[:, [6, 3, 4, 5]]], dim=-1).float().cpu()

    def extrinsics(self, link_tf: torch.Tensor | None = None, names=None) -> dict:
        """{name: {"pos": (3,) np, "rot": (3,3) np}} in Isaac 'world' camera convention,
        identical to stock (data.pos_w, matrix_from_quat(data.quat_w_world))."""
        out = {}
        for n, s in self.specs.items():
            if names is not None and n not in names:
                continue
            if s.body_idx is None:
                out[n] = {"pos": s.static_pos, "rot": s.static_rot}
                continue
            if link_tf is None:
                link_tf = self.link_transforms()
            pb, qb = link_tf[s.body_idx, :3], link_tf[s.body_idx, 3:]
            p, q_gl = compose(pb, qb, s.offset_pos, s.offset_quat_gl)
            q_world = to_world_convention(q_gl)
            out[n] = {
                "pos": p.numpy().astype(np.float32),
                "rot": math_utils.matrix_from_quat(q_world)[0].numpy(),
            }
        return out


class LinkBinding:
    """Maps a robot visual-geometry prim (one SEGMENTED splat) to its rigid-body link,
    so its world pose = link pose (PhysX) * constant local offset (from USD)."""

    def __init__(self, robot, geom_prim_path: str):
        stage = get_current_stage()
        robot_root = resolve_prim_path(robot.cfg.prim_path)
        prim = stage.GetPrimAtPath(geom_prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"robot geometry prim not found: {geom_prim_path}")
        # nearest ancestor (or self) that is a rigid body of the articulation
        link = prim
        while link.IsValid() and str(link.GetPath()).startswith(robot_root):
            if link.HasAPI(UsdPhysics.RigidBodyAPI) and link.GetName() in robot.body_names:
                break
            link = link.GetParent()
        if not link.IsValid() or not link.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"no articulation link above {geom_prim_path}")
        self.body_name = link.GetName()
        self.body_idx = robot.body_names.index(self.body_name)
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        m_geom = cache.GetLocalToWorldTransform(prim)
        m_link = cache.GetLocalToWorldTransform(link)
        # world pose of each without scale, then relative pose (robust to scaled meshes)
        pg, qg = matrix_to_pose(m_geom)
        pl, ql = matrix_to_pose(m_link)
        pg, qg, pl, ql = (torch.tensor(x, dtype=torch.float64) for x in (pg, qg, pl, ql))
        ql_inv = math_utils.quat_inv(ql)
        self.rel_pos = math_utils.quat_apply(ql_inv, pg - pl).float()
        self.rel_quat = math_utils.quat_mul(ql_inv, qg).float()
        self.prim = prim
