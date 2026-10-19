"""Workstream 2 validation (no Isaac): render generated object splats with PolaRiS's own
SplatRenderer/GaussianModel and compare against a software rasterization of the source
USD mesh (same camera). Also renders food_bussing objects posed in the scene splat.

Run: bash ~/spais-demo/work/polaris_wsl/test_surfels_ws2.sh [--envs ...] [--scene-only]
Outputs: ~/spais-demo/work/runs/ws2/*.png  (Windows: spais-demo\\runs\\ws2)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from pxr import Usd, UsdPhysics

sys.path.insert(0, str(Path(__file__).parent))
import mesh_to_surfels as m2s  # noqa: E402


def _stub_polaris_utils():
    """polaris.utils imports isaaclab.envs (needs a running SimulationApp). The splat
    renderer only uses two quaternion helpers from it, so load exactly those function
    definitions from PolaRiS's own source into a stub module."""
    import ast
    import importlib.util
    import types

    spec = importlib.util.find_spec("polaris")
    src_file = Path(spec.submodule_search_locations[0]) / "utils.py"
    tree = ast.parse(src_file.read_text())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name in ("rotate_vector_by_quaternion", "multiply_quaternions")]
    mod = types.ModuleType("polaris.utils")
    mod.torch = torch
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(src_file), "exec"), mod.__dict__)
    import polaris
    sys.modules["polaris.utils"] = mod
    polaris.utils = mod


_stub_polaris_utils()
from polaris.splat_renderer import SplatRenderer  # noqa: E402

OUT = Path.home() / "spais-demo/work/runs/ws2"
W, H = 640, 480
FOVX = math.radians(50.0)
FOVY = 2 * math.atan(math.tan(FOVX / 2) * H / W)


def look_at(eye, target, up=(0, 0, 1)):
    """COLMAP-style camera-to-world rotation (x right, y down, z forward)."""
    eye, target, up = map(lambda v: np.asarray(v, float), (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, (0, 1, 0))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)


def views_for(center, radius, fovx=FOVX):
    d = radius / math.sin(min(FOVX, FOVY) / 2) * 1.1
    out = []
    for az, el in [(30, 35), (120, 20), (210, 50), (300, 5)]:
        a, e = math.radians(az), math.radians(el)
        eye = center + d * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        out.append((look_at(eye, center), eye))
    return out


# ------------------------------------------------------------- mesh rasterizer
def raster_mesh(parts, R, eye, fovx=FOVX, fovy=FOVY, w=W, h=H, bg=0.5):
    """Simple z-buffered triangle rasterizer, flat texture colour (no lighting), same
    pinhole model as the 2DGS renderer (principal point at image centre)."""
    img = np.full((h, w, 3), bg, np.float32)
    zbuf = np.full(h * w, np.inf)
    cols = np.zeros((h * w, 3), np.float32)
    fx = w / (2 * math.tan(fovx / 2))
    fy = h / (2 * math.tan(fovy / 2))
    for p in parts:
        tri = p["tri"]
        cam = (tri - eye) @ R  # (T,3,3) camera coords
        z = cam[..., 2]
        ok = (z > 0.01).all(1)
        cam, z = cam[ok], z[ok]
        uv = p["uv"][ok] if p["uv"] is not None else None
        col = p["col"][ok] if p["col"] is not None else None
        px = fx * cam[..., 0] / z + w / 2 - 0.5
        py = fy * cam[..., 1] / z + h / 2 - 0.5
        x0 = np.clip(np.floor(px.min(1)), 0, w - 1).astype(int)
        x1 = np.clip(np.ceil(px.max(1)), 0, w - 1).astype(int)
        y0 = np.clip(np.floor(py.min(1)), 0, h - 1).astype(int)
        y1 = np.clip(np.ceil(py.max(1)), 0, h - 1).astype(int)
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        bmax = np.maximum(bw, bh)
        for lo, hi in [(0, 4), (4, 16), (16, 64), (64, 10**9)]:  # bucket by bbox size
            sel = np.where((bmax > lo) & (bmax <= hi))[0]
            if len(sel) == 0:
                continue
            S = int(min(hi, bmax[sel].max()))
            for chunk in np.array_split(sel, max(1, len(sel) * S * S // 4_000_000 + 1)):
                gx, gy = np.meshgrid(np.arange(S), np.arange(S))
                X = x0[chunk, None] + gx.ravel()[None]
                Y = y0[chunk, None] + gy.ravel()[None]
                inside_box = (X <= x1[chunk, None]) & (Y <= y1[chunk, None])
                a, b, c = (np.stack([px[chunk, i], py[chunk, i]], 1) for i in range(3))
                den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
                den = np.where(np.abs(den) < 1e-12, 1e-12, den)[:, None]
                l0 = ((b[:, 1, None] - c[:, 1, None]) * (X - c[:, 0, None]) + (c[:, 0, None] - b[:, 0, None]) * (Y - c[:, 1, None])) / den
                l1 = ((c[:, 1, None] - a[:, 1, None]) * (X - c[:, 0, None]) + (a[:, 0, None] - c[:, 0, None]) * (Y - c[:, 1, None])) / den
                l2 = 1 - l0 - l1
                m = inside_box & (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
                ti, pi = np.nonzero(m)
                if len(ti) == 0:
                    continue
                L = np.stack([l0[ti, pi], l1[ti, pi], l2[ti, pi]], 1)
                tri_id = chunk[ti]
                # perspective-correct depth interpolation
                iz = (L / z[tri_id]).sum(1)
                depth = 1 / iz
                pix = Y[ti, pi] * w + X[ti, pi]
                if uv is not None and p["tex"] is not None:
                    Lp = (L / z[tri_id]) / iz[:, None]
                    uvi = np.einsum("nk,nkd->nd", Lp, uv[tri_id])
                    c_ = m2s.sample_texture(p["tex"], uvi)
                elif col is not None:
                    c_ = np.einsum("nk,nkd->nd", L, col[tri_id])
                else:
                    c_ = np.full((len(pix), 3), 0.6)
                closer = depth < zbuf[pix]
                pix, depth, c_ = pix[closer], depth[closer], c_[closer]
                order = np.lexsort((depth, pix))
                pix, depth, c_ = pix[order], depth[order], c_[order]
                _, first = np.unique(pix, return_index=True)
                zbuf[pix[first]] = depth[first]
                cols[pix[first]] = c_[first]
    hit = np.isfinite(zbuf)
    img.reshape(-1, 3)[hit] = cols[hit]
    return img, hit.reshape(h, w)


# ------------------------------------------------------------- splat rendering
def make_renderer(splats: dict, w=W, h=H, fovx=FOVX, fovy=FOVY):
    r = SplatRenderer(splats=splats, device="cuda")
    r.init_cameras({"cam": {"res": (h, w), "fovx": fovx, "fovy": fovy}})
    return r


def render_splat(r, R, eye):
    out = {}
    for bg in (0.0, 1.0):
        r.bg_color = torch.tensor([bg] * 3, device="cuda").float()
        with torch.no_grad():
            out[bg] = r.render_raw({"cam": {"pos": np.asarray(eye, float), "rot": np.asarray(R, float)}})["cam"].cpu().numpy()
    alpha = np.clip(1 - (out[1.0] - out[0.0]).mean(-1), 0, 1)
    img = out[0.0] + 0.5 * (1 - alpha[..., None])  # composite on grey
    return np.clip(img, 0, 1), alpha


def to_u8(x):
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def label(img_u8, text):
    im = Image.fromarray(img_u8)
    ImageDraw.Draw(im).text((6, 4), text, fill=(255, 255, 0))
    return np.asarray(im)


# ------------------------------------------------------------------- tests
def test_objects(envs):
    OUT.mkdir(parents=True, exist_ok=True)
    report = []
    for env in envs:
        stage = Usd.Stage.Open(str(m2s.HUB / env / "scene.usda"))
        for child in stage.GetPrimAtPath("/World").GetChildren():
            if not UsdPhysics.RigidBodyAPI(child):
                continue
            name = child.GetName()
            sp = m2s.HUB / env / "assets" / name / "splat.ply"
            if not (sp.parent / "splat_meta_ws2.json").exists():
                continue
            W2B, _, _ = m2s.body_from_world(child)
            parts = m2s.collect_triangles(child, W2B, dilate_textures=False)
            verts = np.concatenate([p["tri"].reshape(-1, 3) for p in parts])
            r = make_renderer({name: sp})
            xyz = r.big_model._xyz.detach().cpu().numpy()
            mesh_bb = np.stack([verts.min(0), verts.max(0)])
            spl_bb = np.stack([xyz.min(0), xyz.max(0)])
            center = mesh_bb.mean(0)
            radius = np.linalg.norm(mesh_bb[1] - mesh_bb[0]) / 2
            rows, ious = [], []
            for vi, (R, eye) in enumerate(views_for(center, radius)):
                simg, salpha = render_splat(r, R, eye)
                mimg, mhit = raster_mesh(parts, R, eye)
                smask = salpha > 0.5
                iou = (smask & mhit).sum() / max((smask | mhit).sum(), 1)
                ious.append(iou)
                # colour error on pixels covered by both
                both = smask & mhit
                cerr = np.abs(simg[both] - mimg[both]).mean() if both.any() else float("nan")
                rows.append(np.concatenate([label(to_u8(simg), f"{env}/{name} splat v{vi}"),
                                            label(to_u8(mimg), f"mesh v{vi} IoU {iou:.3f} |dRGB| {cerr:.3f}")], 1))
            grid = np.concatenate([np.concatenate(rows[:2], 1), np.concatenate(rows[2:], 1)], 0)
            fn = OUT / f"{env}__{name}.png"
            Image.fromarray(grid).resize((grid.shape[1] // 2, grid.shape[0] // 2), Image.LANCZOS).save(fn)
            Image.fromarray(rows[0]).save(OUT / f"{env}__{name}__v0_full.png")
            rec = dict(env=env, obj=name, n=len(xyz), mesh_bbox=mesh_bb.round(4).tolist(),
                       splat_bbox=spl_bb.round(4).tolist(),
                       bbox_max_abs_diff_mm=float(np.abs(mesh_bb - spl_bb).max() * 1e3),
                       silhouette_iou=[round(float(i), 3) for i in ious], png=str(fn))
            print(json.dumps(rec))
            report.append(rec)
            del r
            torch.cuda.empty_cache()
    (OUT / "report_ws2.json").write_text(json.dumps(report, indent=1))
    return report


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def test_scene(env="food_bussing", ic_index=0):
    """Pose object splats in the scene splat exactly like transform_sim_to_splat does
    (root pos/quat for every rigid object incl. the static scene) and render from the
    scene's external_cam plus a closer top-down-ish view."""
    stage = Usd.Stage.Open(str(m2s.HUB / env / "scene.usda"))
    splats, poses_usd = {}, {}
    for child in stage.GetPrimAtPath("/World").GetChildren():
        if not UsdPhysics.RigidBodyAPI(child):
            continue
        name = child.GetName()
        sp = m2s.HUB / env / "assets" / name / "splat.ply"
        if sp.exists():
            splats[name] = sp
            _, pos, quat = m2s.body_from_world(child)
            poses_usd[name] = (pos, quat)
    ics = json.loads((m2s.HUB / env / "initial_conditions.json").read_text())["poses"]
    cam = stage.GetPrimAtPath("/World/external_cam")
    from pxr import UsdGeom
    M = np.array(UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(0))
    R_gl, eye_ext = M[:3, :3].T, M[3, :3]  # row-vector -> columns = cam axes in world
    R_gl = R_gl / np.linalg.norm(R_gl, axis=0, keepdims=True)
    R_ext = R_gl @ np.diag([1, -1, -1])  # OpenGL (-z fwd, +y up) -> COLMAP
    gcam = UsdGeom.Camera(cam)
    f, ha, va = gcam.GetFocalLengthAttr().Get(), gcam.GetHorizontalApertureAttr().Get(), gcam.GetVerticalApertureAttr().Get()
    fovx, fovy = 2 * math.atan(ha / (2 * f)), 2 * math.atan(va / (2 * f))
    w, h = 960, int(round(960 * va / ha))
    r = SplatRenderer(splats=splats, device="cuda")
    r.init_cameras({"ext": {"res": (h, w), "fovx": fovx, "fovy": fovy},
                    "close": {"res": (480, 640), "fovx": FOVX, "fovy": FOVY}})
    objs = [n for n in splats if "static" not in n]
    centre = np.mean([poses_usd[n][0] for n in objs], 0)
    R_close = look_at(centre + np.array([-0.35, 0.25, 0.30]), centre)
    eye_close = centre + np.array([-0.35, 0.25, 0.30])
    outs = []
    for tag, poses in [("usd_init", poses_usd),
                       (f"ic{ic_index}", {**poses_usd, **{k: (np.array(v[:3]), np.array(v[3:])) for k, v in ics[ic_index].items()}})]:
        tf = {k: (torch.tensor(p, dtype=torch.float32), torch.tensor(q, dtype=torch.float32)) for k, (p, q) in poses.items()}
        r.transform_many(tf)
        with torch.no_grad():
            imgs = r.render_raw({"ext": {"pos": eye_ext, "rot": R_ext}, "close": {"pos": eye_close, "rot": R_close}})
        for cname, im in imgs.items():
            fn = OUT / f"scene_{env}_{tag}_{cname}.png"
            Image.fromarray(to_u8(im.cpu().numpy())).save(fn)
            outs.append(str(fn))
            print("wrote", fn)
    # also: the same close view with object splats removed, to judge contact/occlusion
    return outs


def test_closeup(targets=(("food_bussing", "cup"), ("food_bussing", "battery1"),
                          ("organize_tools", "scissor"), ("organize_tools", "tape"))):
    """Wrist-camera-like close-up at 1280x720 (fovx 90 deg, ~8-12 cm away): the worst
    case for surfel density. Splat (top) vs mesh (bottom)."""
    w, h = 1280, 720
    fovx = math.radians(90)
    fovy = 2 * math.atan(math.tan(fovx / 2) * h / w)
    for env, name in targets:
        stage = Usd.Stage.Open(str(m2s.HUB / env / "scene.usda"))
        child = stage.GetPrimAtPath(f"/World/{name}")
        W2B, _, _ = m2s.body_from_world(child)
        parts = m2s.collect_triangles(child, W2B, dilate_textures=False)
        verts = np.concatenate([p["tri"].reshape(-1, 3) for p in parts])
        c = (verts.min(0) + verts.max(0)) / 2
        ext = np.linalg.norm(verts.max(0) - verts.min(0))
        eye = c + np.array([0.6, -0.5, 0.6]) / np.linalg.norm([0.6, -0.5, 0.6]) * max(0.08, 0.7 * ext)
        R = look_at(eye, c)
        r = make_renderer({name: m2s.HUB / env / "assets" / name / "splat.ply"}, w, h, fovx, fovy)
        simg, _ = render_splat(r, R, eye)
        mimg, _ = raster_mesh(parts, R, eye, fovx, fovy, w, h)
        fn = OUT / f"closeup_{env}__{name}.png"
        Image.fromarray(np.concatenate([label(to_u8(simg), f"splat 1280x720 {name}"), label(to_u8(mimg), "mesh")], 0)).save(fn)
        print("wrote", fn)
        del r
        torch.cuda.empty_cache()


def test_timing(env="food_bussing", reps=20):
    """Render cost with SplatRenderer alone: scene splat + robot link splats, with and
    without the generated object splats; two 1280x720 cameras (scene external_cam and
    a wrist-like close view ~15 cm above the objects). Objects at their USD poses."""
    import time
    from pxr import UsdGeom
    stage = Usd.Stage.Open(str(m2s.HUB / env / "scene.usda"))
    static, objs, poses = {}, {}, {}
    for child in stage.GetPrimAtPath("/World").GetChildren():
        if not UsdPhysics.RigidBodyAPI(child):
            continue
        name = child.GetName()
        sp = m2s.HUB / env / "assets" / name / "splat.ply"
        if not sp.exists():
            continue
        (objs if (sp.parent / "splat_meta_ws2.json").exists() else static)[name] = sp
        _, pos, quat = m2s.body_from_world(child)
        poses[name] = (torch.tensor(pos, dtype=torch.float32), torch.tensor(quat, dtype=torch.float32))
    robot = {p.stem: p for p in sorted((m2s.HUB / "nvidia_droid" / "SEGMENTED").glob("*.ply"))}
    cam = stage.GetPrimAtPath("/World/external_cam")
    if cam.IsValid():
        M = np.array(UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(0))
        R_ext = (M[:3, :3].T / np.linalg.norm(M[:3, :3].T, axis=0)) @ np.diag([1, -1, -1])
        eye_ext = M[3, :3]
    else:  # droid_cfg default external_cam offset (opengl convention)
        eye_ext = np.array([-0.01, -0.33, 0.48])
        R_gl = quat_to_R(np.array([0.76, 0.43, -0.24, -0.42]) / np.linalg.norm([0.76, 0.43, -0.24, -0.42]))
        R_ext = R_gl @ np.diag([1, -1, -1])
    centre = np.mean([poses[n][0].numpy() for n in objs], 0) if objs else np.array([0.45, 0, 0])
    eye_w = centre + np.array([-0.05, 0.0, 0.15])
    R_w = look_at(eye_w, centre)
    fx = 2 * math.atan(2.5452 / (2 * 1.0476))
    fy = 2 * math.atan(1.4721 / (2 * 1.0476))
    res = {}
    for tag, spl in [("scene+robot", {**static}), ("scene+robot+objects", {**static, **objs})]:
        r = SplatRenderer(splats=spl, device="cuda")
        r.add_splats(robot)
        r.init_cameras({"ext": {"res": (720, 1280), "fovx": fx, "fovy": fy},
                        "wrist": {"res": (720, 1280), "fovx": fx, "fovy": fy}})
        r.transform_many({k: v for k, v in poses.items() if k in spl})
        ex = {"ext": {"pos": eye_ext, "rot": R_ext}, "wrist": {"pos": eye_w, "rot": R_w}}
        with torch.no_grad():
            for _ in range(3):
                r.render_raw(ex)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(reps):
                imgs = r.render_raw(ex)
            torch.cuda.synchronize()
        ms = (time.time() - t0) / reps * 1e3
        # same, but with autograd enabled (as the stock env calls it) and incl. the
        # per-step transform_many of all posed splats
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            imgs = r.render_raw(ex)
        torch.cuda.synchronize()
        ms_grad = (time.time() - t0) / reps * 1e3
        t0 = time.time()
        for _ in range(reps):
            r.transform_many({k: v for k, v in poses.items() if k in spl})
        torch.cuda.synchronize()
        ms_tf = (time.time() - t0) / reps * 1e3
        n = r.big_model._xyz.shape[0]
        res[tag] = (n, ms, ms_grad, ms_tf)
        print(f"[timing {env}] {tag}: {n} gaussians, {ms:.1f} ms per 2-camera render (no_grad), "
              f"{ms_grad:.1f} ms with autograd, transform_many {ms_tf:.1f} ms")
        Image.fromarray(to_u8(imgs["wrist"].detach().cpu().numpy())).save(OUT / f"timing_{env}_{tag.replace('+', '_')}_wrist.png")
        del r
        torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", nargs="*", default=None, help="envs to time (render cost)")
    ap.add_argument("--closeup", action="store_true")
    ap.add_argument("--envs", nargs="*", default=m2s.ENVS)
    ap.add_argument("--scene-only", action="store_true")
    ap.add_argument("--no-scene", action="store_true")
    a = ap.parse_args()
    if a.timing is not None:
        OUT.mkdir(parents=True, exist_ok=True)
        for e in (a.timing or ["food_bussing", "organize_tools"]):
            test_timing(e)
        sys.exit(0)
    if a.closeup:
        OUT.mkdir(parents=True, exist_ok=True)
        test_closeup()
        sys.exit(0)
    if not a.scene_only:
        test_objects(a.envs)
    if not a.no_scene:
        test_scene("food_bussing")
