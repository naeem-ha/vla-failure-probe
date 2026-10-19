"""Workstream 2: convert PolaRiS mesh-only rigid objects into 2DGS surfel splats.

Stock PolaRiS renders a rigid object with its own splat path when
``<env>/assets/<rigid prim name>/splat.ply`` exists (see
``ManagerBasedRLSplatEnv.setup_splat_world_and_robot_views``); otherwise the object
is drawn by Isaac RTX, which does not work in WSL2. This script generates those
``splat.ply`` files from the USD meshes, inside the PolaRiS-Hub-wsl mirror.

Frame convention (matches ``SplatRenderer.transform_many``):
    x_world = R(q_root) @ x_splat + p_root
where (p_root, q_root) is ``scene[name].data.root_state_w[:, :7]``, i.e. the rigid
prim's world pose WITHOUT scale. So splat points are expressed in the rigid prim's
body frame and include everything below the pose: the prim's own xformOp:scale,
the payload's internal Xforms (e.g. Y-up -> Z-up rotation of the mesh), and the
mesh points. We compute that with pxr directly from the scene.usda composition:
    x_body = inverse(T_root_noscale) * LocalToWorld(mesh) * p_mesh
Units are metres (all PolaRiS scenes have metersPerUnit = 1, upAxis = Z).

Note the lookup key is the RIGID PRIM NAME in scene.usda (e.g. food_bussing
"bowl", "battery1", "ice_cream_"), not the asset folder name ("yellow_bowl", ...).

PLY layout matches existing PolaRiS object splats (float32):
    x y z nx ny nz f_dc_0..2 f_rest_0..44 opacity scale_0 scale_1 rot_0..3
SH degree 3 (f_rest all zero), opacity pre-sigmoid, scales log, rot = (w,x,y,z)
whose rotation matrix has the surfel normal as its 3rd column.

Usage (inside WSL; needs Isaac's pxr on PYTHONPATH -> use mesh_to_surfels_ws2.sh):
    bash ~/spais-demo/work/polaris_wsl/mesh_to_surfels_ws2.sh            # all envs
    bash ~/spais-demo/work/polaris_wsl/mesh_to_surfels_ws2.sh --envs food_bussing
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

C0 = 0.28209479177387814
HUB = Path(os.path.expanduser("~/spais-demo/polaris/PolaRiS-Hub-wsl"))
ENVS = [
    "food_bussing",
    "organize_tools",
    "pan_clean",
    "move_latte_cup",
    "block_stack_kitchen",
    "tape_into_container",
]


# --------------------------------------------------------------------------- USD
def gf_to_np(m: Gf.Matrix4d) -> np.ndarray:
    """Gf matrices are row-vector convention: p' = p @ M."""
    return np.array(m, dtype=np.float64)


def body_from_world(root: Usd.Prim) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (4x4 row-vector world->body matrix, pos, quat_wxyz) of the rigid prim
    pose without scale (what PhysX / Isaac Lab report as root_state_w)."""
    M = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    tf = Gf.Transform(M)
    q = tf.GetRotation().GetQuat()
    t = tf.GetTranslation()
    T = Gf.Matrix4d(1.0)
    T.SetRotateOnly(Gf.Rotation(q))
    T.SetTranslateOnly(t)
    quat = np.array([q.GetReal(), *q.GetImaginary()], dtype=np.float64)
    return gf_to_np(T.GetInverse()), np.array(t, dtype=np.float64), quat


def read_image(asset_path) -> np.ndarray | None:
    """Load an image referenced by a USD asset path, including files inside .usdz."""
    rp = asset_path.resolvedPath or ""
    if not rp:
        return None
    try:
        if rp.endswith("]") and "[" in rp:  # package-relative: /x/mesh.usdz[0/texture.jpg]
            pkg, inner = rp[:-1].split("[", 1)
            with zipfile.ZipFile(pkg) as z:
                data = z.read(inner)
            img = Image.open(io.BytesIO(data))
        else:
            img = Image.open(rp)
        return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    except Exception as e:  # noqa: BLE001
        print(f"    [warn] could not load texture {rp}: {e}")
        return None


def material_info(mesh_prim: Usd.Prim):
    """Find the bound material's base-colour texture (and a constant fallback)."""
    mat, _ = UsdShade.MaterialBindingAPI(mesh_prim).ComputeBoundMaterial()
    tex, const, st_name, tex_path = None, None, "st", None
    if not mat:
        return tex, const, st_name, tex_path
    candidates = []
    for p in Usd.PrimRange(mat.GetPrim()):
        if not p.IsA(UsdShade.Shader):
            continue
        sh = UsdShade.Shader(p)
        sid = sh.GetIdAttr().Get()
        if sid == "UsdPrimvarReader_float2":
            vn = sh.GetInput("varname")
            if vn and vn.Get():
                st_name = str(vn.Get())
        for inp in sh.GetInputs():
            v = inp.Get()
            if hasattr(v, "resolvedPath") and v.path:
                name = inp.GetBaseName().lower()
                score = 0 if any(k in name for k in ("diffuse", "base", "albedo")) else 1
                candidates.append((score, v))
            if inp.GetBaseName() in ("diffuseColor", "diffuse_color_constant") and not inp.HasConnectedSource():
                if v is not None:
                    const = np.array(v, dtype=np.float32)
    for _, v in sorted(candidates, key=lambda c: c[0]):
        tex = read_image(v)
        if tex is not None:
            tex_path = v.resolvedPath
            break
    return tex, const, st_name, tex_path


def primvar_facevarying(mesh: UsdGeom.Mesh, name: str, fvc, fvi) -> np.ndarray | None:
    """Return primvar values expanded to face-varying (one per face-vertex)."""
    pv = UsdGeom.PrimvarsAPI(mesh.GetPrim()).GetPrimvar(name)
    if not pv or not pv.HasValue():
        return None
    vals = pv.ComputeFlattened()
    if vals is None:
        return None
    vals = np.array(vals, dtype=np.float64)
    if vals.ndim == 1:
        vals = vals[:, None]
    interp = pv.GetInterpolation()
    n_fv = len(fvi)
    if interp == UsdGeom.Tokens.faceVarying and len(vals) == n_fv:
        return vals
    if interp in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying):
        return vals[fvi]
    if interp == UsdGeom.Tokens.uniform and len(vals) == len(fvc):
        return np.repeat(vals, fvc, axis=0)
    if interp == UsdGeom.Tokens.constant or len(vals) == 1:
        return np.repeat(vals[:1], n_fv, axis=0)
    return None


def dilate_texture(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Fill texels outside every UV island with the nearest island texel, so surfels
    whose centre lies near a UV seam do not pick up the atlas background colour."""
    from PIL import ImageDraw
    from scipy.ndimage import binary_dilation, distance_transform_edt

    H, W, _ = tex.shape
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    px = np.stack([(uv[..., 0] % 1.0) * W, (1.0 - (uv[..., 1] % 1.0)) * H], -1)
    for t in px:
        draw.polygon([tuple(t[0]), tuple(t[1]), tuple(t[2])], fill=255)
    cov = np.asarray(mask) > 0
    cov = cov & ~binary_dilation(~cov, iterations=1)  # drop half-covered border texels
    if cov.sum() < 0.01 * cov.size:
        return tex
    _, (iy, ix) = distance_transform_edt(~cov, return_indices=True)
    return tex[iy, ix]


def collect_triangles(root: Usd.Prim, W2B: np.ndarray, dilate_textures: bool = True):
    """Gather all visible render meshes under a rigid prim as body-frame triangles.

    Returns list of dicts with tri (T,3,3) body-frame positions, uv (T,3,2) or None,
    col (T,3,3) or None (displayColor), tex image or None, const colour or None.
    """
    parts = []
    for p in Usd.PrimRange(root):
        if not p.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(p)
        img = UsdGeom.Imageable(p)
        if img.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        if img.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
            continue
        pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
        fvc = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        fvi = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if len(pts) == 0 or len(fvc) == 0:
            continue
        M = gf_to_np(mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default())) @ W2B
        P = np.c_[pts, np.ones(len(pts))] @ M
        P = P[:, :3]
        # fan-triangulate polygons -> face-vertex index triplets
        starts = np.r_[0, np.cumsum(fvc)[:-1]]
        tri_fv = []
        for k in range(3, int(fvc.max()) + 1):
            sel = fvc >= k
            s = starts[sel]
            tri_fv.append(np.stack([s, s + k - 2, s + k - 1], 1))
        tri_fv = np.concatenate(tri_fv, 0)
        tri = P[fvi[tri_fv]]  # (T,3,3)

        tex, const, st_name, tex_path = material_info(p)
        uv = primvar_facevarying(mesh, st_name, fvc, fvi)
        if uv is None and st_name != "st":
            uv = primvar_facevarying(mesh, "st", fvc, fvi)
        uv = uv[tri_fv][..., :2] if (uv is not None and tex is not None) else None
        if uv is not None and dilate_textures:
            tex = dilate_texture(tex, uv)
        dcol = primvar_facevarying(mesh, "displayColor", fvc, fvi)
        dcol = dcol[tri_fv][..., :3] if dcol is not None else None
        parts.append(dict(path=str(p.GetPath()), tri=tri, uv=uv, col=dcol, tex=tex,
                          const=const, tex_path=tex_path))
    return parts


# ---------------------------------------------------------------------- sampling
def sample_texture(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear lookup; USD st origin is bottom-left, wrap = repeat."""
    H, W, _ = tex.shape
    x = (uv[:, 0] % 1.0) * W - 0.5
    y = (1.0 - (uv[:, 1] % 1.0)) * H - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    x0m, x1m = x0 % W, (x0 + 1) % W
    y0m, y1m = np.clip(y0, 0, H - 1), np.clip(y0 + 1, 0, H - 1)
    c = (tex[y0m, x0m] * (1 - fx) * (1 - fy) + tex[y0m, x1m] * fx * (1 - fy)
         + tex[y1m, x0m] * (1 - fx) * fy + tex[y1m, x1m] * fx * fy)
    return c


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Batch rotation matrices (N,3,3) -> quaternions (N,4) w,x,y,z."""
    m = R
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = np.zeros((len(m), 4))
    c1 = tr > 0
    s = np.sqrt(np.maximum(tr[c1] + 1.0, 1e-12)) * 2
    q[c1, 0] = 0.25 * s
    q[c1, 1] = (m[c1, 2, 1] - m[c1, 1, 2]) / s
    q[c1, 2] = (m[c1, 0, 2] - m[c1, 2, 0]) / s
    q[c1, 3] = (m[c1, 1, 0] - m[c1, 0, 1]) / s
    rest = ~c1
    c2 = rest & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    s = np.sqrt(np.maximum(1.0 + m[c2, 0, 0] - m[c2, 1, 1] - m[c2, 2, 2], 1e-12)) * 2
    q[c2, 0] = (m[c2, 2, 1] - m[c2, 1, 2]) / s
    q[c2, 1] = 0.25 * s
    q[c2, 2] = (m[c2, 0, 1] + m[c2, 1, 0]) / s
    q[c2, 3] = (m[c2, 0, 2] + m[c2, 2, 0]) / s
    c3 = rest & ~c2 & (m[:, 1, 1] > m[:, 2, 2])
    s = np.sqrt(np.maximum(1.0 + m[c3, 1, 1] - m[c3, 0, 0] - m[c3, 2, 2], 1e-12)) * 2
    q[c3, 0] = (m[c3, 0, 2] - m[c3, 2, 0]) / s
    q[c3, 1] = (m[c3, 0, 1] + m[c3, 1, 0]) / s
    q[c3, 2] = 0.25 * s
    q[c3, 3] = (m[c3, 1, 2] + m[c3, 2, 1]) / s
    c4 = rest & ~c2 & ~c3
    s = np.sqrt(np.maximum(1.0 + m[c4, 2, 2] - m[c4, 0, 0] - m[c4, 1, 1], 1e-12)) * 2
    q[c4, 0] = (m[c4, 1, 0] - m[c4, 0, 1]) / s
    q[c4, 1] = (m[c4, 0, 2] + m[c4, 2, 0]) / s
    q[c4, 2] = (m[c4, 1, 2] + m[c4, 2, 1]) / s
    q[c4, 3] = 0.25 * s
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5**0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def visible_faces(tri: np.ndarray, n_views: int = 96, max_res: int = 1400) -> np.ndarray:
    """Mark faces seen (z-buffer winner in >=1 pixel) in orthographic views from
    n_views directions around the object.

    Why: many PolaRiS object meshes (both the 5k-face *_csm usdz assets and the
    textured mesh.usd scans) are double-walled -- an inner shell ~0.2 mm inside the
    outer surface, textured dark. A z-buffer hides it, but surfels are alpha-blended
    in centre-depth order, so inner-shell surfels interleave with outer ones and show
    up as dark specks. Sampling only externally visible faces removes them.
    """
    T = len(tri)
    seen = np.zeros(T, bool)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    ext = np.linalg.norm(tri.reshape(-1, 3).max(0) - tri.reshape(-1, 3).min(0))
    pix = max(0.4 * np.sqrt(np.median(area[area > 0])), ext / max_res)
    for d in fibonacci_sphere(n_views):
        a = np.cross(d, [0.0, 0.0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(d, a)
        px = (tri @ a).astype(np.float32)
        py = (tri @ b).astype(np.float32)
        z = (tri @ d).astype(np.float32)  # larger = farther along view direction (camera at -inf*d)
        px = (px - px.min()) / pix
        py = (py - py.min()) / pix
        x0 = np.floor(px.min(1)).astype(np.int64)
        x1 = np.ceil(px.max(1)).astype(np.int64)
        y0 = np.floor(py.min(1)).astype(np.int64)
        y1 = np.ceil(py.max(1)).astype(np.int64)
        Wd = int(x1.max()) + 2
        bmax = np.maximum(x1 - x0, y1 - y0) + 1
        P_all, Z_all, F_all = [], [], []
        for lo, hi in [(0, 2), (2, 4), (4, 8), (8, 16), (16, 64), (64, 10**9)]:
            sel = np.where((bmax > lo) & (bmax <= hi))[0]
            if len(sel) == 0:
                continue
            S = int(min(hi, bmax[sel].max()))
            for chunk in np.array_split(sel, max(1, len(sel) * S * S // 400_000 + 1)):
                gx, gy = np.meshgrid(np.arange(S), np.arange(S))
                X = x0[chunk, None] + gx.ravel()[None] + 0.5  # pixel centres
                Y = y0[chunk, None] + gy.ravel()[None] + 0.5
                ax, ay = px[chunk, 0:1], py[chunk, 0:1]
                bx, by = px[chunk, 1:2], py[chunk, 1:2]
                cx, cy = px[chunk, 2:3], py[chunk, 2:3]
                den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
                ok_den = np.abs(den) > 1e-12
                den = np.where(ok_den, den, 1.0)
                l0 = ((by - cy) * (X - cx) + (cx - bx) * (Y - cy)) / den
                l1 = ((cy - ay) * (X - cx) + (ax - cx) * (Y - cy)) / den
                l2 = 1 - l0 - l1
                m = ok_den & (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
                ti, pi = np.nonzero(m)
                if len(ti) == 0:
                    continue
                fz = z[chunk[ti]]
                depth = l0[ti, pi] * fz[:, 0] + l1[ti, pi] * fz[:, 1] + l2[ti, pi] * fz[:, 2]
                P_all.append((Y[ti, pi] - 0.5).astype(np.int64) * Wd + (X[ti, pi] - 0.5).astype(np.int64))
                Z_all.append(depth)
                F_all.append(chunk[ti])
        if not P_all:
            continue
        P, Z, F = np.concatenate(P_all), np.concatenate(Z_all), np.concatenate(F_all)
        order = np.lexsort((Z, P))
        P, F = P[order], F[order]
        _, first = np.unique(P, return_index=True)
        seen[F[first]] = True
    return seen


def surfelize(parts, spacing, min_pts, max_pts, k_scale, opacity, seed=0, supersample=8,
              cull_hidden=True):
    rng = np.random.default_rng(seed)
    tri = np.concatenate([p["tri"] for p in parts], 0)
    part_id = np.concatenate([np.full(len(p["tri"]), i) for i, p in enumerate(parts)])
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    cr = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    n_hidden = 0
    if cull_hidden:
        vis = visible_faces(tri)
        n_hidden = int((~vis).sum())
        area = np.where(vis, area, 0.0)
    total = area.sum()
    N = int(np.clip(total / spacing**2, min_pts, max_pts))

    # stratified (systematic) area-weighted face selection -> even coverage
    cdf = np.cumsum(area) / total
    u = (np.arange(N) + rng.random(N)) / N
    f = np.minimum(np.searchsorted(cdf, u), len(area) - 1)  # nondecreasing
    # within each face, place its samples on a randomly shifted R2 low-discrepancy
    # sequence instead of i.i.d. points -> fewer clumps/holes (less see-through speckle)
    first = np.searchsorted(f, f, side="left")
    rank = np.arange(N) - first
    g = 1.32471795724474602596
    shift = rng.random((len(area), 2))
    s1 = (shift[f, 0] + rank / g) % 1.0
    s2 = (shift[f, 1] + rank / g**2) % 1.0
    r1 = np.sqrt(s1)
    r2 = s2
    b = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], 1)  # barycentrics
    xyz = np.einsum("nk,nkd->nd", b, tri[f])

    n = cr[f] / np.maximum(np.linalg.norm(cr[f], axis=1, keepdims=True), 1e-20)
    t1 = e1[f] / np.maximum(np.linalg.norm(e1[f], axis=1, keepdims=True), 1e-20)
    # random in-plane spin (irrelevant for isotropic disks, avoids aligned artefacts)
    ang = rng.random(N) * 2 * np.pi
    t2 = np.cross(n, t1)
    t1, t2 = (np.cos(ang)[:, None] * t1 + np.sin(ang)[:, None] * t2,
              -np.sin(ang)[:, None] * t1 + np.cos(ang)[:, None] * t2)
    R = np.stack([t1, t2, n], axis=2)  # columns: tangent u, tangent v, normal
    quat = rotmat_to_quat_wxyz(R)

    sigma = k_scale * np.sqrt(total / N)
    rgb = np.full((N, 3), 0.6)
    for i, p in enumerate(parts):
        sel = part_id[f] == i
        if not sel.any():
            continue
        fl = f[sel] - np.searchsorted(part_id, i)  # local face index in part
        if p["uv"] is not None and p["tex"] is not None:
            uvs = np.einsum("nk,nkd->nd", b[sel], p["uv"][fl])
            # Prefilter: average the texture over the surfel's Gaussian footprint
            # (point-sampling one texel per ~1 mm disk turns texture grain into specks).
            # J maps a 3D in-plane offset to a UV offset for each face.
            fg = f[sel]
            E = np.stack([e1[fg], e2[fg]], 2)  # (n,3,2)
            Ep = np.linalg.pinv(E)  # (n,2,3)
            uvt = p["uv"][fl]
            dUV = np.stack([uvt[:, 1] - uvt[:, 0], uvt[:, 2] - uvt[:, 0]], 2)  # (n,2,2)
            J = dUV @ Ep  # (n,2,3)
            acc = sample_texture(p["tex"], uvs)
            K = supersample
            for _ in range(K - 1):
                g = rng.normal(size=(len(fg), 2)) * sigma
                off = g[:, :1] * t1[sel] + g[:, 1:] * t2[sel]
                acc += sample_texture(p["tex"], uvs + np.einsum("nij,nj->ni", J, off))
            rgb[sel] = acc / K
        elif p["col"] is not None:
            rgb[sel] = np.einsum("nk,nkd->nd", b[sel], p["col"][fl])
        elif p["const"] is not None:
            rgb[sel] = p["const"][:3]
    rgb = np.clip(rgb, 0.0, 1.0)
    return dict(xyz=xyz, normal=n, rgb=rgb, quat=quat, N=N, area=total, sigma=sigma,
                opacity=opacity, n_faces=len(tri), n_hidden_faces=n_hidden)


def write_ply(path: Path, s: dict):
    N = s["N"]
    names = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
             + [f"f_rest_{i}" for i in range(45)]
             + ["opacity", "scale_0", "scale_1", "rot_0", "rot_1", "rot_2", "rot_3"])
    arr = np.zeros(N, dtype=[(nm, "f4") for nm in names])
    for i, c in enumerate("xyz"):
        arr[c] = s["xyz"][:, i]
        arr["n" + c] = s["normal"][:, i]
    dc = (s["rgb"] - 0.5) / C0
    for i in range(3):
        arr[f"f_dc_{i}"] = dc[:, i]
    op = float(np.clip(s["opacity"], 1e-4, 1 - 1e-4))
    arr["opacity"] = np.log(op / (1 - op))
    arr["scale_0"] = np.log(s["sigma"])
    arr["scale_1"] = np.log(s["sigma"])
    for i in range(4):
        arr[f"rot_{i}"] = s["quat"][:, i]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".ply.tmp")
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(tmp))
    os.replace(tmp, path)


# -------------------------------------------------------------------------- main
def convert_env(env: str, args) -> list[dict]:
    scene = HUB / env / "scene.usda"
    stage = Usd.Stage.Open(str(scene))
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    up = UsdGeom.GetStageUpAxis(stage)
    assert abs(mpu - 1.0) < 1e-9 and up == "Z", f"{env}: unexpected units {mpu} {up}"
    out = []
    for child in stage.GetPrimAtPath("/World").GetChildren():
        if not UsdPhysics.RigidBodyAPI(child):
            continue
        name = child.GetName()
        if args.objects and name not in args.objects:
            continue
        dst = HUB / env / "assets" / name / "splat.ply"
        if dst.is_symlink() or (dst.exists() and not (dst.parent / "splat_meta_ws2.json").exists()):
            print(f"[{env}] {name}: has a stock splat, skipping")
            continue
        t0 = time.time()
        W2B, pos, quat = body_from_world(child)
        parts = collect_triangles(child, W2B)
        if not parts:
            print(f"[{env}] {name}: no visible meshes, skipping")
            continue
        s = surfelize(parts, args.spacing, args.min_pts, args.max_pts, args.k, args.opacity,
                      seed=args.seed, supersample=args.supersample,
                      cull_hidden=not args.no_cull)
        write_ply(dst, s)
        bb = np.stack([s["xyz"].min(0), s["xyz"].max(0)])
        meta = dict(
            env=env, rigid_prim=f"/World/{name}", n_points=s["N"], surface_area_m2=s["area"],
            sigma_m=s["sigma"], opacity=s["opacity"], spacing_m=args.spacing,
            n_faces=s["n_faces"], n_hidden_faces_culled=s["n_hidden_faces"],
            frame="rigid-prim body frame (world pose without scale), metres, Z-up",
            usd_init_pos=pos.tolist(), usd_init_quat_wxyz=quat.tolist(),
            body_bbox_min=bb[0].tolist(), body_bbox_max=bb[1].tolist(),
            meshes=[dict(path=p["path"], n_tri=len(p["tri"]), texture=p["tex_path"],
                         has_uv=p["uv"] is not None, has_displayColor=p["col"] is not None)
                    for p in parts],
            generator="spais-demo/polaris_wsl/mesh_to_surfels.py",
        )
        (dst.parent / "splat_meta_ws2.json").write_text(json.dumps(meta, indent=1))
        sz = dst.stat().st_size / 1e6
        print(f"[{env}] {name}: {s['N']} surfels, area {s['area']*1e4:.1f} cm2, "
              f"sigma {s['sigma']*1e3:.2f} mm, hidden faces {s['n_hidden_faces']}/{s['n_faces']}, tex={'yes' if parts[0]['tex'] is not None else 'no'}, "
              f"{sz:.1f} MB, bbox {np.round(bb, 3).tolist()} ({time.time()-t0:.1f}s)")
        out.append(dict(meta, file=str(dst), size_mb=sz))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", nargs="*", default=ENVS)
    ap.add_argument("--objects", nargs="*", default=None, help="rigid prim names to (re)build")
    # Density presets. Render cost scales with surfel count x screen coverage, so the
    # default ("fast") keeps per-env object totals ~100-250k; sigma grows as spacing
    # grows (sigma = k * sqrt(area/N)) so the surface stays closed.
    PRESETS = {"fast": (0.001, 15000, 60000, 0.9), "medium": (0.0008, 25000, 120000, 0.85),
               "high": (0.0005, 50000, 400000, 0.8)}
    ap.add_argument("--quality", choices=list(PRESETS), default="fast")
    ap.add_argument("--spacing", type=float, default=None, help="target surfel spacing [m] (overrides preset)")
    ap.add_argument("--min-pts", type=int, default=None, help="per-object minimum count")
    ap.add_argument("--max-pts", type=int, default=None, help="per-object cap")
    ap.add_argument("--k", type=float, default=None, help="surfel sigma = k * sqrt(area/N)")
    ap.add_argument("--opacity", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cull", action="store_true", help="keep faces not visible from outside")
    ap.add_argument("--supersample", type=int, default=8, help="texture samples per surfel footprint")
    args = ap.parse_args()
    for key, val in zip(("spacing", "min_pts", "max_pts", "k"), PRESETS[args.quality]):
        if getattr(args, key) is None:
            setattr(args, key, val)
    print(f"density: quality={args.quality} spacing={args.spacing*1e3:.2f}mm "
          f"min={args.min_pts} max={args.max_pts} k={args.k}")
    summary = []
    for env in args.envs:
        summary += convert_env(env, args)
    sp = HUB / "splats_ws2_summary.json"
    old = {m["file"]: m for m in json.loads(sp.read_text())} if sp.exists() else {}
    for m in summary:
        old[m["file"]] = {k: v for k, v in m.items()
                          if k in ("env", "rigid_prim", "n_points", "size_mb", "file", "sigma_m")}
    sp.write_text(json.dumps(sorted(old.values(), key=lambda m: m["file"]), indent=1))
    print(f"done: {len(summary)} object splats")


if __name__ == "__main__":
    main()
