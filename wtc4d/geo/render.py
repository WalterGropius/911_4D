"""Offscreen rendering of the scene prior for camera registration.

``render_view(c2w, intrinsics)`` returns ``depth``, ``instance_id`` and
``normals`` buffers for a camera in the world ENU frame.  Camera axes follow
the project convention (OpenCV: +X right, +Y down, +Z forward) and ``c2w`` is
camera-to-world.

Backends
--------
``raster``   pure-numpy z-buffer triangle rasteriser.  No system libraries, no
             GPU; this is the default and the one CI uses.
``pyrender`` pyrender/OpenGL offscreen (EGL or OSMesa).  Faster on machines
             that have a GL stack; selected by ``backend="pyrender"`` or by
             ``backend="auto"`` when a context can actually be created.
``raycast``  trimesh ray casting, one ray per pixel.  Exact but slow; useful
             as an independent cross-check in tests.

``pyrender`` is an optional extra (``pip install pyrender``); it is never
required.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wtc4d.schema.camera import CameraIntrinsics

_OPENCV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def _as_matrix(c2w) -> np.ndarray:
    m = np.asarray(c2w, dtype=np.float64)
    if m.shape == (16,):
        m = m.reshape(4, 4)
    if m.shape != (4, 4):
        raise ValueError(f"c2w must be 4x4 (or 16 floats), got {m.shape}")
    return m


def _scene_triangles(scene) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Flatten a trimesh Scene into ``(V, F, names)`` with a face->instance map.

    Returns vertices ``(N, 3)``, faces ``(M, 4)`` where the 4th column is the
    instance index, and the ordered instance names.
    """
    import trimesh

    if isinstance(scene, trimesh.Trimesh):
        v, f = np.asarray(scene.vertices), np.asarray(scene.faces)
        return v, np.column_stack([f, np.zeros(len(f), dtype=np.int64)]), ["mesh"]
    names: list[str] = []
    verts: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    off = 0
    for name, geom in scene.geometry.items():
        transform = np.eye(4)
        try:
            transform = scene.graph.get(name)[0]
        except Exception:  # noqa: BLE001 - geometry may not be in the graph
            pass
        v = trimesh.transform_points(np.asarray(geom.vertices, dtype=np.float64), transform)
        f = np.asarray(geom.faces, dtype=np.int64) + off
        idx = len(names)
        names.append(name)
        verts.append(v)
        faces.append(np.column_stack([f, np.full(len(f), idx, dtype=np.int64)]))
        off += len(v)
    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 4), dtype=np.int64), []
    return np.vstack(verts), np.vstack(faces), names


def render_view(
    c2w,
    intrinsics: CameraIntrinsics,
    scene=None,
    *,
    backend: str = "auto",
    far_m: float = 20000.0,
    year: int = 2001,
) -> dict[str, Any]:
    """Render depth / instance-id / normal buffers for one camera.

    Parameters
    ----------
    c2w
        4x4 camera-to-world matrix (or 16 floats), world ENU metres.
    intrinsics
        OpenCV pinhole intrinsics; ``dist`` is ignored (the prior is rendered
        as an ideal pinhole -- undistort your image instead).
    scene
        A ``trimesh.Scene`` or ``Trimesh``; defaults to ``load_scene(year)``.

    Returns
    -------
    dict with
        ``depth``        ``(H, W)`` float32, metres along the camera +Z axis,
                         ``inf`` where nothing was hit.
        ``instance_id``  ``(H, W)`` int32 index into ``instance_names``,
                         ``-1`` for background.
        ``normals``      ``(H, W, 3)`` float32 world-frame unit normals, zero
                         where nothing was hit.
        ``instance_names`` list[str], ``backend`` str.
    """
    if scene is None:
        from wtc4d.geo.scene import load_scene

        scene = load_scene(year=year)
    m = _as_matrix(c2w)
    if backend == "auto":
        backend = "pyrender" if _pyrender_available() else "raster"
    if backend == "pyrender":
        try:
            return _render_pyrender(m, intrinsics, scene, far_m)
        except Exception:  # noqa: BLE001 - fall back rather than fail the caller
            backend = "raster"
    if backend == "raycast":
        return _render_raycast(m, intrinsics, scene, far_m)
    return _render_raster(m, intrinsics, scene, far_m)


def _pyrender_available() -> bool:
    try:
        import pyrender  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


# --- numpy z-buffer rasteriser ----------------------------------------------
def _render_raster(c2w: np.ndarray, intr: CameraIntrinsics, scene, far_m: float) -> dict[str, Any]:
    V, F, names = _scene_triangles(scene)
    W, H = int(intr.width), int(intr.height)
    depth = np.full((H, W), np.inf, dtype=np.float64)
    inst = np.full((H, W), -1, dtype=np.int32)
    normals = np.zeros((H, W, 3), dtype=np.float64)
    if len(F) == 0:
        return _pack(depth, inst, normals, names, "raster")

    w2c = np.linalg.inv(c2w)
    cam = V @ w2c[:3, :3].T + w2c[:3, 3]
    tri = cam[F[:, :3]]  # (M, 3, 3) in camera coords
    z = tri[:, :, 2]
    keep = (z > 0.05).all(axis=1) & (z < far_m).any(axis=1)
    if not keep.any():
        return _pack(depth, inst, normals, names, "raster")
    tri, inst_idx = tri[keep], F[keep, 3]
    world_tri = V[F[keep][:, :3]]

    px = intr.fx * tri[:, :, 0] / tri[:, :, 2] + intr.cx
    py = intr.fy * tri[:, :, 1] / tri[:, :, 2] + intr.cy
    x0 = np.maximum(np.floor(px.min(axis=1)).astype(np.int64), 0)
    x1 = np.minimum(np.ceil(px.max(axis=1)).astype(np.int64), W - 1)
    y0 = np.maximum(np.floor(py.min(axis=1)).astype(np.int64), 0)
    y1 = np.minimum(np.ceil(py.max(axis=1)).astype(np.int64), H - 1)
    visible = (x1 >= x0) & (y1 >= y0)

    e = world_tri[:, 1] - world_tri[:, 0]
    f = world_tri[:, 2] - world_tri[:, 0]
    n = np.cross(e, f)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, np.where(ln == 0, 1.0, ln))

    order = np.argsort(tri[:, :, 2].min(axis=1))  # near-to-far helps early-out
    for t in order:
        if not visible[t]:
            continue
        xs = np.arange(x0[t], x1[t] + 1)
        ys = np.arange(y0[t], y1[t] + 1)
        if xs.size == 0 or ys.size == 0:
            continue
        ax, ay = px[t, 0], py[t, 0]
        bx, by = px[t, 1], py[t, 1]
        cx_, cy_ = px[t, 2], py[t, 2]
        area = (bx - ax) * (cy_ - ay) - (cx_ - ax) * (by - ay)
        if abs(area) < 1e-12:
            continue
        gx, gy = np.meshgrid(xs + 0.5, ys + 0.5)
        w0 = ((bx - ax) * (gy - ay) - (gx - ax) * (by - ay)) / area
        w1 = ((cx_ - bx) * (gy - by) - (gx - bx) * (cy_ - by)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not inside.any():
            continue
        # perspective-correct depth: interpolate 1/z in screen space
        inv_z = w1 / tri[t, 0, 2] + w2 / tri[t, 1, 2] + w0 / tri[t, 2, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            zz = np.where(inv_z > 0, 1.0 / inv_z, np.inf)
        sub = depth[ys[0] : ys[-1] + 1, xs[0] : xs[-1] + 1]
        better = inside & (zz < sub)
        if not better.any():
            continue
        sub[better] = zz[better]
        inst[ys[0] : ys[-1] + 1, xs[0] : xs[-1] + 1][better] = inst_idx[t]
        normals[ys[0] : ys[-1] + 1, xs[0] : xs[-1] + 1][better] = n[t]
    return _pack(depth, inst, normals, names, "raster")


def _pack(depth, inst, normals, names, backend) -> dict[str, Any]:
    return {
        "depth": depth.astype(np.float32),
        "instance_id": inst.astype(np.int32),
        "normals": normals.astype(np.float32),
        "instance_names": names,
        "backend": backend,
    }


# --- trimesh ray casting -----------------------------------------------------
def _pixel_rays(intr: CameraIntrinsics, c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    W, H = int(intr.width), int(intr.height)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d_cam = np.stack(
        [(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, np.ones_like(u)], axis=-1
    ).reshape(-1, 3)
    d_cam /= np.linalg.norm(d_cam, axis=1, keepdims=True)
    d_world = d_cam @ c2w[:3, :3].T
    origins = np.repeat(c2w[:3, 3][None, :], len(d_world), axis=0)
    return origins, d_world


def _render_raycast(c2w: np.ndarray, intr: CameraIntrinsics, scene, far_m: float) -> dict[str, Any]:
    import trimesh

    V, F, names = _scene_triangles(scene)
    W, H = int(intr.width), int(intr.height)
    mesh = trimesh.Trimesh(vertices=V, faces=F[:, :3], process=False)
    face_inst = F[:, 3]
    origins, dirs = _pixel_rays(intr, c2w)
    loc, idx_ray, idx_tri = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
    depth = np.full(H * W, np.inf)
    inst = np.full(H * W, -1, dtype=np.int32)
    normals = np.zeros((H * W, 3))
    if len(idx_ray):
        along = ((loc - origins[idx_ray]) @ c2w[:3, :3])[:, 2]
        depth[idx_ray] = along
        inst[idx_ray] = face_inst[idx_tri]
        normals[idx_ray] = mesh.face_normals[idx_tri]
    depth[depth > far_m] = np.inf
    return _pack(
        depth.reshape(H, W), inst.reshape(H, W), normals.reshape(H, W, 3), names, "raycast"
    )


# --- pyrender ----------------------------------------------------------------
def _render_pyrender(
    c2w: np.ndarray, intr: CameraIntrinsics, scene, far_m: float
) -> dict[str, Any]:
    import pyrender
    import trimesh

    V, F, names = _scene_triangles(scene)
    rscene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
    for i, name in enumerate(names):
        sel = F[F[:, 3] == i][:, :3]
        if not len(sel):
            continue
        rscene.add(
            pyrender.Mesh.from_trimesh(
                trimesh.Trimesh(vertices=V, faces=sel, process=False), smooth=False
            ),
            name=name,
        )
    cam = pyrender.IntrinsicsCamera(
        fx=intr.fx, fy=intr.fy, cx=intr.cx, cy=intr.cy, znear=0.5, zfar=far_m
    )
    rscene.add(cam, pose=c2w @ _OPENCV_TO_GL)
    r = pyrender.OffscreenRenderer(int(intr.width), int(intr.height))
    try:
        _, depth = r.render(rscene, flags=pyrender.RenderFlags.DEPTH_ONLY)
    finally:
        r.delete()
    depth = np.asarray(depth, dtype=np.float64)
    depth[depth <= 0] = np.inf
    # pyrender gives depth only; recover ids/normals from the raster pass so
    # the return shape is identical across backends.
    raster = _render_raster(c2w, intr, scene, far_m)
    raster["depth"] = depth.astype(np.float32)
    raster["backend"] = "pyrender"
    return raster
