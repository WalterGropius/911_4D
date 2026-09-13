"""COLMAP **text** model I/O (``cameras.txt``, ``images.txt``, ``points3D.txt``).

We do not run COLMAP for structure-from-motion (our cameras come from
``wtc4d.camreg``: landmark PnP against the geo prior), but the text model is
the lingua franca of every splat trainer, viewer and dataset converter.
Exporting it means nerfstudio / gsplat / Polycam / SuperSplat tooling can
read our scenes, and importing it means somebody *can* drop a COLMAP
reconstruction of, say, the rubble pile into the pipeline.

Conventions: COLMAP stores **world-to-camera** rotation as a quaternion
``(qw, qx, qy, qz)`` plus translation, with OpenCV camera axes.  See
:mod:`wtc4d.recon.conventions` for the exact conversion to our ``c2w``.

Distortion parameters are mapped onto ``CameraIntrinsics.dist`` in OpenCV
order ``[k1, k2, p1, p2, k3]``, truncated to the parameters the COLMAP model
actually carries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from wtc4d.schema.camera import CameraIntrinsics, CameraPose

from .conventions import c2w_to_colmap, colmap_to_c2w

__all__ = [
    "ColmapImage",
    "ColmapModel",
    "ColmapPoint3D",
    "camera_line_to_intrinsics",
    "intrinsics_to_camera_line",
    "read_colmap_model",
    "write_colmap_model",
]

# COLMAP camera model -> parameter names, in file order.
_CAMERA_MODELS: dict[str, tuple[str, ...]] = {
    "SIMPLE_PINHOLE": ("f", "cx", "cy"),
    "PINHOLE": ("fx", "fy", "cx", "cy"),
    "SIMPLE_RADIAL": ("f", "cx", "cy", "k1"),
    "RADIAL": ("f", "cx", "cy", "k1", "k2"),
    "OPENCV": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"),
    "FULL_OPENCV": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"),
}


@dataclass
class ColmapImage:
    """One row of ``images.txt`` (pose + name), without the 2D observations."""

    image_id: int
    qvec: np.ndarray  # (4,) w2c quaternion, (qw, qx, qy, qz)
    tvec: np.ndarray  # (3,) w2c translation
    camera_id: int
    name: str
    xys: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    point3d_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))

    def c2w(self) -> np.ndarray:
        """Camera-to-world 4x4 in our (OpenCV) convention."""
        return colmap_to_c2w(self.qvec, self.tvec)


@dataclass
class ColmapPoint3D:
    """One row of ``points3D.txt``."""

    point_id: int
    xyz: np.ndarray  # (3,)
    rgb: np.ndarray  # (3,) uint8
    error: float = 0.0
    image_ids: list[int] = field(default_factory=list)
    point2d_idxs: list[int] = field(default_factory=list)


@dataclass
class ColmapModel:
    """A COLMAP text model: intrinsics by camera id, images, sparse points."""

    cameras: dict[int, CameraIntrinsics] = field(default_factory=dict)
    camera_models: dict[int, str] = field(default_factory=dict)
    images: dict[int, ColmapImage] = field(default_factory=dict)
    points: dict[int, ColmapPoint3D] = field(default_factory=dict)

    def to_poses(self, shot_id_from_name=None, method: str = "colmap") -> list[CameraPose]:
        """Convert to project :class:`CameraPose` objects.

        ``shot_id_from_name`` maps an image file name to ``(shot_id, frame_idx)``.
        The default splits ``"<shot_id>/<frame_idx>.<ext>"`` and falls back to
        ``(stem, 0)``.
        """
        fn = shot_id_from_name or default_shot_id_from_name
        out: list[CameraPose] = []
        for image in sorted(self.images.values(), key=lambda im: im.image_id):
            shot_id, frame_idx = fn(image.name)
            out.append(
                CameraPose.from_matrix(
                    image.c2w(),
                    shot_id=shot_id,
                    frame_idx=frame_idx,
                    intrinsics=self.cameras[image.camera_id],
                    method=method,
                    notes=f"colmap image {image.image_id} ({image.name})",
                )
            )
        return out

    def points_array(self) -> tuple[np.ndarray, np.ndarray]:
        """``(xyz (M, 3) float64, rgb (M, 3) float32 in [0, 1])``."""
        if not self.points:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.float32)
        ids = sorted(self.points)
        xyz = np.stack([self.points[i].xyz for i in ids]).astype(np.float64)
        rgb = np.stack([self.points[i].rgb for i in ids]).astype(np.float32) / 255.0
        return xyz, rgb


def _parse_observations(tokens: list[str]) -> np.ndarray | None:
    """``(X, Y, POINT3D_ID)`` triples, or ``None`` when this is not such a line."""
    if not tokens:
        return np.zeros((0, 3))
    if len(tokens) % 3 != 0:
        return None
    try:
        return np.array([float(x) for x in tokens]).reshape(-1, 3)
    except ValueError:
        return None


def default_shot_id_from_name(name: str) -> tuple[str, int]:
    """``"shot/000012.jpg"`` -> ``("shot", 12)``; ``"foo.jpg"`` -> ``("foo", 0)``."""
    p = Path(name)
    stem = p.stem
    if p.parent != Path("."):
        try:
            return str(p.parent).replace("\\", "/"), int(stem)
        except ValueError:
            return str(p.parent).replace("\\", "/"), 0
    # "<shot_id>_<frame>" is the other common flat layout
    if "_" in stem:
        head, _, tail = stem.rpartition("_")
        if tail.isdigit() and head:
            return head, int(tail)
    return stem, 0


def camera_line_to_intrinsics(model: str, width: int, height: int, params) -> CameraIntrinsics:
    """COLMAP camera model + params -> :class:`CameraIntrinsics`."""
    if model not in _CAMERA_MODELS:
        raise ValueError(f"unsupported COLMAP camera model: {model}")
    names = _CAMERA_MODELS[model]
    params = [float(p) for p in params]
    if len(params) != len(names):
        raise ValueError(f"{model} expects {len(names)} params, got {len(params)}")
    d = dict(zip(names, params, strict=True))
    fx = d.get("fx", d.get("f", 0.0))
    fy = d.get("fy", d.get("f", 0.0))
    dist = [d.get("k1", 0.0), d.get("k2", 0.0), d.get("p1", 0.0), d.get("p2", 0.0)]
    if "k3" in d:
        dist.append(d["k3"])
    while dist and dist[-1] == 0.0:
        dist.pop()
    out_model = "OPENCV" if model in ("OPENCV", "FULL_OPENCV") else "PINHOLE"
    if model in ("SIMPLE_RADIAL", "RADIAL"):
        out_model = "OPENCV"
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=d["cx"],
        cy=d["cy"],
        model=out_model,
        dist=dist,
    )


def intrinsics_to_camera_line(intr: CameraIntrinsics) -> tuple[str, list[float]]:
    """:class:`CameraIntrinsics` -> ``(colmap_model_name, params)``.

    Emits ``PINHOLE`` when there is no distortion and ``OPENCV`` otherwise
    (``k3`` and beyond are dropped -- ``FULL_OPENCV`` would be needed and no
    2001 broadcast lens model we have justifies it).
    """
    dist = list(intr.dist)
    if not any(abs(d) > 0 for d in dist):
        return "PINHOLE", [intr.fx, intr.fy, intr.cx, intr.cy]
    dist = (dist + [0.0, 0.0, 0.0, 0.0])[:4]
    return "OPENCV", [intr.fx, intr.fy, intr.cx, intr.cy, *dist]


def read_colmap_model(path: str | Path) -> ColmapModel:
    """Read a COLMAP text model directory (``points3D.txt`` optional)."""
    path = Path(path)
    model = ColmapModel()

    with (path / "cameras.txt").open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model.camera_models[cam_id] = parts[1]
            model.cameras[cam_id] = camera_line_to_intrinsics(
                parts[1], int(parts[2]), int(parts[3]), parts[4:]
            )

    # Each image is two lines: the pose, then its 2D observations (often empty
    # for our exports).  Blank lines must therefore be kept, not filtered.
    with (path / "images.txt").open() as fh:
        lines = [ln.rstrip("\n") for ln in fh if not ln.lstrip().startswith("#")]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        parts = line.split()
        image_id = int(parts[0])
        qvec = np.array([float(x) for x in parts[1:5]])
        tvec = np.array([float(x) for x in parts[5:8]])
        camera_id = int(parts[8])
        name = " ".join(parts[9:])
        xys = np.zeros((0, 2))
        pids = np.zeros((0,), dtype=np.int64)
        consumed = 1
        if i + 1 < len(lines):
            obs = lines[i + 1].split()
            arr = _parse_observations(obs)
            if arr is not None:
                consumed = 2
                if len(arr):
                    xys = arr[:, :2]
                    pids = arr[:, 2].astype(np.int64)
        model.images[image_id] = ColmapImage(image_id, qvec, tvec, camera_id, name, xys, pids)
        i += consumed

    p3d = path / "points3D.txt"
    if p3d.exists():
        with p3d.open() as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                pid = int(parts[0])
                track = [int(x) for x in parts[8:]]
                model.points[pid] = ColmapPoint3D(
                    point_id=pid,
                    xyz=np.array([float(x) for x in parts[1:4]]),
                    rgb=np.array([int(x) for x in parts[4:7]], dtype=np.uint8),
                    error=float(parts[7]),
                    image_ids=track[0::2],
                    point2d_idxs=track[1::2],
                )
    return model


def write_colmap_model(
    path: str | Path,
    poses: list[CameraPose],
    points_xyz=None,
    points_rgb=None,
    image_name=None,
) -> Path:
    """Write ``cameras.txt`` / ``images.txt`` / ``points3D.txt`` from our poses.

    One COLMAP camera is emitted per distinct intrinsics object, so a scene
    mixing broadcast feeds (different focal lengths per shot) stays exact.
    ``image_name(pose) -> str`` defaults to ``"<shot_id>/<frame_idx:06d>.jpg"``,
    matching the frame store layout in ``data/README.md``.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    namer = image_name or (lambda p: f"{p.shot_id}/{p.frame_idx:06d}.jpg")

    cam_ids: dict[tuple, int] = {}
    cam_lines: list[str] = []
    img_lines: list[str] = []

    for idx, pose in enumerate(poses, start=1):
        intr = pose.intrinsics
        model_name, params = intrinsics_to_camera_line(intr)
        key = (model_name, intr.width, intr.height, tuple(round(p, 9) for p in params))
        if key not in cam_ids:
            cam_ids[key] = len(cam_ids) + 1
            cam_lines.append(
                f"{cam_ids[key]} {model_name} {intr.width} {intr.height} "
                + " ".join(f"{p:.10g}" for p in params)
            )
        q, t = c2w_to_colmap(pose.matrix())
        img_lines.append(
            f"{idx} {q[0]:.10g} {q[1]:.10g} {q[2]:.10g} {q[3]:.10g} "
            f"{t[0]:.10g} {t[1]:.10g} {t[2]:.10g} {cam_ids[key]} {namer(pose)}"
        )

    with (path / "cameras.txt").open("w") as fh:
        fh.write("# Camera list with one line of data per camera:\n")
        fh.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        fh.write(f"# Number of cameras: {len(cam_lines)}\n")
        fh.write("\n".join(cam_lines) + ("\n" if cam_lines else ""))

    with (path / "images.txt").open("w") as fh:
        fh.write("# Image list with two lines of data per image:\n")
        fh.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        fh.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        fh.write(f"# Number of images: {len(img_lines)}, mean observations per image: 0\n")
        for line in img_lines:
            fh.write(line + "\n\n")  # empty observation line, as COLMAP expects

    xyz = np.zeros((0, 3)) if points_xyz is None else np.asarray(points_xyz, dtype=np.float64)
    if points_rgb is None:
        rgb = np.full((len(xyz), 3), 128, dtype=np.uint8)
    else:
        rgb = np.asarray(points_rgb)
        if rgb.dtype != np.uint8:
            rgb = np.clip(np.asarray(rgb, dtype=np.float64) * 255.0, 0, 255).astype(np.uint8)
    with (path / "points3D.txt").open("w") as fh:
        fh.write("# 3D point list with one line of data per point:\n")
        fh.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        fh.write(f"# Number of points: {len(xyz)}, mean track length: 0\n")
        for i, (p, c) in enumerate(zip(xyz, rgb, strict=True), start=1):
            if not all(math.isfinite(v) for v in p):
                continue
            fh.write(
                f"{i} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {int(c[0])} {int(c[1])} {int(c[2])} 0\n"
            )
    return path
