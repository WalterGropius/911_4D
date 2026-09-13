"""nerfstudio ``transforms.json`` I/O.

nerfstudio (and most NeRF/3DGS forks descended from ``instant-ngp``) stores
``transform_matrix`` as a **camera-to-world** matrix in **OpenGL** axes
(+X right, +Y up, +Z backward).  Ours is camera-to-world in OpenCV axes, so
the two differ by a right-multiplied ``diag(1, -1, -1, 1)``; see
:mod:`wtc4d.recon.conventions`.

Because our shots are different broadcasts with different lenses, intrinsics
are written **per frame** (nerfstudio honours per-frame ``fl_x`` etc.) as
well as at the top level when every frame agrees.

Project-specific fields (``shot_id``, ``frame_idx``, ``t``, ``t_sigma``,
``epoch_id``) are written into each frame under the ``wtc4d`` key.  Foreign
readers ignore unknown keys; our reader uses them to keep provenance and
absolute time attached to the pose.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wtc4d.schema.camera import CameraIntrinsics, CameraPose
from wtc4d.schema.time import TimeEstimate, TimeMethod

from .conventions import c2w_to_opengl, opengl_to_c2w

__all__ = ["read_transforms_json", "write_transforms_json"]

_DIST_KEYS = ("k1", "k2", "p1", "p2", "k3")


def _intrinsics_dict(intr: CameraIntrinsics) -> dict:
    d: dict[str, float | int | str] = {
        "w": intr.width,
        "h": intr.height,
        "fl_x": intr.fx,
        "fl_y": intr.fy,
        "cx": intr.cx,
        "cy": intr.cy,
    }
    for key, value in zip(_DIST_KEYS, intr.dist, strict=False):
        d[key] = float(value)
    return d


def _intrinsics_from(d: dict, fallback: dict | None = None) -> CameraIntrinsics:
    src = dict(fallback or {})
    src.update(
        {k: v for k, v in d.items() if k in ("w", "h", "fl_x", "fl_y", "cx", "cy", *_DIST_KEYS)}
    )
    dist = [float(src[k]) for k in _DIST_KEYS if k in src]
    while dist and dist[-1] == 0.0:
        dist.pop()
    fl_x = float(src["fl_x"])
    return CameraIntrinsics(
        width=int(src["w"]),
        height=int(src["h"]),
        fx=fl_x,
        fy=float(src.get("fl_y", fl_x)),
        cx=float(src["cx"]),
        cy=float(src["cy"]),
        model="OPENCV" if dist else "PINHOLE",
        dist=dist,
    )


def write_transforms_json(
    path: str | Path,
    poses: list[CameraPose],
    file_path=None,
    mask_path=None,
    times: dict[tuple[str, int], TimeEstimate] | None = None,
    epoch_id: str | None = None,
    ply_file_path: str | None = None,
) -> Path:
    """Write ``transforms.json`` for ``poses``.

    ``file_path(pose) -> str`` and ``mask_path(pose) -> str | None`` produce
    the per-frame paths, relative to the json's directory.  Defaults follow
    the frame store layout (``<shot_id>/<frame_idx:06d>.jpg``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    namer = file_path or (lambda p: f"images/{p.shot_id}/{p.frame_idx:06d}.jpg")
    times = times or {}

    frames = []
    for pose in poses:
        frame = {
            "file_path": namer(pose),
            "transform_matrix": c2w_to_opengl(pose.matrix()).tolist(),
            **_intrinsics_dict(pose.intrinsics),
        }
        if mask_path is not None:
            mp = mask_path(pose)
            if mp:
                frame["mask_path"] = mp
        extra: dict[str, object] = {
            "shot_id": pose.shot_id,
            "frame_idx": pose.frame_idx,
            "method": pose.method,
        }
        te = times.get((pose.shot_id, pose.frame_idx))
        if te is not None:
            extra["t"] = te.t
            extra["t_sigma"] = te.sigma
            extra["t_method"] = str(te.method)
        if epoch_id:
            extra["epoch_id"] = epoch_id
        frame["wtc4d"] = extra
        frames.append(frame)

    doc: dict[str, object] = {"camera_model": "OPENCV", "frames": frames}
    if poses:
        first = _intrinsics_dict(poses[0].intrinsics)
        if all(_intrinsics_dict(p.intrinsics) == first for p in poses):
            doc.update(first)
    if ply_file_path:
        doc["ply_file_path"] = ply_file_path
    # Our world frame is already metric ENU; nerfstudio's auto-orientation must
    # be disabled downstream (`--orientation-method none --center-method none`)
    # or poses stop being comparable with `data/cameras/poses.jsonl`.
    doc["applied_transform"] = np.eye(4)[:3].tolist()

    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path


def read_transforms_json(
    path: str | Path,
) -> tuple[list[CameraPose], dict[tuple[str, int], TimeEstimate]]:
    """Read ``transforms.json`` -> ``(poses, times)`` in project conventions.

    ``applied_transform`` is **not** undone: if the file was produced by
    nerfstudio's own auto-orientation the poses are in nerfstudio's
    normalised frame, not our ENU frame, and there is no way to tell from
    the file alone.  A non-identity ``applied_transform`` therefore raises.
    """
    path = Path(path)
    doc = json.loads(path.read_text())
    applied = doc.get("applied_transform")
    if applied is not None:
        a = np.asarray(applied, dtype=np.float64)
        if a.shape == (3, 4):
            a = np.vstack([a, [0, 0, 0, 1]])
        if not np.allclose(a, np.eye(4), atol=1e-9):
            raise ValueError(
                "transforms.json has a non-identity applied_transform; the poses are not "
                "in the wtc4d world frame. Re-export with orientation/centering disabled."
            )
    fallback = {
        k: v for k, v in doc.items() if k in ("w", "h", "fl_x", "fl_y", "cx", "cy", *_DIST_KEYS)
    }

    poses: list[CameraPose] = []
    times: dict[tuple[str, int], TimeEstimate] = {}
    for i, frame in enumerate(doc.get("frames", [])):
        extra = frame.get("wtc4d", {})
        shot_id = extra.get("shot_id") or Path(frame["file_path"]).parent.name or "unknown"
        frame_idx = int(extra.get("frame_idx", _frame_idx_from(frame["file_path"], i)))
        pose = CameraPose.from_matrix(
            opengl_to_c2w(frame["transform_matrix"]),
            shot_id=shot_id,
            frame_idx=frame_idx,
            intrinsics=_intrinsics_from(frame, fallback),
            method=extra.get("method", "nerfstudio"),
        )
        poses.append(pose)
        if "t" in extra:
            times[(shot_id, frame_idx)] = TimeEstimate(
                t=float(extra["t"]),
                sigma=float(extra.get("t_sigma", 0.0)),
                method=TimeMethod(extra.get("t_method", TimeMethod.UNKNOWN)),
                evidence="from transforms.json",
            )
    return poses, times


def _frame_idx_from(file_path: str, default: int) -> int:
    stem = Path(file_path).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else default
