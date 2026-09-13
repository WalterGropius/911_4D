"""Training dataset: registered poses + frames (+ masks, + absolute time).

Inputs come from the sibling workstreams:

* ``data/cameras/poses.jsonl`` -- one :class:`wtc4d.schema.camera.CameraPose`
  per line (``camreg``);
* a frames directory, laid out as ``<frames_dir>/<shot_id>/<frame_idx:06d>.jpg``
  (``corpus``; on the compute volume this is ``/data/frames``);
* an optional masks directory with the same layout -- **1 = dynamic** pixel
  (smoke, fire, dust, debris, broadcast graphics), 0 = usable static content
  (``camreg``).  Pass ``mask_is_static=True`` if a producer emits the
  opposite polarity;
* optional ``TimeEstimate``s (``sync``), attached per (shot, frame).

The dataset is deliberately plain: it returns one :class:`FrameSample` per
frame with tensors on the requested device.  3DGS trains on whole images, so
there is no batching machinery to get wrong.

This module also carries the interoperability layer (re-exported from
:mod:`~wtc4d.recon.colmap` and :mod:`~wtc4d.recon.nerfstudio`) so that a
scene can round-trip to COLMAP text or nerfstudio ``transforms.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from wtc4d import timeline
from wtc4d.schema.camera import CameraIntrinsics, CameraPose
from wtc4d.schema.time import TimeEstimate

from .colmap import ColmapModel, read_colmap_model, write_colmap_model
from .conventions import (
    c2w_to_colmap,
    c2w_to_opengl,
    colmap_to_c2w,
    invert_rigid,
    look_at_c2w,
    opengl_to_c2w,
    quat_to_rotmat,
    rotmat_to_quat,
)
from .nerfstudio import read_transforms_json, write_transforms_json

__all__ = [
    "ColmapModel",
    "FrameSample",
    "ReconDataset",
    "c2w_to_colmap",
    "c2w_to_opengl",
    "colmap_to_c2w",
    "invert_rigid",
    "load_poses_jsonl",
    "load_time_estimates",
    "look_at_c2w",
    "opengl_to_c2w",
    "quat_to_rotmat",
    "read_colmap_model",
    "read_transforms_json",
    "rotmat_to_quat",
    "save_poses_jsonl",
    "scale_intrinsics",
    "write_colmap_model",
    "write_transforms_json",
]

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp")


@dataclass
class FrameSample:
    """One training view."""

    index: int
    shot_id: str
    frame_idx: int
    c2w: Tensor  # (4, 4) camera-to-world, OpenCV axes, world ENU metres
    intrinsics: CameraIntrinsics  # already scaled to (width, height)
    width: int
    height: int
    image: Tensor | None = None  # (H, W, 3) float in [0, 1], linear-ish sRGB values
    dynamic_mask: Tensor | None = None  # (H, W, 1) float, 1 = dynamic pixel
    t: float | None = None  # project seconds
    t_sigma: float | None = None
    image_path: Path | None = None
    prior_depth: Tensor | None = None  # (H, W, 1) metres, optional depth prior (see train_static)

    @property
    def static_weight(self) -> Tensor | None:
        """(H, W, 1) supervision weight for the *static* layers (1 - dynamic)."""
        if self.dynamic_mask is None:
            return None
        return 1.0 - self.dynamic_mask

    @property
    def key(self) -> tuple[str, int]:
        return (self.shot_id, self.frame_idx)


def scale_intrinsics(intr: CameraIntrinsics, downscale: float) -> CameraIntrinsics:
    """Scale intrinsics for an image resized by ``1 / downscale``.

    Uses the pixel-centre-at-(i + 0.5) convention, so the principal point maps
    as ``cx' = (cx + 0.5) / s - 0.5`` rather than the naive ``cx / s``.
    """
    if downscale == 1:
        return intr.model_copy(deep=True)
    s = float(downscale)
    return CameraIntrinsics(
        width=max(1, int(round(intr.width / s))),
        height=max(1, int(round(intr.height / s))),
        fx=intr.fx / s,
        fy=intr.fy / s,
        cx=(intr.cx + 0.5) / s - 0.5,
        cy=(intr.cy + 0.5) / s - 0.5,
        model=intr.model,
        dist=list(intr.dist),
    )


def load_poses_jsonl(path: str | Path) -> list[CameraPose]:
    """Read ``poses.jsonl`` (one :class:`CameraPose` JSON object per line)."""
    out: list[CameraPose] = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(CameraPose.model_validate_json(line))
    return out


def save_poses_jsonl(path: str | Path, poses: list[CameraPose]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for pose in poses:
            fh.write(pose.model_dump_json() + "\n")
    return path


def load_time_estimates(path: str | Path) -> dict[tuple[str, int], TimeEstimate]:
    """Read per-frame time estimates keyed by ``(shot_id, frame_idx)``.

    Tolerates the shapes the ``sync`` workstream might emit: a flat object
    carrying ``shot_id``/``frame_idx`` alongside the estimate fields, or a
    nested ``{"shot_id": ..., "frame_idx": ..., "estimate": {...}}``.  Lines
    without a frame index apply to frame 0 of the shot; callers that need
    shot-level times should interpolate with the shot's frame rate.
    """
    out: dict[tuple[str, int], TimeEstimate] = {}
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            doc = json.loads(line)
            shot_id = doc.get("shot_id") or doc.get("shot") or ""
            frame_idx = int(doc.get("frame_idx", doc.get("frame", 0)) or 0)
            payload = doc.get("estimate", doc)
            out[(shot_id, frame_idx)] = TimeEstimate.model_validate(
                {k: v for k, v in payload.items() if k in TimeEstimate.model_fields}
            )
    return out


class ReconDataset:
    """Registered views for one epoch or dynamic window.

    ``downscale`` divides the stored frame resolution (2001 broadcast SD is
    already small; training at half resolution first is still the fastest way
    to a stable geometry).  Images are cached in memory by default -- a few
    hundred SD frames are well under a gigabyte.
    """

    def __init__(
        self,
        poses: list[CameraPose],
        frames_dir: str | Path | None = None,
        masks_dir: str | Path | None = None,
        times: dict[tuple[str, int], TimeEstimate] | None = None,
        *,
        downscale: float = 1.0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        cache: bool = True,
        mask_is_static: bool = False,
        image_suffix: str | None = None,
    ) -> None:
        self.poses = list(poses)
        self.frames_dir = Path(frames_dir) if frames_dir else None
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.times = dict(times or {})
        self.downscale = float(downscale)
        self.device = torch.device(device)
        self.dtype = dtype
        self.cache = cache
        self.mask_is_static = mask_is_static
        self.image_suffix = image_suffix
        self._cache: dict[int, FrameSample] = {}

    # ------------------------------------------------------------- factories
    @classmethod
    def from_paths(
        cls,
        poses_path: str | Path,
        frames_dir: str | Path | None = None,
        masks_dir: str | Path | None = None,
        times_path: str | Path | None = None,
        **kwargs,
    ) -> ReconDataset:
        times = load_time_estimates(times_path) if times_path else None
        return cls(load_poses_jsonl(poses_path), frames_dir, masks_dir, times, **kwargs)

    @classmethod
    def from_colmap(
        cls, model_dir: str | Path, frames_dir: str | Path | None = None, **kwargs
    ) -> ReconDataset:
        model = read_colmap_model(model_dir)
        return cls(model.to_poses(), frames_dir, **kwargs)

    @classmethod
    def from_transforms_json(
        cls, path: str | Path, frames_dir: str | Path | None = None, **kwargs
    ) -> ReconDataset:
        poses, times = read_transforms_json(path)
        base = frames_dir if frames_dir is not None else Path(path).parent
        return cls(poses, base, times=times, **kwargs)

    # ------------------------------------------------------------- accessors
    def __len__(self) -> int:
        return len(self.poses)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index: int) -> FrameSample:
        if index in self._cache:
            return self._cache[index]
        pose = self.poses[index]
        intr = scale_intrinsics(pose.intrinsics, self.downscale)
        te = self.times.get((pose.shot_id, pose.frame_idx))
        image_path = self.frame_path(pose)
        image = self._load_image(image_path, intr.width, intr.height) if image_path else None
        mask = self._load_mask(pose, intr.width, intr.height)
        sample = FrameSample(
            index=index,
            shot_id=pose.shot_id,
            frame_idx=pose.frame_idx,
            c2w=torch.as_tensor(pose.matrix(), device=self.device, dtype=self.dtype),
            intrinsics=intr,
            width=intr.width,
            height=intr.height,
            image=image,
            dynamic_mask=mask,
            t=te.t if te else None,
            t_sigma=te.sigma if te else None,
            image_path=image_path,
        )
        if self.cache:
            self._cache[index] = sample
        return sample

    def frame_path(self, pose: CameraPose) -> Path | None:
        """Locate the image for a pose, trying the usual layouts and suffixes."""
        if self.frames_dir is None:
            return None
        return _find_frame(self.frames_dir, pose, self.image_suffix)

    # --------------------------------------------------------------- subsets
    def subset(self, indices) -> ReconDataset:
        ds = ReconDataset(
            [self.poses[i] for i in indices],
            self.frames_dir,
            self.masks_dir,
            self.times,
            downscale=self.downscale,
            device=self.device,
            dtype=self.dtype,
            cache=self.cache,
            mask_is_static=self.mask_is_static,
            image_suffix=self.image_suffix,
        )
        for new_i, old_i in enumerate(indices):
            if old_i in self._cache:
                ds._cache[new_i] = self._cache[old_i]
        return ds

    def filter_time(self, t_start: float, t_end: float, keep_untimed: bool = False) -> ReconDataset:
        """Keep frames whose absolute time falls in ``[t_start, t_end)``."""
        keep = []
        for i, pose in enumerate(self.poses):
            te = self.times.get((pose.shot_id, pose.frame_idx))
            if te is None:
                if keep_untimed:
                    keep.append(i)
                continue
            if t_start <= te.t < t_end:
                keep.append(i)
        return self.subset(keep)

    def filter_epoch(self, epoch_id: str, keep_untimed: bool = False) -> ReconDataset:
        ep = timeline.EPOCHS_BY_ID[epoch_id]
        return self.filter_time(ep.t_start, ep.t_end, keep_untimed)

    def split(self, holdout_every: int = 8) -> tuple[ReconDataset, ReconDataset]:
        """Train/validation split holding out every ``n``-th frame.

        Held-out frames are strided rather than random so that evaluation
        covers every camera: with ~10-30 broadcasts per window, a random split
        can drop an entire viewpoint.
        """
        if holdout_every <= 1:
            return self, self.subset([])
        train = [i for i in range(len(self)) if i % holdout_every != holdout_every - 1]
        val = [i for i in range(len(self)) if i % holdout_every == holdout_every - 1]
        return self.subset(train), self.subset(val)

    def shots(self) -> list[str]:
        return sorted({p.shot_id for p in self.poses})

    def time_range(self) -> tuple[float, float] | None:
        ts = [te.t for te in (self.times.get((p.shot_id, p.frame_idx)) for p in self.poses) if te]
        return (min(ts), max(ts)) if ts else None

    # ---------------------------------------------------------------- export
    def export_colmap(self, path: str | Path, points_xyz=None, points_rgb=None) -> Path:
        return write_colmap_model(path, self.poses, points_xyz, points_rgb)

    def export_transforms_json(self, path: str | Path, **kwargs) -> Path:
        return write_transforms_json(path, self.poses, times=self.times, **kwargs)

    # ----------------------------------------------------------------- I/O
    def _load_image(self, path: Path, width: int, height: int) -> Tensor | None:
        arr = _read_image(path, width, height)
        if arr is None:
            return None
        return torch.as_tensor(arr, device=self.device, dtype=self.dtype)

    def _load_mask(self, pose: CameraPose, width: int, height: int) -> Tensor | None:
        if self.masks_dir is None:
            return None
        path = _find_frame(self.masks_dir, pose, None)
        if path is None:
            return None
        arr = _read_image(path, width, height)
        if arr is None:
            return None
        m = arr.mean(axis=-1, keepdims=True)
        if self.mask_is_static:
            m = 1.0 - m
        return torch.as_tensor(m, device=self.device, dtype=self.dtype)


def _find_frame(root: Path, pose: CameraPose, suffix: str | None) -> Path | None:
    stems = [f"{pose.frame_idx:06d}", str(pose.frame_idx), f"{pose.shot_id}_{pose.frame_idx:06d}"]
    dirs = [root / pose.shot_id, root]
    suffixes = [suffix] if suffix else IMAGE_SUFFIXES
    for d in dirs:
        for stem in stems:
            for ext in suffixes:
                p = d / f"{stem}{ext}"
                if p.exists():
                    return p
    return None


def _read_image(path: Path, width: int, height: int) -> np.ndarray | None:
    """Read an image as float32 ``(H, W, 3)`` in [0, 1], resized to (width, height)."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - pillow is in the recon extra
        raise RuntimeError("pillow is required to read frames (pip install '.[recon]')") from exc
    with Image.open(path) as img:
        img = img.convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0
