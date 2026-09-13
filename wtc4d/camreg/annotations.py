"""Annotation file format: clicked 2D landmark observations for one frame.

One JSON file per annotated frame::

    data/cameras/annotations/<shot_id>_<frame_idx:06d>.json

The file is the *only* thing committed -- never the frame image.  A frame is
identified by ``(shot_id, frame_idx)`` plus, when available, the SHA-256 of the
image bytes, so an annotation can be re-validated against a re-extracted frame.

Schema (``schema_version: 1``)::

    {
      "schema_version": 1,
      "shot_id": "naudet-01",
      "frame_idx": 0,
      "image_width": 720, "image_height": 480,
      "image_sha256": "…",                # optional
      "camera_prior_id": "jc_exchange_place",   # optional
      "annotator": "", "created_at": "…", "notes": "",
      "observations": [
        {"landmark_id": "wtc1_roof_ne", "u": 312.5, "v": 88.0,
         "sigma_px": 2.5, "occluded": false, "notes": ""}
      ]
    }

``u`` and ``v`` are pixel coordinates in the *stored frame*, origin at the
centre of the top-left pixel, +u right, +v down.  ``sigma_px`` is the
annotator's 1-sigma click uncertainty: use 1-2 px for a crisp corner against
sky, 5-15 px for a feature guessed through smoke or motion blur.  Bad sigmas
are the main reason a pose's reported uncertainty is wrong, so annotate them
honestly -- :func:`wtc4d.camreg.pnp.solve_pose` weights every residual by
``sigma_px`` and rescales the covariance by the achieved chi-square.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from wtc4d.camreg.landmarks import landmark_registry
from wtc4d.schema.geometry import LatLonAlt

SCHEMA_VERSION = 1

__all__ = [
    "SCHEMA_VERSION",
    "FrameAnnotation",
    "LandmarkObservation",
    "annotation_path",
    "image_sha256",
    "load_annotation",
    "save_annotation",
]


class LandmarkObservation(BaseModel):
    """One clicked landmark in one frame."""

    landmark_id: str
    u: float
    v: float
    sigma_px: float = Field(default=2.0, gt=0.0)
    occluded: bool = False
    """True when the annotator inferred the point through smoke/an obstruction
    rather than seeing it; such points should carry a large ``sigma_px``."""
    notes: str = ""


class FrameAnnotation(BaseModel):
    """All landmark observations for a single frame."""

    shot_id: str
    frame_idx: int = 0
    image_width: int
    image_height: int
    observations: list[LandmarkObservation] = Field(default_factory=list)
    image_sha256: str | None = None
    image_path: str | None = None
    """Path the annotation was made from, for the annotator's convenience only.
    It points at the frame store (never into the repo) and may be absent."""
    camera_prior_id: str | None = None
    custom_points: dict[str, LatLonAlt] = Field(default_factory=dict)
    """Ad-hoc 3D points used by this annotation only, keyed by the same
    ``landmark_id`` the observations refer to.  Use sparingly: anything
    generally useful belongs in the shared registry instead."""
    annotator: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    notes: str = ""
    schema_version: int = SCHEMA_VERSION

    # --- derived ---------------------------------------------------------
    @property
    def landmark_ids(self) -> list[str]:
        return [o.landmark_id for o in self.observations]

    def uv(self) -> np.ndarray:
        """(N, 2) observed pixel coordinates."""
        return np.array([[o.u, o.v] for o in self.observations], dtype=np.float64).reshape(-1, 2)

    def sigmas(self) -> np.ndarray:
        """(N,) click uncertainties in pixels."""
        return np.array([o.sigma_px for o in self.observations], dtype=np.float64)

    def world_points(self, registry: dict | None = None) -> np.ndarray:
        """(N, 3) world ENU coordinates of the observed landmarks, in order."""
        from wtc4d.world import Landmark, latlon_to_enu

        reg = dict(registry if registry is not None else landmark_registry())
        for lid, lla in self.custom_points.items():
            reg[lid] = Landmark(
                id=lid, name=lid, point=lla, height_m=lla.alt_m, kind="other", notes="custom"
            )
        missing = [lid for lid in self.landmark_ids if lid not in reg]
        if missing:
            raise KeyError(
                f"annotation {self.shot_id}#{self.frame_idx} references unknown landmarks: {missing}"
            )
        return np.array(
            [latlon_to_enu(reg[lid].point) for lid in self.landmark_ids], dtype=np.float64
        ).reshape(-1, 3)

    def check(self) -> list[str]:
        """Return a list of human-readable problems (empty when the file is sane)."""
        problems: list[str] = []
        if self.schema_version != SCHEMA_VERSION:
            problems.append(f"schema_version {self.schema_version} != {SCHEMA_VERSION}")
        seen: set[str] = set()
        for o in self.observations:
            if o.landmark_id in seen:
                problems.append(f"duplicate observation of {o.landmark_id}")
            seen.add(o.landmark_id)
            if not (-0.5 <= o.u <= self.image_width - 0.5):
                problems.append(f"{o.landmark_id}: u={o.u:.1f} outside image width")
            if not (-0.5 <= o.v <= self.image_height - 0.5):
                problems.append(f"{o.landmark_id}: v={o.v:.1f} outside image height")
        try:
            self.world_points()
        except KeyError as exc:  # unknown landmark ids
            problems.append(str(exc))
        if len(self.observations) < 3:
            problems.append(f"only {len(self.observations)} observations; PnP needs at least 3")
        return problems


def annotation_path(shot_id: str, frame_idx: int, root: str | Path | None = None) -> Path:
    """Canonical path of an annotation file."""
    base = (
        Path(root)
        if root is not None
        else Path(__file__).resolve().parents[2] / "data" / "cameras" / "annotations"
    )
    safe = "".join(c if c.isalnum() or c in "-_." else "-" for c in shot_id)
    return base / f"{safe}_{frame_idx:06d}.json"


def save_annotation(ann: FrameAnnotation, path: str | Path | None = None) -> Path:
    """Write an annotation to ``path`` (default: :func:`annotation_path`)."""
    p = Path(path) if path is not None else annotation_path(ann.shot_id, ann.frame_idx)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(ann.model_dump_json(indent=2, exclude_none=False) + "\n", encoding="utf-8")
    return p


def load_annotation(path: str | Path) -> FrameAnnotation:
    """Read an annotation file."""
    with open(path, encoding="utf-8") as fh:
        return FrameAnnotation.model_validate(json.load(fh))


def load_annotations(root: str | Path | None = None) -> list[FrameAnnotation]:
    """Read every ``*.json`` under the annotation directory (sorted by filename)."""
    base = (
        Path(root)
        if root is not None
        else Path(__file__).resolve().parents[2] / "data" / "cameras" / "annotations"
    )
    if not base.exists():
        return []
    return [load_annotation(p) for p in sorted(base.glob("*.json"))]


def image_sha256(path: str | Path) -> str:
    """SHA-256 of an image file, so an annotation can be tied to exact pixels."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
