"""Manifest consumed by the web viewer (``web/``).  Produced by ``recon`` and
``procedural``; camera entries come from ``camreg``.  Keep this JSON-friendly
and stable: the viewer is developed independently against sample manifests."""

from __future__ import annotations

from pydantic import BaseModel, Field

from wtc4d.schema.camera import CameraIntrinsics
from wtc4d.schema.geometry import LatLonAlt


class SplatAsset(BaseModel):
    id: str
    url: str  # relative to the manifest or absolute
    format: str = "ply"  # ply | splat | ksplat | spz | sog
    t_start: float
    t_end: float
    kind: str = "static"  # static | dynamic (time-varying, format-specific) | procedural
    layer: str = "scene"  # scene | smoke | debris | towers | ground
    epoch_id: str | None = None
    notes: str = ""


class CameraRef(BaseModel):
    """A registered footage viewpoint the viewer can jump to."""

    id: str
    shot_id: str
    frame_idx: int
    t: float
    c2w: list[float] = Field(min_length=16, max_length=16)
    intrinsics: CameraIntrinsics
    thumbnail_url: str | None = None
    source_url: str | None = None
    label: str = ""


class TimelineEvent(BaseModel):
    id: str
    name: str
    t: float
    sigma: float = 0.0


class SceneManifest(BaseModel):
    schema_version: int = 1
    name: str = "911_4D"
    world_origin: LatLonAlt
    t_min: float
    t_max: float
    events: list[TimelineEvent] = Field(default_factory=list)
    assets: list[SplatAsset] = Field(default_factory=list)
    cameras: list[CameraRef] = Field(default_factory=list)
    notes: str = ""
