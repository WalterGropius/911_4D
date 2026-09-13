from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from wtc4d.schema.geometry import LatLonAlt
from wtc4d.schema.time import TimeEstimate


class SourceKind(StrEnum):
    VIDEO = "video"  # raw / amateur / documentary video
    TV_BROADCAST = "tv_broadcast"  # off-air recording with graphics, likely timed
    PHOTO = "photo"
    AERIAL = "aerial"  # aerial / satellite imagery
    LIDAR = "lidar"
    DOCUMENT = "document"  # drawings, reports (NIST), maps
    OTHER = "other"


class Source(BaseModel):
    """One retrievable item (a video file, an image, a dataset).  Sources are
    catalogued, not redistributed: the repo stores metadata and pointers, the
    bytes live on the compute volume (see infra/)."""

    id: str = Field(description="stable slug, e.g. 'ia-<identifier>' or 'yt-<video_id>'")
    kind: SourceKind
    url: str
    archive: str = Field(
        description="archive.org | youtube | wikimedia | nist_foia | flickr | noaa | usgs | other"
    )
    title: str = ""
    creator: str = ""  # photographer / broadcaster / uploader
    license: str = "unknown"  # SPDX id, 'public-domain', 'fair-use-research', 'unknown'
    retrieved_at: datetime | None = None
    checksum_sha256: str | None = None
    bytes: int | None = None
    duration_s: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    # coarse hints filled at catalogue time
    time_hint: TimeEstimate | None = None  # e.g. broadcast airtime for the start of the item
    location_hint: LatLonAlt | None = None
    location_hint_text: str = ""  # 'Brooklyn Promenade', 'WABC NewsCopter 7', ...
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    quality_score: float | None = Field(default=None, ge=0, le=1)


class Shot(BaseModel):
    """A contiguous single-camera segment of a video source."""

    id: str
    source_id: str
    start_frame: int
    end_frame: int  # exclusive
    fps: float
    camera_motion: str = "unknown"  # static | pan_tilt | handheld | aerial | zoom | unknown
    towers_visible: bool | None = None
    quality_score: float | None = Field(default=None, ge=0, le=1)
    time: TimeEstimate | None = None  # absolute time of start_frame
    camera_prior_id: str | None = None  # -> CameraPrior.id when the vantage point is known
    notes: str = ""

    @property
    def start_s(self) -> float:
        return self.start_frame / self.fps

    @property
    def end_s(self) -> float:
        return self.end_frame / self.fps


class Frame(BaseModel):
    """Reference to one frame of a shot (or a photo source with frame_idx=0)."""

    shot_id: str
    frame_idx: int
    time: TimeEstimate | None = None
    path: str | None = None  # path on the compute volume, if extracted
