"""Shared data types for the sync workstream.

Units and conventions
---------------------
* Absolute times are **project seconds** (seconds since 2001-09-11 00:00:00
  EDT) as defined by :mod:`wtc4d.timeline`.
* Relative times inside a media file are **seconds from the first frame of
  the file**, never from the first frame of a shot, unless a name says
  ``shot_``.
* Image coordinates are pixels, origin top-left, +x right, +y down (OpenCV).
* Every uncertainty is a **1-sigma** value in seconds unless stated.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from wtc4d.schema.time import TimeEstimate, TimeMethod

# --- geometry helpers --------------------------------------------------------


class BBox(BaseModel):
    """Axis-aligned image rectangle, pixels, origin top-left."""

    x: int
    y: int
    w: int = Field(gt=0)
    h: int = Field(gt=0)

    @property
    def x1(self) -> int:
        return self.x + self.w

    @property
    def y1(self) -> int:
        return self.y + self.h

    def clip(self, width: int, height: int) -> BBox:
        """Clamp to an image of the given size (never returns a zero-area box)."""
        x0 = max(0, min(self.x, width - 1))
        y0 = max(0, min(self.y, height - 1))
        x1 = max(x0 + 1, min(self.x1, width))
        y1 = max(y0 + 1, min(self.y1, height))
        return BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0)

    def scaled(self, sx: float, sy: float) -> BBox:
        return BBox(
            x=int(round(self.x * sx)),
            y=int(round(self.y * sy)),
            w=max(1, int(round(self.w * sx))),
            h=max(1, int(round(self.h * sy))),
        )

    def as_slices(self) -> tuple[slice, slice]:
        """``img[bbox.as_slices()]`` -> the cropped region (rows, cols)."""
        return slice(self.y, self.y1), slice(self.x, self.x1)


class RoiTrack(BaseModel):
    """A frame-indexed region of interest.

    This is the interface the ``camreg`` workstream is expected to fill in
    later: given a registered camera and the world position of a tower, it
    can emit the bounding box of that tower for every frame.  Until then a
    caller supplies ``boxes`` by hand (one entry is enough for a static
    camera).  Boxes are linearly interpolated between the supplied keys and
    held constant outside the key range.
    """

    boxes: dict[int, BBox] = Field(default_factory=dict)
    label: str = ""

    def at(self, frame_idx: int) -> BBox | None:
        if not self.boxes:
            return None
        keys = sorted(self.boxes)
        if frame_idx <= keys[0]:
            return self.boxes[keys[0]]
        if frame_idx >= keys[-1]:
            return self.boxes[keys[-1]]
        # bracket
        lo = max(k for k in keys if k <= frame_idx)
        hi = min(k for k in keys if k >= frame_idx)
        if lo == hi:
            return self.boxes[lo]
        a, b = self.boxes[lo], self.boxes[hi]
        f = (frame_idx - lo) / (hi - lo)

        def _mix(u: int, v: int) -> int:
            return int(round(u + f * (v - u)))

        return BBox(x=_mix(a.x, b.x), y=_mix(a.y, b.y), w=_mix(a.w, b.w), h=_mix(a.h, b.h))

    @classmethod
    def static(cls, box: BBox, label: str = "") -> RoiTrack:
        return cls(boxes={0: box}, label=label)


# --- timing ------------------------------------------------------------------


class LinearClock(BaseModel):
    """Maps a frame index to project seconds: ``t = t0 + (i - i0) / rate``.

    ``rate`` is the *effective* frame rate in frames per second of absolute
    time.  It differs from the nominal fps when a recording was sped up or
    slowed down (PAL/NTSC transfers, VHS, telecine), which is why it is
    fitted rather than assumed.
    """

    t0: float = Field(description="project seconds at frame i0")
    i0: int = 0
    rate: float = Field(gt=0.0, description="frames per second of absolute time")
    sigma_t0: float = Field(default=0.0, ge=0.0)
    sigma_rate: float = Field(default=0.0, ge=0.0)

    def time_of(self, frame_idx: int | float) -> float:
        return self.t0 + (float(frame_idx) - self.i0) / self.rate

    def sigma_of(self, frame_idx: int | float) -> float:
        """1-sigma of ``time_of``, propagating both fitted parameters."""
        dn = float(frame_idx) - self.i0
        d_rate = abs(dn) * self.sigma_rate / (self.rate**2)
        return float((self.sigma_t0**2 + d_rate**2) ** 0.5)

    def frame_of(self, t: float) -> float:
        return self.i0 + (t - self.t0) * self.rate


class PairwiseOffset(BaseModel):
    """A relative timing constraint: ``t(b) - t(a) = dt`` (+- sigma).

    ``a`` and ``b`` are shot ids.  ``dt`` relates the *start frames* of the
    two shots in absolute seconds.  Produced by audio cross-correlation or
    visual matching; consumed by :func:`wtc4d.sync.fuse.solve_graph`.
    """

    a: str
    b: str
    dt: float
    sigma: float = Field(gt=0.0)
    method: TimeMethod = TimeMethod.AUDIO_XCORR
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: str = ""
    rate_ratio: float = Field(
        default=1.0, gt=0.0, description="playback rate of b relative to a (1.0 = same speed)"
    )


class ShotTimeRecord(BaseModel):
    """One line of ``data/time/time_estimates.jsonl``.

    ``time`` is the fused absolute time of ``start_frame`` of the shot.
    ``candidates`` keeps *every* input estimate (including the ones that were
    rejected, listed again in ``rejected``) so a reviewer can retrace how the
    number was reached.
    """

    shot_id: str
    source_id: str = ""
    fps: float = Field(gt=0.0, description="nominal frame rate of the media")
    start_frame: int = 0
    n_frames: int | None = None
    time: TimeEstimate | None = None
    clock: LinearClock | None = Field(
        default=None,
        description="fitted frame->time map; when absent, nominal fps from start_frame is used",
    )
    candidates: list[TimeEstimate] = Field(default_factory=list)
    rejected: list[TimeEstimate] = Field(default_factory=list)
    chi2: float | None = None
    dof: int | None = None
    notes: str = ""

    def frame_time(self, frame_idx: int) -> TimeEstimate | None:
        """Absolute time of an arbitrary frame of this shot.

        Uses the fitted :class:`LinearClock` when one is available, otherwise
        extrapolates from the shot start at the nominal fps (which adds no
        uncertainty of its own for short shots, but see
        :meth:`LinearClock.sigma_of` for the fitted case).
        """
        if self.time is None:
            return None
        if self.clock is not None:
            return TimeEstimate(
                t=self.clock.time_of(frame_idx),
                sigma=max(self.clock.sigma_of(frame_idx), 1e-6),
                method=self.time.method,
                evidence=f"{self.evidence_head()} + fitted clock rate {self.clock.rate:.6f} fps",
                derived_from=[self.shot_id],
            )
        dt = (frame_idx - self.start_frame) / self.fps
        return TimeEstimate(
            t=self.time.t + dt,
            sigma=self.time.sigma,
            method=self.time.method,
            evidence=f"{self.evidence_head()} + {dt:+.3f} s at nominal {self.fps:g} fps",
            derived_from=[self.shot_id],
        )

    def evidence_head(self) -> str:
        if self.time is None:
            return ""
        return self.time.evidence.splitlines()[0] if self.time.evidence else self.time.method.value


class EventKind(StrEnum):
    """Image-detectable anchors.  These map onto :mod:`wtc4d.timeline` events."""

    IMPACT_FLASH = "impact_flash"
    COLLAPSE_ONSET = "collapse_onset"


class EventCandidate(BaseModel):
    """A detected anchor inside a media file (times relative to the file)."""

    kind: EventKind
    frame_idx: int
    t_rel: float = Field(description="seconds from the first analysed frame of the file")
    confidence: float = Field(ge=0.0, le=1.0)
    sigma_rel: float = Field(
        default=0.0, ge=0.0, description="1-sigma of the detected onset within the file, seconds"
    )
    features: dict[str, float] = Field(default_factory=dict)
    roi_label: str = ""

    def anchor_ids(self) -> tuple[str, ...]:
        """Timeline event ids this kind of candidate can be matched against."""
        if self.kind is EventKind.IMPACT_FLASH:
            return ("wtc1_impact", "wtc2_impact")
        return ("wtc2_collapse", "wtc1_collapse")


__all__ = [
    "BBox",
    "EventCandidate",
    "EventKind",
    "LinearClock",
    "PairwiseOffset",
    "RoiTrack",
    "ShotTimeRecord",
    "TimeEstimate",
    "TimeMethod",
]
