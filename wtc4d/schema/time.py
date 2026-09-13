from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class TimeMethod(StrEnum):
    """How an absolute time was obtained (ordered roughly by typical accuracy)."""

    BROADCAST_METADATA = "broadcast_metadata"  # archive.org TV archive airtime, EXIF, etc.
    ONSCREEN_CLOCK = "onscreen_clock"  # OCR of a broadcast clock bug
    EVENT_ANCHOR = "event_anchor"  # impact / collapse visible in frame
    AUDIO_XCORR = "audio_xcorr"  # cross-correlation with an already-timed clip
    VISUAL_XCORR = "visual_xcorr"  # smoke shape / visual event matching with timed clip
    SOLAR_SHADOW = "solar_shadow"  # shadow direction vs computed sun position
    MANUAL = "manual"
    UNKNOWN = "unknown"


class TimeEstimate(BaseModel):
    """Absolute time in project seconds (see ``wtc4d.timeline``) with 1-sigma
    uncertainty.  ``t`` refers to a specific frame or to the start of a shot,
    depending on where the estimate is attached."""

    t: float = Field(description="seconds since 2001-09-11 00:00:00 EDT")
    sigma: float = Field(ge=0.0, description="1-sigma uncertainty, seconds")
    method: TimeMethod = TimeMethod.UNKNOWN
    evidence: str = ""  # human-readable justification / pointer
    derived_from: list[str] = Field(default_factory=list)  # ids of estimates this depends on
