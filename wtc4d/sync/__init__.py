"""Temporal alignment: absolute time (with 1-sigma uncertainty) for shots and
frames, fused from broadcast metadata, on-screen clocks, event anchors, audio
cross-correlation and solar shadows. See ``wtc4d/sync/README.md`` for the
full pipeline; ``wtc4d sync --help`` for the CLI.

Submodules import their own heavy optional dependencies (OpenCV, RapidOCR,
Pillow, ``internetarchive``) lazily inside the functions that need them, so
``import wtc4d.sync`` and its submodules stay cheap and never require those
extras to be installed just to read this package's types.
"""

from wtc4d.sync.types import (
    BBox,
    EventCandidate,
    EventKind,
    LinearClock,
    PairwiseOffset,
    RoiTrack,
    ShotTimeRecord,
)

__all__ = [
    "BBox",
    "EventCandidate",
    "EventKind",
    "LinearClock",
    "PairwiseOffset",
    "RoiTrack",
    "ShotTimeRecord",
]
