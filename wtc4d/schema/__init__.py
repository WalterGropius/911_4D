"""Shared data contract between workstreams.

Everything that crosses a package boundary is one of these pydantic models,
serialised as JSON (one object per line in ``.jsonl`` files, or a single
object in ``.json``).  Keep models additive: add optional fields, do not
rename or remove without a PR that updates every consumer.
"""

from wtc4d.schema.camera import CameraIntrinsics, CameraPose, CameraPrior
from wtc4d.schema.corpus import Frame, Shot, Source, SourceKind
from wtc4d.schema.geometry import LatLonAlt
from wtc4d.schema.scene import CameraRef, SceneManifest, SplatAsset, TimelineEvent
from wtc4d.schema.time import TimeEstimate, TimeMethod

__all__ = [
    "CameraIntrinsics",
    "CameraPose",
    "CameraPrior",
    "CameraRef",
    "Frame",
    "LatLonAlt",
    "SceneManifest",
    "Shot",
    "Source",
    "SourceKind",
    "SplatAsset",
    "TimeEstimate",
    "TimeMethod",
    "TimelineEvent",
]
