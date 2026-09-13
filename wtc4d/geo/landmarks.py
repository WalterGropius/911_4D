"""Landmark registry: fixed 3D points usable as PnP correspondences.

``data/geo/landmarks.json`` is the authoritative registry; it is merged over
:data:`wtc4d.world.LANDMARKS` (entries in the JSON win on ``id``) so that the
bootstrap list in the shared module stays valid while this workstream owns the
detail.

Conventions
-----------
``Landmark.point.alt_m`` is **metres above MSL/NAVD88** (the same datum as the
world frame's ``Z``), so ``Landmark.enu()`` is directly usable as an object
point.  ``Landmark.height_m`` is the feature's height above the *local ground*
and is metadata only.  ``sigma_m`` is a 1-sigma 3D position uncertainty.

Landmarks that did not exist on 2001-09-11 are kept in the registry with
``existed_on_2001_09_11=False`` so that registration code can explicitly
exclude them (and so nobody re-adds them by accident).
"""

from __future__ import annotations

import functools
import json

import numpy as np

from wtc4d.geo.paths import DATA_DIR
from wtc4d.world import LANDMARKS as WORLD_LANDMARKS
from wtc4d.world import Landmark

REGISTRY_PATH = DATA_DIR / "landmarks.json"


@functools.lru_cache(maxsize=1)
def load_landmarks() -> tuple[Landmark, ...]:
    """All landmarks, ``data/geo/landmarks.json`` overriding ``world.LANDMARKS``."""
    merged: dict[str, Landmark] = {lm.id: lm for lm in WORLD_LANDMARKS}
    if REGISTRY_PATH.exists():
        doc = json.loads(REGISTRY_PATH.read_text())
        for raw in doc["landmarks"]:
            lm = Landmark.model_validate(raw)
            merged[lm.id] = lm
    return tuple(merged.values())


def landmarks_by_id() -> dict[str, Landmark]:
    return {lm.id: lm for lm in load_landmarks()}


def landmarks_existing_on_2001_09_11() -> tuple[Landmark, ...]:
    """Only the landmarks that were physically present on the morning."""
    return tuple(lm for lm in load_landmarks() if lm.existed_on_2001_09_11)


def landmark_enu_points(
    landmarks: tuple[Landmark, ...] | None = None,
) -> tuple[list[str], np.ndarray]:
    """``(ids, (N, 3) ENU metres)`` for use as PnP object points."""
    lms = landmarks if landmarks is not None else landmarks_existing_on_2001_09_11()
    ids = [lm.id for lm in lms]
    pts = np.array([lm.enu() for lm in lms], dtype=np.float64).reshape(-1, 3)
    return ids, pts


def save_landmarks(landmarks, path=REGISTRY_PATH, *, note: str = "") -> int:
    """Write the registry JSON (used by ``wtc4d geo landmarks``)."""
    lms = sorted(landmarks, key=lambda lm: lm.id)
    doc = {
        "$schema": "wtc4d.world.Landmark",
        "note": note
        or (
            "Landmark registry for camera registration. point.alt_m is metres above "
            "MSL/NAVD88; height_m is the feature height above local ground; sigma_m is "
            "a 1-sigma 3D position uncertainty in metres."
        ),
        "count": len(lms),
        "landmarks": [lm.model_dump() for lm in lms],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n")
    load_landmarks.cache_clear()
    return len(lms)
