"""Registry of known vantage points (``CameraPrior``) for 2001-09-11 footage.

The records live in ``data/cameras/priors.yaml`` and are loaded into
:class:`CameraPriorRecord`, a superset of the shared
:class:`wtc4d.schema.camera.CameraPrior`.  The extra fields (``alt_sigma_m``,
``kind``, ``region``, ``confidence``, ``sources``) are local to the camreg
workstream: a consumer that parses the same YAML with the shared model simply
ignores them, so nothing in ``wtc4d.schema`` had to change.

A prior is a *soft constraint*, never an answer.  Ground vantage points are
neighbourhoods (a promenade, a park), not points; aircraft priors cover a whole
orbit.  ``position_sigma_m`` is sized accordingly and
:func:`wtc4d.camreg.pnp.solve_pose` uses it as the weight of a soft penalty,
so a well-observed frame is free to move far from the prior while a frame with
three landmarks is not.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from pydantic import Field

from wtc4d.schema.camera import CameraPrior
from wtc4d.schema.geometry import LatLonAlt
from wtc4d.world import WTC1, WTC2, latlon_to_enu

DEFAULT_PRIORS_PATH = Path(__file__).resolve().parents[2] / "data" / "cameras" / "priors.yaml"

__all__ = [
    "DEFAULT_PRIORS_PATH",
    "CameraPriorRecord",
    "bearing_to_wtc_deg",
    "get_prior",
    "load_priors",
    "priors_by_id",
    "range_to_wtc_m",
]


class CameraPriorRecord(CameraPrior):
    """A vantage point with camreg-local metadata.

    Inherits ``id``, ``name``, ``location``, ``position_sigma_m``, ``moving``
    and ``notes`` from :class:`wtc4d.schema.camera.CameraPrior`.
    """

    alt_sigma_m: float = 50.0
    """1-sigma uncertainty of ``location.alt_m`` (metres)."""

    kind: str = "ground"
    """helicopter | fixed_wing | rooftop | observatory | ground | bridge |
    vessel | vehicle | waterfront."""

    region: str = "manhattan"
    """manhattan | brooklyn | new_jersey | harbour | midtown."""

    confidence: str = "low"
    """high | medium | low -- how well established the vantage point is."""

    sources: list[str] = Field(default_factory=list)

    def enu(self) -> np.ndarray:
        """Prior camera centre in world ENU metres (3,)."""
        return latlon_to_enu(self.location)

    def to_camera_prior(self) -> CameraPrior:
        """Downcast to the shared schema model (drops the camreg-local fields)."""
        return CameraPrior(
            id=self.id,
            name=self.name,
            location=self.location,
            position_sigma_m=self.position_sigma_m,
            moving=self.moving,
            notes=self.notes,
        )


def _wtc_centroid() -> np.ndarray:
    """Midpoint of the two tower footprints, at half roof height (ENU metres).

    Used as the nominal "look at" target for prior-derived viewing directions.
    """
    a, b = WTC1.enu_center(), WTC2.enu_center()
    mid = (a + b) / 2.0
    mid[2] = (WTC1.roof_height_m + WTC2.roof_height_m) / 4.0
    return mid


def bearing_to_wtc_deg(prior: CameraPriorRecord) -> float:
    """Compass bearing (degrees clockwise from north) from the prior to the WTC."""
    d = _wtc_centroid() - prior.enu()
    return math.degrees(math.atan2(d[0], d[1])) % 360.0


def range_to_wtc_m(prior: CameraPriorRecord) -> float:
    """Horizontal distance from the prior to the WTC centroid (metres)."""
    d = _wtc_centroid() - prior.enu()
    return float(math.hypot(d[0], d[1]))


def elevation_to_wtc_top_deg(prior: CameraPriorRecord) -> float:
    """Elevation angle (degrees) from the prior to the top of WTC1."""
    target = WTC1.enu_center().copy()
    target[2] = WTC1.roof_height_m
    d = target - prior.enu()
    return math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))


def load_priors(path: str | Path | None = None) -> list[CameraPriorRecord]:
    """Parse ``priors.yaml``.  Raises on duplicate ids or unparseable records."""
    p = Path(path) if path is not None else DEFAULT_PRIORS_PATH
    with open(p, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    items = raw.get("priors", [])
    out: list[CameraPriorRecord] = []
    seen: set[str] = set()
    for item in items:
        rec = CameraPriorRecord.model_validate(item)
        if rec.id in seen:
            raise ValueError(f"duplicate camera prior id: {rec.id}")
        seen.add(rec.id)
        out.append(rec)
    return out


@lru_cache(maxsize=4)
def _cached_priors(path: str) -> tuple[CameraPriorRecord, ...]:
    return tuple(load_priors(path))


def priors_by_id(path: str | Path | None = None) -> dict[str, CameraPriorRecord]:
    """All priors keyed by id (cached)."""
    key = str(Path(path) if path is not None else DEFAULT_PRIORS_PATH)
    return {p.id: p for p in _cached_priors(key)}


def get_prior(prior_id: str, path: str | Path | None = None) -> CameraPriorRecord:
    """Look one prior up by id."""
    table = priors_by_id(path)
    if prior_id not in table:
        raise KeyError(f"unknown camera prior {prior_id!r}; known: {sorted(table)}")
    return table[prior_id]


def nearest_priors(
    location: LatLonAlt, k: int = 3, path: str | Path | None = None
) -> list[tuple[float, CameraPriorRecord]]:
    """The ``k`` priors closest to ``location``, as ``(distance_m, prior)``."""
    here = latlon_to_enu(location)
    scored = [(float(np.linalg.norm(p.enu() - here)), p) for p in priors_by_id(path).values()]
    scored.sort(key=lambda t: t[0])
    return scored[:k]
