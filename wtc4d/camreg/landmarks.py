"""Landmark registry used for 2D-3D correspondences.

Three sources are merged, in increasing order of authority:

1. **Derived tower corners** -- the eight corners of each tower box from
   :func:`wtc4d.world.TowerSpec.box_corners`, named by compass quadrant
   (``wtc1_roof_ne``, ``wtc2_base_sw``, ...).  These are always available and
   are the features an annotator can actually click on in a frame: the vertical
   corner edges of the towers are the single most visible structure in almost
   every shot of that morning.
2. **wtc4d.world.LANDMARKS** -- the shared registry (spires, bridge towers,
   WFC roofs, ...).  Entries here override derived corners with the same id.
3. **wtc4d.geo.load_landmarks()** -- the geo workstream's verified registry,
   when that package is importable.  It overrides everything.

Only landmarks that existed on 2001-09-11 are returned by default;
``include_nonexistent=True`` keeps the rest (useful for *detecting* a
misdated photo: seeing 30 Hudson Street proves the image is post-2004).
"""

from __future__ import annotations

import math

import numpy as np

from wtc4d.world import LANDMARKS, TOWERS, Landmark, TowerSpec, enu_to_latlon

__all__ = ["derived_tower_corners", "landmark_registry", "landmark_points"]

_QUADRANTS = (
    ("s", "w"),  # (north/south label, east/west label) chosen from the sign of the offset
)


def _quadrant_name(offset_xy: np.ndarray) -> str:
    """Compass label ('ne', 'nw', 'se', 'sw') for a corner offset in ENU."""
    ns = "n" if offset_xy[1] >= 0 else "s"
    ew = "e" if offset_xy[0] >= 0 else "w"
    return ns + ew


def derived_tower_corners(towers: list[TowerSpec] | None = None) -> dict[str, Landmark]:
    """Roof and base corners of each tower as :class:`~wtc4d.world.Landmark`.

    Ids are ``<tower_id_lower>_{roof,base}_{ne,nw,se,sw}``, e.g.
    ``wtc1_roof_ne``.  Heights are above plaza level (the world frame's
    ``z``), matching ``TowerSpec.roof_height_m``.
    """
    out: dict[str, Landmark] = {}
    for tower in towers if towers is not None else TOWERS:
        corners = tower.box_corners()  # (8, 3): 4 base then 4 roof
        centre = tower.enu_center()
        for i, pt in enumerate(corners):
            level = "base" if i < 4 else "roof"
            name = _quadrant_name(pt[:2] - centre[:2])
            lid = f"{tower.id.lower()}_{level}_{name}"
            lla = enu_to_latlon(pt)
            out[lid] = Landmark(
                id=lid,
                name=f"{tower.id} {level} corner ({name.upper()})",
                point=lla,
                height_m=float(pt[2]),
                kind="corner",
                approx=True,
                notes=f"Derived from wtc4d.world.{tower.id}.box_corners(); "
                "as approximate as the tower footprint and rotation.",
            )
    return out


def _geo_landmarks() -> dict[str, Landmark]:
    """Landmarks from the geo workstream, or ``{}`` when it is not available."""
    try:  # the geo package is developed in parallel and may not be merged yet
        from wtc4d import geo  # type: ignore[attr-defined]

        loaded = geo.load_landmarks()
    except Exception:  # noqa: BLE001 -- absence, import error and load failure are all "not available"
        return {}
    if isinstance(loaded, dict):
        items = list(loaded.values())
    else:
        items = list(loaded)
    out: dict[str, Landmark] = {}
    for item in items:
        if isinstance(item, Landmark):
            out[item.id] = item
        else:  # duck-typed / different model: re-validate through the shared model
            try:
                out[item.id] = Landmark.model_validate(
                    item if isinstance(item, dict) else item.model_dump()
                )
            except Exception:  # noqa: BLE001
                continue
    return out


def landmark_registry(include_nonexistent: bool = False) -> dict[str, Landmark]:
    """All usable landmarks keyed by id (see the module docstring for merge order)."""
    reg: dict[str, Landmark] = derived_tower_corners()
    reg.update({lm.id: lm for lm in LANDMARKS})
    reg.update(_geo_landmarks())
    if not include_nonexistent:
        reg = {k: v for k, v in reg.items() if v.existed_on_2001_09_11}
    return reg


def landmark_points(ids: list[str], registry: dict[str, Landmark] | None = None) -> np.ndarray:
    """World ENU coordinates (N, 3) for the given landmark ids, in order."""
    reg = registry if registry is not None else landmark_registry()
    missing = [i for i in ids if i not in reg]
    if missing:
        raise KeyError(f"unknown landmark id(s): {missing}")
    return np.array([reg[i].enu() for i in ids], dtype=np.float64).reshape(-1, 3)


def spread_metrics(points: np.ndarray) -> dict[str, float]:
    """Geometric conditioning of a 3D point set used for PnP.

    Returns the three PCA standard deviations (metres, descending) and
    ``planarity`` = sigma3 / sigma1.  A landmark set whose ``planarity`` is
    below ~0.02 is effectively planar (very common: skyline tops seen from far
    away lie close to a single plane, or even a line), which means the
    camera's distance along the optical axis trades off against focal length
    and must be pinned by a prior.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 3:
        return {"sigma1_m": 0.0, "sigma2_m": 0.0, "sigma3_m": 0.0, "planarity": 0.0}
    centred = pts - pts.mean(axis=0)
    sv = np.linalg.svd(centred, compute_uv=False) / math.sqrt(len(pts))
    sigma = list(sv) + [0.0, 0.0, 0.0]
    s1 = max(sigma[0], 1e-12)
    return {
        "sigma1_m": float(sigma[0]),
        "sigma2_m": float(sigma[1]),
        "sigma3_m": float(sigma[2]),
        "planarity": float(sigma[2] / s1),
    }
