"""Parametric meshes for the World Trade Center complex as of 2001-09-11.

The complex is specified in ``data/geo/wtc_complex.yaml`` in the **WTC site
frame**: a right-handed 2D frame aligned with the WTC superblock, whose axes
are the two tower face directions.  ``u`` runs at azimuth 119.118 deg (toward
Church Street), ``v`` at azimuth 29.118 deg (toward Vesey Street), origin at
:data:`wtc4d.world.WORLD_ORIGIN`.

The grid azimuth is measured from the two September 11 Memorial reflecting
pools, which are laid out on the original tower footprints.  See the YAML
header and ``wtc4d/geo/README.md`` for the full provenance and the accuracy
budget.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from wtc4d.geo.meshes import box_from_center, cylinder, extrude
from wtc4d.geo.paths import DATA_DIR

SITE_GRID_AZIMUTH_DEG = 29.118
"""Bearing (deg east of true north) of the WTC superblock's 'north' axis.

Mean bearing of the eight sides of the two memorial pools (OSM ways
697722178 / 697722181); the individual side bearings agree to 0.19 deg."""

SITE_ROTATION_DEG = 90.0 - (SITE_GRID_AZIMUTH_DEG + 90.0)
"""Rotation of a tower footprint about +Z, CCW from east-aligned faces (deg).

Equal to ``-29.118``; a square has 90 deg symmetry so ``+60.882`` is the same
footprint.  This is the value carried by :class:`wtc4d.world.TowerSpec`."""

PLAZA_ELEVATION_M = 8.23
"""Austin J. Tobin Plaza, metres above MSL/NAVD88 (sigma 1.0 m).

Port Authority 'WTC datum' = MSL + 300.0 ft; the plaza sat at datum el. 327 ft.
Cross-checked against NYC DoITT ``ground_elevation`` on Church Street (21 ft)
and a USGS 3DEP sample at the memorial plaza (4.34 m vs. datum el. 313 ft)."""

SPEC_PATH = DATA_DIR / "wtc_complex.yaml"


# --- site frame --------------------------------------------------------------
def _axes() -> tuple[np.ndarray, np.ndarray]:
    au = math.radians(SITE_GRID_AZIMUTH_DEG + 90.0)
    av = math.radians(SITE_GRID_AZIMUTH_DEG)
    return (
        np.array([math.sin(au), math.cos(au)]),
        np.array([math.sin(av), math.cos(av)]),
    )


def site_uv_to_enu(uv) -> np.ndarray:
    """Site ``(u, v)`` metres -> world ENU ``(east, north)`` metres."""
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    u_hat, v_hat = _axes()
    out = uv[:, :1] * u_hat + uv[:, 1:2] * v_hat
    return out[0] if np.ndim(np.asarray(uv)) == 1 else out


def enu_to_site_uv(en) -> np.ndarray:
    """World ENU ``(east, north)`` metres -> site ``(u, v)`` metres."""
    en = np.atleast_2d(np.asarray(en, dtype=np.float64))[:, :2]
    u_hat, v_hat = _axes()
    return np.column_stack([en @ u_hat, en @ v_hat])


# --- spec --------------------------------------------------------------------
class WTCBuilding(BaseModel):
    """One structure of the WTC complex, already in world ENU metres."""

    id: str
    name: str
    kind: str = "block"
    footprint_enu: list[list[float]] = Field(default_factory=list)
    base_elev_m: float = 0.0
    roof_elev_m: float = 0.0
    top_elev_m: float = 0.0
    floors: int | None = None
    sigma_m: float = 10.0
    approx: bool = True
    source: str = ""
    existed_on_2001_09_11: bool = True

    @property
    def height_m(self) -> float:
        return self.roof_elev_m - self.base_elev_m

    def footprint(self) -> np.ndarray:
        return np.asarray(self.footprint_enu, dtype=np.float64)


@functools.lru_cache(maxsize=1)
def load_wtc_spec() -> dict[str, Any]:
    """Parsed ``data/geo/wtc_complex.yaml``."""
    import yaml

    return yaml.safe_load(SPEC_PATH.read_text())


def _square_uv(center_uv, side: float) -> np.ndarray:
    h = side / 2.0
    cu, cv = float(center_uv[0]), float(center_uv[1])
    return np.array([[cu - h, cv - h], [cu + h, cv - h], [cu + h, cv + h], [cu - h, cv + h]])


@functools.lru_cache(maxsize=1)
def wtc_complex_buildings() -> tuple[WTCBuilding, ...]:
    """The WTC complex in world ENU metres, straight from the spec."""
    spec = load_wtc_spec()
    site = spec["site"]
    out: list[WTCBuilding] = []
    for b in spec["buildings"]:
        if b["shape"] == "square":
            uv = _square_uv(b["center_uv"], b["side_m"])
        elif b["shape"] == "superblock":
            uv = np.asarray(site["superblock_uv"], dtype=np.float64)
        else:
            uv = np.asarray(b["footprint_uv"], dtype=np.float64)
        base = float(b["base_elev_m"])
        roof = base + float(b["roof_height_m"])
        out.append(
            WTCBuilding(
                id=b["id"],
                name=b["name"],
                kind=b.get("kind", "block"),
                footprint_enu=[[float(x), float(y)] for x, y in site_uv_to_enu(uv)],
                base_elev_m=base,
                roof_elev_m=roof,
                top_elev_m=base + float(b.get("top_height_m", b["roof_height_m"])),
                floors=b.get("floors"),
                sigma_m=float(b.get("sigma_m", 10.0)),
                approx=bool(b.get("approx", True)),
                source=b.get("source", ""),
            )
        )
    return tuple(out)


def wtc_building(building_id: str) -> WTCBuilding:
    for b in wtc_complex_buildings():
        if b.id == building_id:
            return b
    raise KeyError(building_id)


def wtc_complex_meshes() -> dict[str, Any]:
    """``{building_id: trimesh.Trimesh}`` for the whole complex, ENU metres.

    The WTC 1 rooftop mast is returned separately as ``"WTC1_MAST"`` so it can
    be shown, hidden or given its own instance id independently of the tower.
    """
    import trimesh

    spec = {b["id"]: b for b in load_wtc_spec()["buildings"]}
    meshes: dict[str, Any] = {}
    for b in wtc_complex_buildings():
        m = extrude(b.footprint(), b.base_elev_m, b.roof_elev_m)
        if m is not None:
            meshes[b.id] = m
        mast = spec[b.id].get("mast")
        if mast and b.top_elev_m > b.roof_elev_m:
            centre = b.footprint().mean(axis=0)
            meshes[f"{b.id}_MAST"] = cylinder(
                centre, float(mast["radius_m"]), b.roof_elev_m, b.top_elev_m
            )
    assert isinstance(meshes.get("WTC1"), trimesh.Trimesh)
    return meshes


def tower_mesh(tower_id: str = "WTC1", *, with_mast: bool = True):
    """Single tower as one mesh (square prism, plus mast for WTC 1)."""
    import trimesh

    b = wtc_building(tower_id)
    parts = [
        box_from_center(
            b.footprint().mean(axis=0), _side(b), SITE_ROTATION_DEG, b.base_elev_m, b.roof_elev_m
        )
    ]
    if with_mast and b.top_elev_m > b.roof_elev_m:
        mast = {x["id"]: x for x in load_wtc_spec()["buildings"]}[tower_id].get("mast")
        if mast:
            parts.append(
                cylinder(
                    b.footprint().mean(axis=0), float(mast["radius_m"]), b.roof_elev_m, b.top_elev_m
                )
            )
    parts = [p for p in parts if p is not None]
    return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]


def _side(b: WTCBuilding) -> float:
    f = b.footprint()
    return float(np.linalg.norm(f[1] - f[0]))


def tower_roof_corners(tower_id: str) -> np.ndarray:
    """(4, 3) ENU roof corners of a tower, ordered S-E-N-W-ish (CCW)."""
    b = wtc_building(tower_id)
    f = b.footprint()
    return np.column_stack([f, np.full(len(f), b.roof_elev_m)])
