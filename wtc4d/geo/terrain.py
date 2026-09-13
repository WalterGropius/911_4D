"""Terrain for the 2001 scene: water at 0 m, land from a coarse elevation grid.

Heights are metres above MSL/NAVD88, the same datum as the world frame's ``Z``.

The committed grid ``data/geo/terrain_grid.json`` is a coarse (200 m cell)
sample of the USGS 3DEP 1 m DEM over the reconstruction box, built by
``wtc4d geo terrain``.  3DEP is a *present-day* surface: in Lower Manhattan the
street grid has not changed height since 2001, but the World Trade Center site
itself has (the 2011 memorial plaza sits about 4 m lower than the original
Austin J. Tobin Plaza).  The WTC superblock therefore carries its own datum,
:data:`wtc4d.geo.wtc.PLAZA_ELEVATION_M`, and is excluded from the grid.

If the grid file is missing, :func:`ground_elevation_m` degrades to the
per-zone constants in ``ZONES`` (each with a documented source), so the module
never needs the network at import or test time.
"""

from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass

import numpy as np

from wtc4d.geo.paths import DATA_DIR
from wtc4d.schema.geometry import LatLonAlt
from wtc4d.world import WORLD_ORIGIN, latlon_to_enu

GRID_PATH = DATA_DIR / "terrain_grid.json"
WATER_ELEVATION_M = 0.0


@dataclass(frozen=True)
class Zone:
    """A rectangular ENU zone with one constant ground elevation."""

    id: str
    u_min: float
    u_max: float
    v_min: float
    v_max: float
    elevation_m: float
    sigma_m: float
    source: str

    def contains(self, x: float, y: float) -> bool:
        return self.u_min <= x <= self.u_max and self.v_min <= y <= self.v_max


# Fallback zones, ENU metres relative to WORLD_ORIGIN.  Elevations are the
# median of NYC DoITT BUILDING.ground_elevation (feet, NAVD88) over each zone,
# converted to metres; the New Jersey and Brooklyn values are USGS 3DEP
# samples.  All are coarse: sigma is 2-3 m.
ZONES: tuple[Zone, ...] = (
    Zone(
        "wtc_site",
        -200.0,
        260.0,
        -150.0,
        300.0,
        6.0,
        2.0,
        "NYC DoITT BUILDING.ground_elevation around the WTC superblock (~20 ft)",
    ),
    Zone(
        "lower_manhattan",
        -900.0,
        1500.0,
        -1800.0,
        2200.0,
        5.5,
        3.0,
        "median NYC DoITT BUILDING.ground_elevation, Lower Manhattan (~18 ft)",
    ),
    Zone(
        "jersey_city",
        -3500.0,
        -900.0,
        -1500.0,
        2500.0,
        3.0,
        3.0,
        "USGS 3DEP samples, Jersey City / Hoboken waterfront",
    ),
    Zone(
        "brooklyn",
        1500.0,
        4500.0,
        -3500.0,
        800.0,
        6.0,
        4.0,
        "USGS 3DEP samples, Brooklyn Heights / DUMBO waterfront",
    ),
)


@functools.lru_cache(maxsize=1)
def _grid() -> dict | None:
    if not GRID_PATH.exists():
        return None
    g = json.loads(GRID_PATH.read_text())
    rows = [[np.nan if v is None else float(v) for v in row] for row in g["values"]]
    g["values"] = np.asarray(rows, dtype=np.float64)
    return g


def ground_elevation_m_enu(x: float, y: float) -> float:
    """Ground elevation (m above MSL) at a world ENU ``(x, y)`` position."""
    g = _grid()
    if g is not None:
        j = (float(x) - g["x0"]) / g["cell_m"]
        i = (float(y) - g["y0"]) / g["cell_m"]
        vals = g["values"]
        if 0 <= i <= vals.shape[0] - 1 and 0 <= j <= vals.shape[1] - 1:
            v = _bilinear(vals, i, j)
            if not math.isnan(v):
                return float(v)
    for z in ZONES:
        if z.contains(float(x), float(y)):
            return z.elevation_m
    return WATER_ELEVATION_M


def _bilinear(a: np.ndarray, i: float, j: float) -> float:
    i0, j0 = int(math.floor(i)), int(math.floor(j))
    i1, j1 = min(i0 + 1, a.shape[0] - 1), min(j0 + 1, a.shape[1] - 1)
    di, dj = i - i0, j - j0
    q = np.array([[a[i0, j0], a[i0, j1]], [a[i1, j0], a[i1, j1]]], dtype=np.float64)
    if np.isnan(q).any():
        finite = q[~np.isnan(q)]
        return float(finite.mean()) if finite.size else float("nan")
    return float(
        q[0, 0] * (1 - di) * (1 - dj)
        + q[0, 1] * (1 - di) * dj
        + q[1, 0] * di * (1 - dj)
        + q[1, 1] * di * dj
    )


def ground_elevation_m(lat: float, lon: float) -> float:
    """Ground elevation (m above MSL) at a geodetic position."""
    e = latlon_to_enu(LatLonAlt(lat=lat, lon=lon, alt_m=0.0), WORLD_ORIGIN)
    return ground_elevation_m_enu(float(e[0]), float(e[1]))


def build_terrain_grid(
    *, half_size_m: float = 2600.0, cell_m: float = 200.0, progress=None
) -> dict:
    """Sample the USGS 3DEP point service onto a coarse ENU grid.

    Needs the network; run via ``wtc4d geo terrain``.  Returns the dict that is
    written to ``data/geo/terrain_grid.json``.
    """
    from wtc4d.geo.sources import usgs_elevation
    from wtc4d.world import enu_to_latlon

    n = int(2 * half_size_m / cell_m) + 1
    x0 = y0 = -half_size_m
    values = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            x, y = x0 + j * cell_m, y0 + i * cell_m
            p = enu_to_latlon([x, y, 0.0])
            v = usgs_elevation(p.lat, p.lon)
            if v is not None and -30.0 < v < 400.0:
                values[i, j] = max(v, 0.0) if v > -0.5 else np.nan
            if progress is not None:
                progress(i * n + j + 1, n * n)
    return {
        "description": "coarse USGS 3DEP ground elevation, metres above NAVD88, on the world ENU grid",
        "source": "USGS 3DEP via https://epqs.nationalmap.gov/v1/json (1 m DEM, point sampled)",
        "origin": {"lat": WORLD_ORIGIN.lat, "lon": WORLD_ORIGIN.lon},
        "x0": x0,
        "y0": y0,
        "cell_m": cell_m,
        "shape": [n, n],
        "values": [[None if math.isnan(v) else round(float(v), 2) for v in row] for row in values],
    }
