"""Epoch E5: the Ground Zero debris pile (2001-09-11 evening onward).

No aftermath raster is committed: the authoritative products are large and
their licences/hosting change, so this module documents the exact fetch
procedure and provides a loader for a small derived height field once it has
been produced.

Sources
-------
**NOAA / USGS Ground Zero lidar (Sept-Oct 2001).**  Flown for FEMA by EarthData
International; distributed through the USGS and the NOAA Digital Coast archive
of the World Trade Center response.  Several epochs exist (2001-09-15,
2001-09-17, 2001-09-19, 2001-09-23, 2001-09-27 and later), which makes the
pile itself a 4D object.  Fetch:

1. Browse https://coast.noaa.gov/dataviewer/ (or
   https://www.fisheries.noaa.gov/inport/ and search "World Trade Center
   lidar"); the USGS entry point is https://www.usgs.gov/ - search
   "World Trade Center lidar 2001".
2. Download the LAS/ASCII tiles that cover the WTC superblock and put them in
   ``$WTC4D_GEO_CACHE/aftermath/lidar/<date>/``.  **Do not commit them.**
3. Run ``wtc4d geo aftermath --lidar-dir <dir> --date 2001-09-23`` to rasterise
   a 2 m height field clipped to the superblock; the result is a few hundred
   kilobytes and *is* committed as
   ``data/geo/aftermath/pile_<date>_2m.json``.

**NOAA 2001-09-23 orthophoto.**  The NOAA Remote Sensing Division flew Lower
Manhattan on 23 September 2001 at roughly 0.15 m GSD.  The imagery is in the
public domain and reachable from the same NOAA Digital Coast / NGS emergency
response archive ("World Trade Center, September 2001").  It is the reference
for registering aftermath ground photography; store it in
``$WTC4D_GEO_CACHE/aftermath/ortho/``.

Both datasets are vertically referenced to NAVD88, the same datum as the world
frame's ``Z``, so no vertical shift is needed -- only the horizontal
projection (NY State Plane Long Island, ft) has to be converted to ENU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wtc4d.geo.paths import DATA_DIR

AFTERMATH_DIR = DATA_DIR / "aftermath"
PILE_GLOB = "pile_*_2m.json"


def available_pile_dates() -> list[str]:
    """Dates for which a committed derived height field exists."""
    if not AFTERMATH_DIR.exists():
        return []
    return sorted(p.name.split("_")[1] for p in AFTERMATH_DIR.glob(PILE_GLOB))


def load_pile(date: str) -> dict:
    """Load a committed 2 m debris-pile height field (ENU metres)."""
    path = AFTERMATH_DIR / f"pile_{date}_2m.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not present. See wtc4d/geo/aftermath.py for the fetch procedure; "
            f"available: {available_pile_dates() or 'none'}"
        )
    doc = json.loads(path.read_text())
    doc["values"] = np.asarray(
        [[np.nan if v is None else float(v) for v in row] for row in doc["values"]]
    )
    return doc


def rasterise_pile(points_enu: np.ndarray, *, cell_m: float = 2.0, pad_m: float = 40.0) -> dict:
    """Max-height rasterise an ENU point cloud onto a regular grid.

    ``points_enu`` is ``(N, 3)`` in world ENU metres.  Returns the dict that is
    written to ``data/geo/aftermath/pile_<date>_2m.json``.
    """
    p = np.asarray(points_enu, dtype=np.float64).reshape(-1, 3)
    if len(p) == 0:
        raise ValueError("no points")
    x0, y0 = p[:, 0].min() - pad_m, p[:, 1].min() - pad_m
    x1, y1 = p[:, 0].max() + pad_m, p[:, 1].max() + pad_m
    nx = int(np.ceil((x1 - x0) / cell_m)) + 1
    ny = int(np.ceil((y1 - y0) / cell_m)) + 1
    grid = np.full((ny, nx), np.nan)
    j = np.clip(((p[:, 0] - x0) / cell_m).astype(int), 0, nx - 1)
    i = np.clip(((p[:, 1] - y0) / cell_m).astype(int), 0, ny - 1)
    for ii, jj, zz in zip(i, j, p[:, 2], strict=True):
        if np.isnan(grid[ii, jj]) or zz > grid[ii, jj]:
            grid[ii, jj] = zz
    return {
        "description": "Ground Zero debris pile, max-height raster, world ENU metres",
        "source": "NOAA/USGS Ground Zero lidar (2001); see wtc4d/geo/aftermath.py",
        "x0": float(x0),
        "y0": float(y0),
        "cell_m": float(cell_m),
        "shape": [ny, nx],
        "values": [[None if np.isnan(v) else round(float(v), 2) for v in row] for row in grid],
    }


def write_pile(doc: dict, date: str, out_dir: Path = AFTERMATH_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"pile_{date}_2m.json"
    path.write_text(json.dumps(doc) + "\n")
    return path
