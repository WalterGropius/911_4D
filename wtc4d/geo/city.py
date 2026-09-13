"""LOD1 city model for Lower Manhattan, corrected to 2001-09-11.

Pipeline
--------
1. Fetch NYC DoITT planimetric building footprints (``BUILDING``) inside the
   reconstruction box.  Each carries ``construction_year``, ``height_roof``
   (feet above local ground) and ``ground_elevation`` (feet, NAVD88).
2. Fetch NYC ``BUILDING_HISTORIC`` and re-add every footprint that was built
   on or before 2001 and demolished *after* 2001 (130 Liberty St / Deutsche
   Bank, Fiterman Hall, ...), so the 2001 skyline is complete.
3. Fetch OSM footprints for the New Jersey and Brooklyn waterfronts, where
   NYC planimetrics stop, using ``height``/``building:levels`` for heights.
4. Drop everything built after 2001, apply the explicit overrides in
   ``data/geo/corrections_2001.yaml``, and add the WTC complex from
   :mod:`wtc4d.geo.wtc`.
5. Extrude each footprint from the terrain to its roof.

Every emitted building records where its geometry and its height came from.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable
from typing import Any

import numpy as np
import yaml

from wtc4d.geo.paths import DATA_DIR
from wtc4d.geo.sources import BBox, fetch_nyc_buildings, fetch_osm_buildings
from wtc4d.schema.geometry import LatLonAlt
from wtc4d.world import WORLD_ORIGIN, latlon_to_enu

FT = 0.3048
REFERENCE_YEAR = 2001
CORRECTIONS_PATH = DATA_DIR / "corrections_2001.yaml"

# Reconstruction box: ~2.6 km around the WTC, which reaches the Jersey City and
# Hoboken waterfronts to the west and Brooklyn Heights / DUMBO to the east.
DEFAULT_RADIUS_M = 2600.0

# OSM is fetched only for the waterfronts NYC planimetrics do not cover (a
# single all-Manhattan Overpass query for this box size is large enough to be
# timeout-prone, and would only duplicate DoITT anyway).  Boxes are
# (south, west, north, east) hand-picked to cover the camera-relevant
# waterfront strip without asking Overpass for all of Jersey City or Brooklyn.
WATERFRONT_BOXES: dict[str, BBox] = {
    "jersey_city": BBox(40.700, -74.045, 40.735, -74.028),
    "brooklyn": BBox(40.690, -73.998, 40.715, -73.983),
}

# Median storey height used when OSM gives only `building:levels`.  3.2 m is a
# common LOD1 convention for mixed residential/commercial stock.
OSM_LEVEL_HEIGHT_M = 3.2

MIN_AREA_M2 = 40.0
MIN_HEIGHT_M = 3.0


def reconstruction_box(radius_m: float = DEFAULT_RADIUS_M) -> BBox:
    return BBox.around(WORLD_ORIGIN.lat, WORLD_ORIGIN.lon, radius_m)


# --- corrections -------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def load_corrections() -> dict[str, Any]:
    """Parsed ``data/geo/corrections_2001.yaml``."""
    if not CORRECTIONS_PATH.exists():
        return {"drop_bins": [], "drop_doitt_ids": [], "keep_bins": [], "overrides": {}, "add": []}
    doc = yaml.safe_load(CORRECTIONS_PATH.read_text()) or {}
    doc.setdefault("drop_bins", [])
    doc.setdefault("drop_doitt_ids", [])
    doc.setdefault("keep_bins", [])
    doc.setdefault("overrides", {})
    doc.setdefault("add", [])
    return doc


def _as_str_set(values: Iterable[Any]) -> set[str]:
    return {str(v).strip() for v in values if str(v).strip()}


# --- geometry helpers --------------------------------------------------------
def _ring_to_enu(ring) -> np.ndarray:
    """[[lon, lat], ...] -> (N, 2) ENU metres."""
    pts = [latlon_to_enu(LatLonAlt(lat=float(p[1]), lon=float(p[0]), alt_m=0.0))[:2] for p in ring]
    a = np.asarray(pts, dtype=np.float64)
    if len(a) > 1 and np.allclose(a[0], a[-1]):
        a = a[:-1]
    return a


def _polygon_area(p: np.ndarray) -> float:
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))) / 2.0


def simplify_ring(p: np.ndarray, tol_m: float) -> np.ndarray:
    """Douglas-Peucker on a closed ring, keeping it a valid polygon."""
    if tol_m <= 0 or len(p) <= 4:
        return p
    from shapely.geometry import Polygon

    poly = Polygon(np.vstack([p, p[:1]]))
    if not poly.is_valid:
        poly = poly.buffer(0)
    s = poly.simplify(tol_m, preserve_topology=True)
    if s.is_empty or s.geom_type != "Polygon":
        return p
    out = np.asarray(s.exterior.coords, dtype=np.float64)[:-1]
    return out if len(out) >= 3 else p


# --- record --------------------------------------------------------------
def _record(
    *,
    bid: str,
    name: str,
    footprint: np.ndarray,
    base_elev_m: float,
    roof_elev_m: float,
    year: int | None,
    source: str,
    height_source: str,
    sigma_m: float,
    approx: bool,
) -> dict[str, Any]:
    return {
        "id": bid,
        "name": name,
        "footprint_enu": footprint,
        "base_elev_m": float(base_elev_m),
        "roof_elev_m": float(roof_elev_m),
        "height_m": float(roof_elev_m - base_elev_m),
        "year": year,
        "source": source,
        "height_source": height_source,
        "sigma_m": float(sigma_m),
        "approx": bool(approx),
    }


# --- NYC ---------------------------------------------------------------------
def _nyc_rows_to_records(rows: list[dict], *, historic: bool) -> list[dict]:
    from wtc4d.geo.terrain import ground_elevation_m_enu

    corr = load_corrections()
    drop_bins = _as_str_set(corr["drop_bins"])
    drop_ids = _as_str_set(corr["drop_doitt_ids"])
    keep_bins = _as_str_set(corr["keep_bins"])
    overrides = {str(k): v for k, v in corr["overrides"].items()}

    out: list[dict] = []
    seen_ids: set[str] = set()
    for i, r in enumerate(rows):
        geom = r.get("the_geom") or {}
        coords = geom.get("coordinates") or []
        if geom.get("type") == "Polygon":
            coords = [coords]
        bin_ = str(r.get("bin") or "")
        doitt = str(r.get("doitt_id") or "")
        if bin_ in drop_bins or doitt in drop_ids:
            continue
        year = _int(r.get("construction_year"))
        demo = _int(r.get("demolition_year"))
        if historic:
            # Re-add only what stood on 2001-09-11: demolished after 2001 and
            # (where the year is known) built by 2001.  construction_year 0 in
            # this dataset means "unknown", which for Lower Manhattan stock is
            # overwhelmingly pre-2001.
            if demo is None or demo <= REFERENCE_YEAR:
                continue
            if year is not None and year > REFERENCE_YEAR:
                continue
        else:
            if year is not None and year > REFERENCE_YEAR and bin_ not in keep_bins:
                continue
        ov = overrides.get(bin_) or overrides.get(doitt) or {}
        h_ft = _float(ov.get("height_roof_ft", r.get("height_roof")))
        ground_ft = _float(r.get("ground_elevation"))
        for poly_idx, poly in enumerate(coords):
            if not poly:
                continue
            ring = _ring_to_enu(poly[0])
            if _polygon_area(ring) < MIN_AREA_M2:
                continue
            base = (
                ground_ft * FT
                if ground_ft is not None
                else ground_elevation_m_enu(*ring.mean(axis=0))
            )
            if h_ft is None or h_ft < MIN_HEIGHT_M / FT:
                continue
            key = _unique_key(bin_, doitt, f"{i}_{poly_idx}", seen_ids)
            out.append(
                _record(
                    bid=f"nyc_h_{key}" if historic else f"nyc_{key}",
                    name=str(ov.get("name", "") or ""),
                    footprint=ring,
                    base_elev_m=base,
                    roof_elev_m=base + h_ft * FT,
                    year=year,
                    source=(
                        "NYC DoITT BUILDING_HISTORIC (ipkp-snf6)"
                        if historic
                        else "NYC DoITT BUILDING (5zhs-2jue)"
                    ),
                    height_source=(
                        "corrections_2001.yaml"
                        if "height_roof_ft" in ov
                        else "DoITT height_roof (ft)"
                    ),
                    sigma_m=float(ov.get("sigma_m", 2.0)),
                    approx="height_roof_ft" in ov,
                )
            )
    return out


def _unique_key(bin_: str, doitt: str, index: str, seen: set[str]) -> str:
    """A stable, unique building key.

    NYC DoITT ``bin`` (Building Identification Number) is the more reliable
    key; ``doitt_id`` is frequently ``"0"`` (unknown) in ``BUILDING_HISTORIC``,
    and a handful of ``bin`` values are also ``"0"`` -- the ``index`` fallback
    (a ``<row>_<polygon>`` position, always unique per call) guarantees
    uniqueness in every case, so no two footprints collide under one id
    (which would otherwise silently drop one when building the scene).
    """
    for candidate in (bin_, doitt):
        if candidate and candidate != "0" and candidate not in seen:
            seen.add(candidate)
            return candidate
    key = f"idx{index}"
    seen.add(key)
    return key


def _int(v) -> int | None:
    try:
        i = int(float(v))
    except (TypeError, ValueError):
        return None
    return None if i <= 0 else i


def _float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# --- OSM ---------------------------------------------------------------------
def _osm_height_m(tags: dict) -> tuple[float | None, str]:
    for key in ("height", "building:height", "roof:height"):
        raw = tags.get(key)
        if raw:
            try:
                txt = str(raw).strip().lower().replace("m", "").strip()
                return float(txt), f"OSM {key}"
            except ValueError:
                pass
    for key in ("building:levels", "levels"):
        raw = tags.get(key)
        if raw:
            try:
                return float(str(raw).split(";")[0]) * OSM_LEVEL_HEIGHT_M, f"OSM {key}"
            except ValueError:
                pass
    return None, ""


def _osm_elements_to_records(data: dict, *, box: BBox, exclude: Iterable[str] = ()) -> list[dict]:
    """OSM buildings, used only outside the NYC planimetric footprint."""
    from wtc4d.geo.terrain import ground_elevation_m_enu

    exclude = set(exclude)
    out: list[dict] = []
    for el in data.get("elements", []):
        tags = el.get("tags") or {}
        if tags.get("building") in ("no",):
            continue
        rings = []
        if el.get("type") == "way" and el.get("geometry"):
            rings = [[[p["lon"], p["lat"]] for p in el["geometry"]]]
        elif el.get("type") == "relation":
            for m in el.get("members", []):
                if m.get("role") == "outer" and m.get("geometry"):
                    rings.append([[p["lon"], p["lat"]] for p in m["geometry"]])
        year = _int(tags.get("start_date", "")[:4] if tags.get("start_date") else None)
        if year is not None and year > REFERENCE_YEAR:
            continue
        h, hsrc = _osm_height_m(tags)
        for ring_ll in rings:
            if len(ring_ll) < 4:
                continue
            ring = _ring_to_enu(ring_ll)
            area = _polygon_area(ring)
            if area < MIN_AREA_M2:
                continue
            key = f"osm_{el['type']}_{el['id']}"
            if key in exclude:
                continue
            height = h if h is not None else 6.0
            base = ground_elevation_m_enu(*ring.mean(axis=0))
            out.append(
                _record(
                    bid=key,
                    name=str(tags.get("name", "") or ""),
                    footprint=ring,
                    base_elev_m=base,
                    roof_elev_m=base + height,
                    year=year,
                    source="OpenStreetMap (ODbL)",
                    height_source=hsrc or "assumed 6 m (no OSM height tag)",
                    sigma_m=4.0 if hsrc else 8.0,
                    approx=not hsrc,
                )
            )
    return out


# --- assembly ----------------------------------------------------------------
def _added_records() -> list[dict]:
    """Buildings listed explicitly in ``corrections_2001.yaml``."""
    from wtc4d.geo.wtc import site_uv_to_enu

    out: list[dict] = []
    for a in load_corrections()["add"]:
        if "footprint_uv" in a:
            ring = site_uv_to_enu(np.asarray(a["footprint_uv"], dtype=np.float64))
        elif "footprint_lonlat" in a:
            ring = _ring_to_enu(a["footprint_lonlat"])
        else:
            continue
        base = float(a.get("base_elev_m", 0.0))
        out.append(
            _record(
                bid=a["id"],
                name=a.get("name", a["id"]),
                footprint=np.asarray(ring, dtype=np.float64),
                base_elev_m=base,
                roof_elev_m=base + float(a["height_m"]),
                year=_int(a.get("year")),
                source=a.get("source", "corrections_2001.yaml"),
                height_source=a.get("source", "corrections_2001.yaml"),
                sigma_m=float(a.get("sigma_m", 10.0)),
                approx=bool(a.get("approx", True)),
            )
        )
    return out


def _wtc_records() -> list[dict]:
    from wtc4d.geo.wtc import wtc_complex_buildings

    return [
        _record(
            bid=b.id,
            name=b.name,
            footprint=b.footprint(),
            base_elev_m=b.base_elev_m,
            roof_elev_m=b.roof_elev_m,
            year=None,
            source=b.source or "wtc4d.geo.wtc",
            height_source=b.source or "wtc4d.geo.wtc",
            sigma_m=b.sigma_m,
            approx=b.approx,
        )
        for b in wtc_complex_buildings()
    ]


def build_city_records(
    *,
    radius_m: float = DEFAULT_RADIUS_M,
    include_osm: bool = True,
    simplify_tol_m: float = 0.5,
    refresh: bool = False,
) -> list[dict]:
    """Fetch, correct to 2001 and return LOD1 building records (needs network)."""
    box = reconstruction_box(radius_m)
    recs = _nyc_rows_to_records(fetch_nyc_buildings(box, refresh=refresh), historic=False)
    recs += _nyc_rows_to_records(
        fetch_nyc_buildings(box, historic=True, refresh=refresh), historic=True
    )
    if include_osm:
        nyc_hull = _nyc_coverage(recs)
        for name, wbox in WATERFRONT_BOXES.items():
            data = fetch_osm_buildings(wbox, cache_key=f"waterfront_{name}", refresh=refresh)
            osm = _osm_elements_to_records(data, box=wbox)
            recs += [r for r in osm if not _inside(nyc_hull, r["footprint_enu"].mean(axis=0))]
    recs += _wtc_records()
    recs += _added_records()
    if simplify_tol_m:
        for r in recs:
            r["footprint_enu"] = simplify_ring(r["footprint_enu"], simplify_tol_m)
    return [r for r in recs if len(r["footprint_enu"]) >= 3 and r["height_m"] >= MIN_HEIGHT_M]


def _nyc_coverage(recs: list[dict]):
    """Concave-ish coverage of the NYC planimetric data, to avoid duplicating
    Manhattan buildings with OSM ones."""
    from shapely.geometry import MultiPoint

    pts = [tuple(r["footprint_enu"].mean(axis=0)) for r in recs]
    if len(pts) < 3:
        return None
    return MultiPoint(pts).buffer(60.0).buffer(-20.0)


def _inside(poly, pt) -> bool:
    if poly is None:
        return False
    from shapely.geometry import Point

    return bool(poly.contains(Point(float(pt[0]), float(pt[1]))))
