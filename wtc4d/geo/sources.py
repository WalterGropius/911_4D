"""Remote data sources for the 2001 scene prior, with on-disk caching.

Nothing in this module is imported at test time with network access; every
function caches into :func:`wtc4d.geo.paths.cache_dir` and raises
:class:`SourceUnavailable` rather than silently returning partial data.

Sources
-------
``NYC_BUILDINGS``  NYC Open Data "BUILDING" (DoITT planimetric building
    footprints, dataset ``5zhs-2jue``).  Attributes used: ``the_geom``,
    ``construction_year``, ``height_roof`` (ft above local ground),
    ``ground_elevation`` (ft, NAVD88), ``bin``, ``base_bbl``, ``doitt_id``.
``NYC_BUILDINGS_HISTORIC``  NYC Open Data "BUILDING_HISTORIC"
    (``ipkp-snf6``) — footprints of demolished/merged buildings, with
    ``demolition_year``.  This is how buildings that stood in 2001 and were
    later demolished (e.g. 130 Liberty St / Deutsche Bank, Fiterman Hall)
    are put back into the 2001 scene.
``OVERPASS``  OpenStreetMap, for the New Jersey and Brooklyn waterfronts
    where NYC planimetrics do not reach, and for the September 11 Memorial
    pool outlines that mark the original tower footprints.
``USGS 3DEP``  https://epqs.nationalmap.gov point elevation service, used to
    sample a coarse terrain grid.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wtc4d.geo.paths import cache_dir

NYC_DOMAIN = "https://data.cityofnewyork.us"
NYC_BUILDINGS = "5zhs-2jue"
NYC_BUILDINGS_HISTORIC = "ipkp-snf6"
NYC_PLUTO = "64uk-42ks"

OVERPASS_MIRRORS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
USGS_EPQS = "https://epqs.nationalmap.gov/v1/json"

USER_AGENT = (
    "911_4D-geo/0.1 (https://github.com/WalterGropius/911_4D; "
    "open-source 2001-09-11 4D reconstruction, geo workstream)"
)


class SourceUnavailable(RuntimeError):
    """A remote source could not be reached or returned unusable data."""


@dataclass(frozen=True)
class BBox:
    """Geographic bounding box in degrees (WGS84)."""

    south: float
    west: float
    north: float
    east: float

    def slug(self) -> str:
        return f"{self.south:.5f}_{self.west:.5f}_{self.north:.5f}_{self.east:.5f}"

    @classmethod
    def around(cls, lat: float, lon: float, radius_m: float) -> BBox:
        """Square box of half-size ``radius_m`` around a point."""
        import math

        dlat = radius_m / 111320.0
        dlon = radius_m / (111320.0 * math.cos(math.radians(lat)))
        return cls(lat - dlat, lon - dlon, lat + dlat, lon + dlon)


def _requests():
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SourceUnavailable(
            "the 'requests' package is required: pip install wtc4d[geo]"
        ) from exc
    return requests


def _get(url: str, *, params: dict[str, Any] | None = None, tries: int = 5, timeout: int = 180):
    requests = _requests()
    last: Exception | None = None
    for i in range(tries):
        try:
            r = requests.get(
                url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT}
            )
            if r.status_code == 200:
                return r
            last = SourceUnavailable(f"{url} -> HTTP {r.status_code}")
        except Exception as exc:  # noqa: BLE001 - network flakiness is expected
            last = exc
        time.sleep(2.0 * (i + 1))
    raise SourceUnavailable(f"GET {url} failed after {tries} tries: {last}")


# --- Socrata ----------------------------------------------------------------
def socrata(
    dataset: str,
    *,
    select: str | None = None,
    where: str | None = None,
    order: str | None = None,
    page: int = 5000,
    max_rows: int = 200_000,
) -> list[dict[str, Any]]:
    """Run a paginated SoQL query against NYC Open Data."""
    out: list[dict[str, Any]] = []
    offset = 0
    while len(out) < max_rows:
        params: dict[str, Any] = {"$limit": page, "$offset": offset}
        if select:
            params["$select"] = select
        if where:
            params["$where"] = where
        params["$order"] = order or ":id"
        rows = _get(f"{NYC_DOMAIN}/resource/{dataset}.json", params=params).json()
        if not isinstance(rows, list):
            raise SourceUnavailable(f"{dataset}: unexpected response {str(rows)[:200]}")
        out.extend(rows)
        if len(rows) < page:
            break
        offset += page
    return out


def within_box(b: BBox) -> str:
    """SoQL ``within_box`` clause for the ``the_geom`` column."""
    return f"within_box(the_geom, {b.north}, {b.west}, {b.south}, {b.east})"


def fetch_nyc_buildings(b: BBox, *, historic: bool = False, refresh: bool = False) -> list[dict]:
    """DoITT building footprints inside ``b``.  Cached as JSON."""
    ds = NYC_BUILDINGS_HISTORIC if historic else NYC_BUILDINGS
    path = cache_dir() / f"nyc_{ds}_{b.slug()}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    cols = [
        "the_geom",
        "doitt_id",
        "bin",
        "base_bbl",
        "construction_year",
        "height_roof",
        "ground_elevation",
        "shape_area",
        "last_status_type",
    ]
    if historic:
        cols.append("demolition_year")
    rows = socrata(ds, select=",".join(cols), where=within_box(b), order="doitt_id")
    path.write_text(json.dumps(rows))
    return rows


def fetch_pluto(b: BBox, *, refresh: bool = False) -> list[dict]:
    """PLUTO tax-lot records (for addresses / storey counts), by lat-lon box."""
    path = cache_dir() / f"pluto_{b.slug()}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    where = (
        f"latitude > {b.south} AND latitude < {b.north} "
        f"AND longitude > {b.west} AND longitude < {b.east}"
    )
    rows = socrata(
        NYC_PLUTO,
        select="bbl,address,numfloors,yearbuilt,latitude,longitude",
        where=where,
        order="bbl",
    )
    path.write_text(json.dumps(rows))
    return rows


# --- Overpass ---------------------------------------------------------------
def overpass(query: str, *, cache_key: str | None = None, refresh: bool = False) -> dict:
    """Run an Overpass QL query, rotating over mirrors.  Optionally cached."""
    path = cache_dir() / f"overpass_{cache_key}.json" if cache_key else None
    if path is not None and path.exists() and not refresh:
        return json.loads(path.read_text())
    requests = _requests()
    last: Exception | None = None
    empty_fallback: dict | None = None
    for attempt in range(3):
        for mirror in OVERPASS_MIRRORS:
            try:
                r = requests.post(
                    mirror,
                    data={"data": query},
                    timeout=300,
                    headers={"User-Agent": USER_AGENT},
                )
                if r.status_code == 200 and r.text.lstrip().startswith("{"):
                    data = r.json()
                    elements = data.get("elements")
                    if elements:
                        if path is not None:
                            path.write_text(json.dumps(data))
                        return data
                    if elements is not None and empty_fallback is None:
                        # structurally valid but empty: could be a genuinely
                        # empty area, or a stale/rate-limited mirror. Keep it
                        # as a fallback but keep trying other mirrors first.
                        empty_fallback = data
                last = SourceUnavailable(f"{mirror}: HTTP {r.status_code} {r.text[:120]}")
            except Exception as exc:  # noqa: BLE001
                last = exc
            time.sleep(3.0)
        time.sleep(10.0 * (attempt + 1))
    if empty_fallback is not None:
        if path is not None:
            path.write_text(json.dumps(empty_fallback))
        return empty_fallback
    raise SourceUnavailable(f"Overpass failed on every mirror: {last}")


def fetch_osm_buildings(b: BBox, *, cache_key: str | None = None, refresh: bool = False) -> dict:
    """OSM building ways/relations with geometry inside ``b``."""
    q = (
        "[out:json][timeout:300];"
        f'(way["building"]({b.south},{b.west},{b.north},{b.east});'
        f'relation["building"]["type"="multipolygon"]({b.south},{b.west},{b.north},{b.east}););'
        "out tags geom;"
    )
    return overpass(q, cache_key=cache_key or f"buildings_{b.slug()}", refresh=refresh)


def fetch_memorial_pools(refresh: bool = False) -> dict:
    """The two September 11 Memorial reflecting pools (tower footprint proxy)."""
    q = (
        "[out:json][timeout:120];"
        'way["name"~"Memorial (North|South) Pool"](40.7090,-74.0160,40.7140,-74.0100);'
        "out tags geom;"
    )
    return overpass(q, cache_key="memorial_pools", refresh=refresh)


# --- USGS 3DEP --------------------------------------------------------------
def usgs_elevation(lat: float, lon: float) -> float | None:
    """Point elevation in metres above NAVD88 from the USGS 3DEP service."""
    try:
        r = _get(
            USGS_EPQS,
            params={"x": lon, "y": lat, "units": "Meters", "wkid": 4326, "includeDate": "false"},
            tries=3,
            timeout=60,
        )
        value = r.json().get("value")
        return None if value in (None, "", "-1000000") else float(value)
    except (SourceUnavailable, ValueError, TypeError):
        return None


def write_json(path: Path, obj: Any, *, indent: int | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, sort_keys=False) + "\n")
    return path
