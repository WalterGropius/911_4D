"""Builders for the committed derived data in ``data/geo/``.

These are the only functions in the package that need the network.  They are
driven by the ``wtc4d geo`` CLI; the runtime API (``load_scene``,
``load_landmarks``, ``render_view``) reads only the committed outputs.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from wtc4d.geo.city import FT, _float, _int, _ring_to_enu, build_city_records
from wtc4d.geo.paths import DATA_DIR
from wtc4d.geo.sources import BBox, socrata, within_box
from wtc4d.geo.wtc import PLAZA_ELEVATION_M, wtc_building, wtc_complex_buildings
from wtc4d.schema.geometry import LatLonAlt
from wtc4d.world import WORLD_ORIGIN, Landmark, enu_to_latlon

# Buildings worth a named landmark.  Keyed by DoITT ``bin``; every one of these
# stood on 2001-09-11.  Heights and footprints come from NYC DoITT BUILDING
# (dataset 5zhs-2jue): roof elevation = ground_elevation + height_roof, both in
# feet, converted to metres above NAVD88.
NAMED_BINS: dict[str, tuple[str, str]] = {
    "1000057": ("wfc1_roof", "1 World Financial Center (200 Liberty St) roof"),
    "1000058": ("wfc2_roof", "2 World Financial Center (225 Liberty St) roof"),
    "1000053": ("wfc_winter_garden", "Winter Garden Atrium roof"),
    "1001068": ("one_liberty_plaza_roof", "One Liberty Plaza roof"),
    "1086510": ("wtc7_new_roof", "7 World Trade Center (2006) roof"),
}

# Landmarks that NYC planimetrics cannot supply: outside the five boroughs, not
# buildings, or destroyed before the dataset existed.  ``alt_m`` is metres
# above MSL/NAVD88.
MANUAL_LANDMARKS: list[dict[str, Any]] = [
    dict(
        id="statue_of_liberty_torch",
        name="Statue of Liberty, torch flame",
        lat=40.689247,
        lon=-74.044502,
        alt_m=92.99,
        height_m=92.99,
        kind="statue",
        sigma_m=3.0,
        source="NPS: 305 ft 1 in (92.99 m) from foundation base to torch; Liberty Island ground ~2 m",
    ),
    dict(
        id="colgate_clock_jc",
        name="Colgate Clock, Jersey City (centre of the dial)",
        lat=40.712478,
        lon=-74.035289,
        alt_m=15.0,
        height_m=15.0,
        kind="other",
        sigma_m=6.0,
        source="OSM node; 50 ft octagonal dial on the Exchange Place waterfront",
    ),
    dict(
        id="brooklyn_bridge_manhattan_tower",
        name="Brooklyn Bridge, Manhattan tower top",
        lat=40.70706,
        lon=-73.99873,
        alt_m=84.3,
        height_m=84.3,
        kind="bridge_tower",
        sigma_m=3.0,
        source="Brooklyn Bridge towers rise 276 ft 6 in (84.3 m) above mean high water",
    ),
    dict(
        id="brooklyn_bridge_brooklyn_tower",
        name="Brooklyn Bridge, Brooklyn tower top",
        lat=40.70439,
        lon=-73.99338,
        alt_m=84.3,
        height_m=84.3,
        kind="bridge_tower",
        sigma_m=3.0,
        source="Brooklyn Bridge towers rise 276 ft 6 in (84.3 m) above mean high water",
    ),
    dict(
        id="manhattan_bridge_manhattan_tower",
        name="Manhattan Bridge, Manhattan tower top",
        lat=40.70866,
        lon=-73.99195,
        alt_m=102.0,
        height_m=102.0,
        kind="bridge_tower",
        sigma_m=6.0,
        source="Manhattan Bridge towers ~335 ft (102 m) above water; position from OSM",
    ),
    dict(
        id="manhattan_bridge_brooklyn_tower",
        name="Manhattan Bridge, Brooklyn tower top",
        lat=40.70418,
        lon=-73.98708,
        alt_m=102.0,
        height_m=102.0,
        kind="bridge_tower",
        sigma_m=6.0,
        source="Manhattan Bridge towers ~335 ft (102 m) above water; position from OSM",
    ),
    dict(
        id="empire_state_spire",
        name="Empire State Building, antenna tip",
        lat=40.748441,
        lon=-73.985664,
        alt_m=458.0,
        height_m=443.2,
        kind="spire",
        sigma_m=4.0,
        source="1,454 ft (443.2 m) to the antenna tip above a ground elevation of ~14.8 m",
    ),
    dict(
        id="empire_state_roof",
        name="Empire State Building, 102nd floor observatory roof",
        lat=40.748441,
        lon=-73.985664,
        alt_m=396.0,
        height_m=381.0,
        kind="building_top",
        sigma_m=4.0,
        source="1,250 ft (381 m) architectural roof above ~14.8 m ground",
    ),
    dict(
        id="chrysler_spire",
        name="Chrysler Building, spire tip",
        lat=40.751652,
        lon=-73.975311,
        alt_m=333.0,
        height_m=318.9,
        kind="spire",
        sigma_m=4.0,
        source="1,046 ft (318.9 m) to the spire above ~14 m ground",
    ),
    dict(
        id="metlife_roof",
        name="MetLife Building (200 Park Ave) roof",
        lat=40.754444,
        lon=-73.976389,
        alt_m=261.0,
        height_m=246.3,
        kind="building_top",
        sigma_m=6.0,
        source="808 ft (246.3 m) roof above ~15 m ground",
    ),
    dict(
        id="citicorp_roof",
        name="Citigroup Center (601 Lexington Ave), top of the sloped roof",
        lat=40.758611,
        lon=-73.970556,
        alt_m=294.0,
        height_m=279.0,
        kind="building_top",
        sigma_m=8.0,
        source="915 ft (279 m) above ~15 m ground; the 45-degree crown is unmistakable in skyline shots",
    ),
    dict(
        id="101_hudson_jc_roof",
        name="101 Hudson Street, Jersey City, roof",
        lat=40.717695,
        lon=-74.033327,
        alt_m=170.0,
        height_m=167.0,
        kind="building_top",
        sigma_m=8.0,
        source="548 ft (167 m), completed 1992; ground ~3 m (USGS 3DEP)",
    ),
    dict(
        id="newport_tower_jc_roof",
        name="Newport Tower, Jersey City, roof",
        lat=40.726944,
        lon=-74.035,
        alt_m=155.0,
        height_m=152.4,
        kind="building_top",
        sigma_m=8.0,
        source="500 ft (152.4 m), completed 1990; ground ~3 m",
    ),
    dict(
        id="ten_exchange_place_jc_roof",
        name="10 Exchange Place, Jersey City, roof",
        lat=40.716389,
        lon=-74.032222,
        alt_m=141.0,
        height_m=138.4,
        kind="building_top",
        sigma_m=8.0,
        source="454 ft (138.4 m), completed 1989; ground ~3 m",
    ),
    dict(
        id="goldman_30_hudson_jc",
        name="30 Hudson Street, Jersey City (Goldman Sachs Tower)",
        lat=40.7125,
        lon=-74.0325,
        alt_m=241.0,
        height_m=238.0,
        kind="building_top",
        sigma_m=10.0,
        existed_on_2001_09_11=False,
        source="Completed 2004 -- MUST NOT appear in the 2001 scene",
    ),
    dict(
        id="one_wtc_spire",
        name="One World Trade Center (2014) spire tip",
        lat=40.713,
        lon=-74.0134,
        alt_m=546.2,
        height_m=541.3,
        kind="spire",
        sigma_m=10.0,
        existed_on_2001_09_11=False,
        source="Completed 2014 -- MUST NOT appear in the 2001 scene",
    ),
    dict(
        id="st_nicholas_church",
        name="St Nicholas Greek Orthodox Church (155 Cedar St) roof",
        lat=40.708630,
        lon=-74.013730,
        alt_m=13.0,
        height_m=9.0,
        kind="building_top",
        sigma_m=8.0,
        source="Four-storey 1832 townhouse church destroyed on 2001-09-11; position from the 155 Cedar St lot",
    ),
]


def _roof_elev_m(row: dict) -> float | None:
    h = _float(row.get("height_roof"))
    g = _float(row.get("ground_elevation"))
    if h is None or h <= 0:
        return None
    return (g or 0.0) * FT + h * FT


def fetch_tall_buildings(radius_m: float = 7000.0, min_height_ft: float = 280.0) -> list[dict]:
    """Every DoITT footprint taller than ``min_height_ft`` near the WTC."""
    box = BBox.around(WORLD_ORIGIN.lat, WORLD_ORIGIN.lon, radius_m)
    return socrata(
        "5zhs-2jue",
        select="the_geom,doitt_id,bin,base_bbl,construction_year,height_roof,ground_elevation,shape_area",
        where=f"{within_box(box)} AND height_roof > {min_height_ft}",
        order="doitt_id",
    )


def fetch_addresses(bbls: list[str]) -> dict[str, str]:
    """PLUTO addresses for a set of BBLs (used to name auto-derived landmarks)."""
    out: dict[str, str] = {}
    uniq = sorted({b for b in bbls if b})
    for i in range(0, len(uniq), 200):
        chunk = uniq[i : i + 200]
        quoted = ",".join(f"'{b}'" for b in chunk)
        for r in socrata("64uk-42ks", select="bbl,address", where=f"bbl in ({quoted})"):
            bbl = str(r.get("bbl", "")).split(".")[0]
            if bbl and r.get("address"):
                out[bbl] = str(r["address"]).title()
    return out


def _wtc_landmarks() -> list[Landmark]:
    """Roof corners, centres and the mast tip of the WTC complex."""
    out: list[Landmark] = []
    quadrant = ("sw", "se", "ne", "nw")  # footprint vertex order in the site frame
    for b in wtc_complex_buildings():
        if b.kind == "slab":
            continue
        fp = b.footprint()
        for i, p in enumerate(fp):
            ll = enu_to_latlon([p[0], p[1], b.roof_elev_m])
            tag = quadrant[i] if len(fp) == 4 else str(i)
            out.append(
                Landmark(
                    id=f"{b.id.lower()}_roof_{tag}",
                    name=f"{b.name} roof corner ({tag.upper()})",
                    point=LatLonAlt(lat=ll.lat, lon=ll.lon, alt_m=round(b.roof_elev_m, 2)),
                    height_m=round(b.roof_elev_m - b.base_elev_m, 2),
                    kind="corner",
                    approx=b.approx,
                    sigma_m=round(math.hypot(b.sigma_m, 1.5), 1),
                    source=b.source,
                    notes="Site-frame footprint corner; see data/geo/wtc_complex.yaml",
                )
            )
        if b.top_elev_m > b.roof_elev_m:
            c = fp.mean(axis=0)
            ll = enu_to_latlon([c[0], c[1], b.top_elev_m])
            out.append(
                Landmark(
                    id=f"{b.id.lower()}_antenna",
                    name=f"{b.name} rooftop mast tip",
                    point=LatLonAlt(lat=ll.lat, lon=ll.lon, alt_m=round(b.top_elev_m, 2)),
                    height_m=round(b.top_elev_m - b.base_elev_m, 2),
                    kind="spire",
                    approx=False,
                    sigma_m=3.0,
                    source=b.source,
                    notes="Mast tip, on the tower centre line",
                )
            )
    return out


def _doitt_landmarks(rows: list[dict], addresses: dict[str, str], limit: int) -> list[Landmark]:
    """Roof-centre landmarks for the tallest 2001 buildings in the DoITT data."""
    cands: list[tuple[float, dict, np.ndarray]] = []
    for r in rows:
        year = _int(r.get("construction_year"))
        if year is not None and year > 2001:
            continue
        roof = _roof_elev_m(r)
        if roof is None:
            continue
        geom = r.get("the_geom") or {}
        coords = geom.get("coordinates") or []
        if geom.get("type") == "Polygon":
            coords = [coords]
        if not coords or not coords[0]:
            continue
        ring = _ring_to_enu(coords[0][0])
        if len(ring) < 3:
            continue
        cands.append((roof, r, ring))
    cands.sort(key=lambda t: -t[0])

    out: list[Landmark] = []
    seen: set[str] = set()
    for roof, r, ring in cands:
        bin_ = str(r.get("bin") or "")
        bbl = str(r.get("base_bbl") or "")
        lid, name = NAMED_BINS.get(bin_, ("", ""))
        if not lid:
            addr = addresses.get(bbl) or f"BIN {bin_}"
            lid = "roof_" + (addr.lower().replace(" ", "_").replace(".", "").replace(",", ""))
            name = f"{addr} roof"
        if lid in seen:
            continue
        seen.add(lid)
        c = ring.mean(axis=0)
        ll = enu_to_latlon([c[0], c[1], roof])
        ground = (_float(r.get("ground_elevation")) or 0.0) * FT
        out.append(
            Landmark(
                id=lid,
                name=name,
                point=LatLonAlt(lat=round(ll.lat, 7), lon=round(ll.lon, 7), alt_m=round(roof, 2)),
                height_m=round(roof - ground, 2),
                kind="building_top",
                approx=True,
                sigma_m=4.0,
                source=(
                    "NYC DoITT BUILDING (5zhs-2jue): footprint centroid, "
                    "alt = ground_elevation + height_roof"
                ),
                notes=f"bin {bin_}; built {_int(r.get('construction_year'))}",
            )
        )
        if len(out) >= limit:
            break
    return out


def build_landmarks(*, limit_doitt: int = 45, radius_m: float = 7000.0) -> list[Landmark]:
    """Assemble the landmark registry (needs the network)."""
    rows = fetch_tall_buildings(radius_m=radius_m)
    addresses = fetch_addresses([str(r.get("base_bbl") or "") for r in rows])
    out = _wtc_landmarks()
    out += _doitt_landmarks(rows, addresses, limit_doitt)
    for m in MANUAL_LANDMARKS:
        d = dict(m)
        out.append(
            Landmark(
                id=d.pop("id"),
                name=d.pop("name"),
                point=LatLonAlt(lat=d.pop("lat"), lon=d.pop("lon"), alt_m=d.pop("alt_m")),
                height_m=d.pop("height_m"),
                kind=d.pop("kind", "building_top"),
                approx=True,
                **d,
            )
        )
    by_id: dict[str, Landmark] = {}
    for lm in out:
        by_id[lm.id] = lm
    return sorted(by_id.values(), key=lambda lm: lm.id)


def build_geojson(**kw) -> tuple[dict, list[dict]]:
    """Build the city and return ``(geojson, records)``."""
    from wtc4d.geo.scene import records_to_geojson

    records = build_city_records(**kw)
    return records_to_geojson(records), records


def export_glb(scene, path) -> int:
    """Write a scene to a binary glTF file; returns the byte size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = scene.export(file_type="glb")
    path.write_bytes(data)
    return len(data)


__all__ = [
    "DATA_DIR",
    "PLAZA_ELEVATION_M",
    "build_geojson",
    "build_landmarks",
    "export_glb",
    "fetch_addresses",
    "fetch_tall_buildings",
    "wtc_building",
]
