"""Assemble the 2001 scene prior as a ``trimesh.Scene`` in world ENU metres."""

from __future__ import annotations

import functools
import json
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from wtc4d.geo.meshes import extrude
from wtc4d.geo.paths import DATA_DIR
from wtc4d.schema.geometry import LatLonAlt
from wtc4d.world import enu_to_latlon, latlon_to_enu

BUILDINGS_PATH = DATA_DIR / "buildings_2001.geojson"


class Building(BaseModel):
    """One LOD1 building of the 2001 scene, in world ENU metres."""

    id: str
    name: str = ""
    footprint_enu: list[list[float]] = Field(default_factory=list)
    base_elev_m: float = 0.0
    roof_elev_m: float = 0.0
    year: int | None = None
    source: str = ""
    height_source: str = ""
    sigma_m: float = 5.0
    approx: bool = True

    @property
    def height_m(self) -> float:
        return self.roof_elev_m - self.base_elev_m

    def footprint(self) -> np.ndarray:
        return np.asarray(self.footprint_enu, dtype=np.float64)

    def mesh(self):
        return extrude(self.footprint(), self.base_elev_m, self.roof_elev_m)

    def metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "height_m": round(self.height_m, 2),
            "base_elev_m": round(self.base_elev_m, 2),
            "roof_elev_m": round(self.roof_elev_m, 2),
            "year": self.year,
            "source": self.source,
            "height_source": self.height_source,
            "sigma_m": self.sigma_m,
            "approx": self.approx,
        }


def records_to_geojson(records: list[dict], *, decimals: int = 6) -> dict:
    """LOD1 records (ENU) -> a WGS84 GeoJSON FeatureCollection."""
    feats = []
    for r in records:
        ring = np.asarray(r["footprint_enu"], dtype=np.float64)
        ll = [enu_to_latlon([x, y, 0.0]) for x, y in ring]
        coords = [[round(p.lon, decimals), round(p.lat, decimals)] for p in ll]
        coords.append(coords[0])
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {
                    "id": r["id"],
                    "name": r.get("name", ""),
                    "height_m": round(float(r["height_m"]), 2),
                    "base_elev_m": round(float(r["base_elev_m"]), 2),
                    "roof_elev_m": round(float(r["roof_elev_m"]), 2),
                    "year": r.get("year"),
                    "source": r.get("source", ""),
                    "height_source": r.get("height_source", ""),
                    "sigma_m": round(float(r.get("sigma_m", 5.0)), 1),
                    "approx": bool(r.get("approx", True)),
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "name": "lower_manhattan_2001",
        "description": (
            "LOD1 building footprints of Lower Manhattan and the surrounding "
            "waterfronts as they stood on 2001-09-11. Heights are metres; "
            "base_elev_m / roof_elev_m are metres above MSL/NAVD88."
        ),
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": feats,
    }


def geojson_to_buildings(doc: dict) -> tuple[Building, ...]:
    out = []
    for f in doc.get("features", []):
        p = f.get("properties", {})
        ring = f["geometry"]["coordinates"][0]
        if len(ring) > 1 and ring[0] == ring[-1]:
            ring = ring[:-1]
        enu = [
            latlon_to_enu(LatLonAlt(lat=float(lat), lon=float(lon), alt_m=0.0))[:2].tolist()
            for lon, lat in ring
        ]
        out.append(
            Building(
                id=p["id"],
                name=p.get("name", "") or "",
                footprint_enu=enu,
                base_elev_m=float(p.get("base_elev_m", 0.0)),
                roof_elev_m=float(p.get("roof_elev_m", p.get("height_m", 0.0))),
                year=p.get("year"),
                source=p.get("source", ""),
                height_source=p.get("height_source", ""),
                sigma_m=float(p.get("sigma_m", 5.0)),
                approx=bool(p.get("approx", True)),
            )
        )
    return tuple(out)


@functools.lru_cache(maxsize=2)
def load_buildings(year: int = 2001) -> tuple[Building, ...]:
    """Buildings from ``data/geo/buildings_2001.geojson`` (plus the WTC complex).

    If the committed GeoJSON is missing (a fresh checkout that has not run
    ``wtc4d geo build``), only the parametric WTC complex is returned, so the
    API and the tests still work without the network.
    """
    if year != 2001:
        raise ValueError("only the 2001 epoch is modelled; pass year=2001")
    if BUILDINGS_PATH.exists():
        return geojson_to_buildings(json.loads(BUILDINGS_PATH.read_text()))
    from wtc4d.geo.wtc import wtc_complex_buildings

    return tuple(
        Building(
            id=b.id,
            name=b.name,
            footprint_enu=b.footprint_enu,
            base_elev_m=b.base_elev_m,
            roof_elev_m=b.roof_elev_m,
            source=b.source,
            height_source=b.source,
            sigma_m=b.sigma_m,
            approx=b.approx,
        )
        for b in wtc_complex_buildings()
    )


def load_scene(year: int = 2001, *, with_mast: bool = True, with_ground: bool = False):
    """The static scene prior as a ``trimesh.Scene``, world ENU metres.

    Each building is a named geometry whose ``metadata`` carries ``id``,
    ``name``, ``height_m``, ``year``, ``source`` and ``sigma_m``.
    """
    import trimesh

    scene = trimesh.Scene()
    for b in load_buildings(year):
        m = b.mesh()
        if m is None:
            continue
        m.metadata.update(b.metadata())
        scene.add_geometry(m, geom_name=b.id)
    if with_mast:
        from wtc4d.geo.wtc import load_wtc_spec, wtc_complex_meshes

        specs = {s["id"]: s for s in load_wtc_spec()["buildings"]}
        for name, mesh in wtc_complex_meshes().items():
            if not name.endswith("_MAST"):
                continue
            base = specs[name[: -len("_MAST")]]
            mesh.metadata.update(
                {
                    "id": name,
                    "name": f"{base['name']} rooftop mast",
                    "height_m": round(base["top_height_m"] - base["roof_height_m"], 2),
                    "source": base.get("source", ""),
                    "approx": True,
                    "sigma_m": 3.0,
                }
            )
            scene.add_geometry(mesh, geom_name=name)
    if with_ground:
        g = ground_mesh()
        if g is not None:
            g.metadata.update({"id": "GROUND", "name": "terrain", "approx": True})
            scene.add_geometry(g, geom_name="GROUND")
    return scene


def ground_mesh(half_size_m: float = 2600.0, cell_m: float = 200.0):
    """Coarse terrain + water surface as a single mesh (optional scene layer)."""
    import trimesh

    from wtc4d.geo.terrain import ground_elevation_m_enu

    n = int(2 * half_size_m / cell_m) + 1
    xs = np.linspace(-half_size_m, half_size_m, n)
    ys = np.linspace(-half_size_m, half_size_m, n)
    zz = np.array([[ground_elevation_m_enu(x, y) for x in xs] for y in ys])
    gx, gy = np.meshgrid(xs, ys)
    verts = np.column_stack([gx.ravel(), gy.ravel(), zz.ravel()])
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            faces += [[a, a + 1, a + n + 1], [a, a + n + 1, a + n]]
    return trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=False)


def scene_metadata(year: int = 2001) -> dict[str, Any]:
    """Summary of what is in the scene: counts, tallest structures, sources."""
    from wtc4d.world import WORLD_ORIGIN

    bs = load_buildings(year)
    tallest = sorted(bs, key=lambda b: -b.roof_elev_m)[:15]
    sources: dict[str, int] = {}
    for b in bs:
        sources[b.source] = sources.get(b.source, 0) + 1
    return {
        "year": year,
        "n_buildings": len(bs),
        "world_origin": {
            "lat": WORLD_ORIGIN.lat,
            "lon": WORLD_ORIGIN.lon,
            "alt_m": WORLD_ORIGIN.alt_m,
        },
        "sources": sources,
        "tallest": [
            {"id": b.id, "name": b.name, "roof_elev_m": round(b.roof_elev_m, 1)} for b in tallest
        ],
    }
