"""World coordinate frame, tower geometry and landmark registry.

World frame
-----------
A local **East-North-Up (ENU)** tangent frame, in **metres**, with origin at
``WORLD_ORIGIN`` (roughly the midpoint between the two towers at plaza
level).  +X = east, +Y = north, +Z = up.  Everything spatial in this
project (camera poses, gaussians, meshes, landmarks) lives in this frame.

Camera convention
-----------------
OpenCV / COLMAP pinhole: camera +X right, +Y down, +Z forward.  Poses are
stored **camera-to-world** (``c2w``), 4x4 row-major.  See ``wtc4d.schema``.

Accuracy notes
--------------
Values marked ``approx`` are placeholders good to tens of metres, taken
from public sources (Wikipedia, NYC open data, memorial footprints).  The
``geo`` workstream is responsible for replacing them with surveyed values
(NIST structural drawings, NYC DoITT planimetrics, memorial pool footprints,
which coincide with the original tower footprints).
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel

from wtc4d.schema.geometry import LatLonAlt

# --- WGS84 -------------------------------------------------------------------
_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)

WORLD_ORIGIN = LatLonAlt(lat=40.71120, lon=-74.01320, alt_m=0.0)
"""ENU origin: between the towers, at approximately mean sea level.  Plaza
level was roughly +3..+5 m above MSL; the geo workstream sets the datum."""


def geodetic_to_ecef(p: LatLonAlt) -> np.ndarray:
    lat = math.radians(p.lat)
    lon = math.radians(p.lon)
    n = _A / math.sqrt(1.0 - _E2 * math.sin(lat) ** 2)
    x = (n + p.alt_m) * math.cos(lat) * math.cos(lon)
    y = (n + p.alt_m) * math.cos(lat) * math.sin(lon)
    z = (n * (1.0 - _E2) + p.alt_m) * math.sin(lat)
    return np.array([x, y, z], dtype=np.float64)


def _enu_rotation(origin: LatLonAlt) -> np.ndarray:
    lat = math.radians(origin.lat)
    lon = math.radians(origin.lon)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    return np.array(
        [
            [-so, co, 0.0],
            [-sl * co, -sl * so, cl],
            [cl * co, cl * so, sl],
        ]
    )


def latlon_to_enu(p: LatLonAlt, origin: LatLonAlt = WORLD_ORIGIN) -> np.ndarray:
    """Geodetic -> world ENU (metres)."""
    d = geodetic_to_ecef(p) - geodetic_to_ecef(origin)
    return _enu_rotation(origin) @ d


def enu_to_latlon(xyz, origin: LatLonAlt = WORLD_ORIGIN) -> LatLonAlt:
    """World ENU (metres) -> geodetic. Iterative ECEF->geodetic."""
    xyz = np.asarray(xyz, dtype=np.float64)
    ecef = geodetic_to_ecef(origin) + _enu_rotation(origin).T @ xyz
    x, y, z = ecef
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - _E2))
    for _ in range(10):
        n = _A / math.sqrt(1.0 - _E2 * math.sin(lat) ** 2)
        alt = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1.0 - _E2 * n / (n + alt)))
    n = _A / math.sqrt(1.0 - _E2 * math.sin(lat) ** 2)
    alt = p / math.cos(lat) - n
    return LatLonAlt(lat=math.degrees(lat), lon=math.degrees(lon), alt_m=alt)


# --- Towers ------------------------------------------------------------------
class TowerSpec(BaseModel):
    id: str
    name: str
    center: LatLonAlt  # footprint centre at plaza level
    footprint_m: float  # square side
    roof_height_m: float  # roof above plaza level
    top_height_m: float  # antenna / highest point above plaza level
    rotation_deg: float  # rotation of the square footprint about +Z, CCW from east-aligned faces
    floors: int
    impact_floors: tuple[int, int] | None = None
    notes: str = ""

    def enu_center(self) -> np.ndarray:
        return latlon_to_enu(self.center)

    def box_corners(self) -> np.ndarray:
        """(8, 3) ENU corners of the tower as a rotated box from plaza to roof."""
        c = self.enu_center()
        h = self.footprint_m / 2.0
        r = math.radians(self.rotation_deg)
        rot = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
        sq = np.array([[-h, -h], [h, -h], [h, h], [-h, h]]) @ rot.T
        bottom = np.column_stack([sq + c[:2], np.full(4, c[2])])
        top = np.column_stack([sq + c[:2], np.full(4, c[2] + self.roof_height_m)])
        return np.vstack([bottom, top])


# Both towers: 207 ft (63.1 m) square, 110 storeys.  Centre coordinates are
# approx (memorial pools mark the footprints).  Faces were aligned with the
# WTC superblock, which is rotated slightly from true north; ``rotation_deg``
# is approx and MUST be verified by the geo workstream.
WTC1 = TowerSpec(
    id="WTC1",
    name="One World Trade Center (North Tower)",
    center=LatLonAlt(lat=40.71195, lon=-74.01338, alt_m=0.0),
    footprint_m=63.1,
    roof_height_m=417.0,
    top_height_m=526.3,  # rooftop antenna
    rotation_deg=0.0,  # approx
    floors=110,
    impact_floors=(93, 99),
    notes="Approx centre; verify against memorial North Pool footprint.",
)
WTC2 = TowerSpec(
    id="WTC2",
    name="Two World Trade Center (South Tower)",
    center=LatLonAlt(lat=40.71055, lon=-74.01285, alt_m=0.0),
    footprint_m=63.1,
    roof_height_m=415.1,
    top_height_m=415.1,  # observation deck roof, no antenna
    rotation_deg=0.0,  # approx
    floors=110,
    impact_floors=(77, 85),
    notes="Approx centre; verify against memorial South Pool footprint.",
)
TOWERS: list[TowerSpec] = [WTC1, WTC2]


# --- Landmarks ----------------------------------------------------------------
class Landmark(BaseModel):
    """A fixed, identifiable 3D point useful for camera registration.

    ``point`` is the geodetic location of the identifiable feature itself
    (e.g. a spire tip), not the building footprint.  ``height_m`` is the
    feature height above local ground for convenience.
    """

    id: str
    name: str
    point: LatLonAlt
    height_m: float
    kind: str = "building_top"  # building_top | spire | bridge_tower | statue | corner | other
    existed_on_2001_09_11: bool = True
    approx: bool = True
    notes: str = ""

    def enu(self) -> np.ndarray:
        return latlon_to_enu(self.point)


def _lm(
    id: str, name: str, lat: float, lon: float, h: float, kind: str = "building_top", **kw
) -> Landmark:
    return Landmark(
        id=id, name=name, point=LatLonAlt(lat=lat, lon=lon, alt_m=h), height_m=h, kind=kind, **kw
    )


# All approx; the geo workstream verifies and extends this registry (aim for
# 50+ landmarks with <5 m accuracy, including Jersey City / Brooklyn / Midtown).
LANDMARKS: list[Landmark] = [
    _lm("wtc1_roof_ne", "WTC1 roof, NE corner", 40.71223, -74.01300, 417.0, "corner"),
    _lm("wtc1_antenna", "WTC1 antenna tip", 40.71195, -74.01338, 526.3, "spire"),
    _lm("wtc2_roof_ne", "WTC2 roof, NE corner", 40.71083, -74.01247, 415.1, "corner"),
    _lm("wtc7_roof", "7 WTC (original, 1987) roof", 40.71360, -74.01200, 174.0),
    _lm(
        "wfc1_roof",
        "1 World Financial Center (200 Liberty St) pyramid top",
        40.71190,
        -74.01550,
        175.0,
    ),
    _lm(
        "wfc2_roof",
        "2 World Financial Center (225 Liberty St) dome top",
        40.71310,
        -74.01550,
        197.0,
    ),
    _lm(
        "wfc3_roof",
        "3 World Financial Center (200 Vesey St) pyramid top",
        40.71390,
        -74.01520,
        225.0,
    ),
    _lm("wfc4_roof", "4 World Financial Center (250 Vesey St) roof", 40.71470, -74.01640, 150.0),
    _lm("verizon_roof", "Verizon Building (140 West St) roof", 40.71400, -74.01300, 152.0),
    _lm("millenium_hilton_roof", "Millenium Hilton roof", 40.71130, -74.01070, 175.0),
    _lm("one_liberty_plaza_roof", "One Liberty Plaza roof", 40.70950, -74.01110, 226.0),
    _lm(
        "deutsche_bank_roof",
        "Deutsche Bank Building (130 Liberty St) roof",
        40.70980,
        -74.01340,
        158.0,
    ),
    _lm("woolworth_spire", "Woolworth Building spire", 40.71237, -74.00814, 241.0, "spire"),
    _lm("forty_wall_spire", "40 Wall Street spire", 40.70730, -74.00960, 282.0, "spire"),
    _lm(
        "brooklyn_bridge_manhattan_tower",
        "Brooklyn Bridge, Manhattan tower top",
        40.70707,
        -73.99870,
        84.0,
        "bridge_tower",
    ),
    _lm(
        "brooklyn_bridge_brooklyn_tower",
        "Brooklyn Bridge, Brooklyn tower top",
        40.70430,
        -73.99340,
        84.0,
        "bridge_tower",
    ),
    _lm("statue_of_liberty_torch", "Statue of Liberty torch", 40.68925, -74.04450, 93.0, "statue"),
    _lm(
        "empire_state_spire",
        "Empire State Building antenna tip",
        40.74844,
        -73.98565,
        443.0,
        "spire",
    ),
    _lm("chrysler_spire", "Chrysler Building spire", 40.75174, -73.97557, 319.0, "spire"),
    _lm("101_hudson_jc_roof", "101 Hudson Street, Jersey City roof", 40.71760, -74.03250, 167.0),
    _lm(
        "goldman_30_hudson_jc",
        "30 Hudson Street, Jersey City (built 2004; must NOT appear)",
        40.71400,
        -74.03360,
        238.0,
        existed_on_2001_09_11=False,
    ),
]
LANDMARKS_BY_ID: dict[str, Landmark] = {lm.id: lm for lm in LANDMARKS}


# --- Environment on the day ----------------------------------------------------
WIND_NOTES = (
    "Surface wind from the NW at roughly 10-15 kt; smoke plumes drifted SE/SSE over "
    "Brooklyn. Aloft (at tower height) wind was more westerly. Verify with LGA/EWR METARs."
)
