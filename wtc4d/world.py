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

Vertical datum
--------------
``Z`` (and every ``LatLonAlt.alt_m`` in this module) is **metres above
MSL/NAVD88**, which at The Battery agree to about 0.1 m.  Ground in Lower
Manhattan is roughly +2 to +10 m; the Austin J. Tobin Plaza (the WTC plaza,
the datum for the tower heights below) was at +8.23 m.

Accuracy notes
--------------
The WTC numbers below were verified by the ``geo`` workstream against the
September 11 Memorial pool footprints (which are laid out on the original
tower footprints), NIST NCSTAR 1 / 1A and NYC DoITT planimetrics; see
``wtc4d/geo/README.md`` for the accuracy budget.  ``LANDMARKS`` here remains a
small bootstrap list -- the full registry is ``data/geo/landmarks.json``, and
``wtc4d.geo.load_landmarks()`` merges the two (JSON wins on ``id``).
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
"""ENU origin, at mean sea level (MSL/NAVD88), between the two towers.

Verified by the geo workstream: the exact midpoint of the two tower centres is
40.711584 N, 74.013128 W, i.e. 46.1 m NNE of this origin.  The origin is kept
unchanged because its only job is to be *fixed* -- moving it would shift every
ENU coordinate produced by the other workstreams for no accuracy gain.  The
plaza datum is :data:`wtc4d.geo.wtc.PLAZA_ELEVATION_M` = 8.23 m."""


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
    center_sigma_m: float = 3.0  # 1-sigma horizontal uncertainty of ``center``
    height_sigma_m: float = 1.0  # 1-sigma uncertainty of the height values
    source: str = ""  # provenance of the numbers above

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


# Tower geometry, verified by the geo workstream (2026):
#   * Footprint side 207 ft 2 in = 63.14 m square -- NIST NCSTAR 1-1 sec. 2.2.
#   * Centres = the centroids of the two September 11 Memorial reflecting
#     pools (OpenStreetMap ways 697722178 / 697722181), which are laid out on
#     the original tower footprints.  Their eight sides fit an axis-aligned
#     square in the site frame to within 0.2 m.  OSM agrees with NYC DoITT
#     planimetrics in this block to 0.8 m mean / 3 m scatter over 31 matched
#     buildings, hence center_sigma_m = 2.5.
#   * rotation_deg = -29.118: the pool sides run at azimuth 29.118 deg and
#     119.118 deg (the Manhattan/WTC grid).  A square has 90 deg symmetry, so
#     +60.882 describes the same footprint.
#   * alt_m = 8.23 m: Austin J. Tobin Plaza above MSL, the datum for the roof
#     and antenna heights (Port Authority WTC datum el. 327 ft, WTC datum =
#     MSL + 300 ft; cross-checked against NYC DoITT ground_elevation and a
#     USGS 3DEP sample -- see wtc4d/geo/README.md).
#   * Roof heights 1,368 ft / 1,362 ft and impact floors: NIST NCSTAR 1.
WTC1 = TowerSpec(
    id="WTC1",
    name="One World Trade Center (North Tower)",
    center=LatLonAlt(lat=40.7121392, lon=-74.0131756, alt_m=8.23),
    footprint_m=63.14,
    roof_height_m=417.0,
    top_height_m=526.3,  # 1,727 ft rooftop TV mast tip (sources give 1,727-1,728 ft)
    rotation_deg=-29.118,
    floors=110,
    impact_floors=(93, 99),
    center_sigma_m=2.5,
    height_sigma_m=1.0,
    source="OSM memorial North Pool (way 697722178); NIST NCSTAR 1 / 1-1",
    notes="Heights are above the plaza (center.alt_m). Roof 425.23 m above MSL.",
)
WTC2 = TowerSpec(
    id="WTC2",
    name="Two World Trade Center (South Tower)",
    center=LatLonAlt(lat=40.7110296, lon=-74.0130805, alt_m=8.23),
    footprint_m=63.14,
    roof_height_m=415.1,
    top_height_m=415.1,  # observation deck roof, no antenna
    rotation_deg=-29.118,
    floors=110,
    impact_floors=(77, 85),
    center_sigma_m=2.5,
    height_sigma_m=1.0,
    source="OSM memorial South Pool (way 697722181); NIST NCSTAR 1 / 1-1",
    notes="Heights are above the plaza (center.alt_m). Roof 423.33 m above MSL.",
)
TOWERS: list[TowerSpec] = [WTC1, WTC2]


# --- Landmarks ----------------------------------------------------------------
class Landmark(BaseModel):
    """A fixed, identifiable 3D point useful for camera registration.

    ``point`` is the geodetic location of the identifiable feature itself
    (e.g. a spire tip), not the building footprint.  ``point.alt_m`` is metres
    above MSL/NAVD88 -- the same datum as the world frame's ``Z`` -- so
    :meth:`enu` is directly usable as a PnP object point.  ``height_m`` is the
    feature height above the *local ground*, and is metadata only.
    """

    id: str
    name: str
    point: LatLonAlt
    height_m: float
    kind: str = "building_top"  # building_top | spire | bridge_tower | statue | corner | other
    existed_on_2001_09_11: bool = True
    approx: bool = True
    notes: str = ""
    sigma_m: float = 10.0  # 1-sigma 3D position uncertainty, metres
    source: str = ""  # where the position and height came from

    def enu(self) -> np.ndarray:
        return latlon_to_enu(self.point)


def _lm(
    id: str,
    name: str,
    lat: float,
    lon: float,
    alt_m: float,
    height_m: float,
    kind: str = "building_top",
    **kw,
) -> Landmark:
    """``alt_m`` is metres above MSL; ``height_m`` is above local ground."""
    return Landmark(
        id=id,
        name=name,
        point=LatLonAlt(lat=lat, lon=lon, alt_m=alt_m),
        height_m=height_m,
        kind=kind,
        **kw,
    )


# Small bootstrap list, kept for backward compatibility; the full 50+ entry
# registry verified by the geo workstream is data/geo/landmarks.json --
# use wtc4d.geo.load_landmarks() (JSON entries win on id).
LANDMARKS: list[Landmark] = [
    _lm(
        "wtc1_antenna",
        "WTC1 rooftop mast tip",
        40.7121391,
        -74.0131756,
        534.53,
        108.3,
        "spire",
        sigma_m=3.0,
        source="wtc4d.geo.wtc (site frame, memorial North Pool)",
    ),
    _lm(
        "wtc1_roof_ne",
        "WTC1 roof, NE corner",
        40.7125258,
        -74.0133202,
        425.23,
        417.0,
        "corner",
        sigma_m=3.5,
        source="wtc4d.geo.wtc (site frame, memorial North Pool)",
    ),
    _lm(
        "wtc2_roof_ne",
        "WTC2 roof, NE corner",
        40.7114163,
        -74.0132251,
        423.33,
        415.1,
        "corner",
        sigma_m=3.5,
        source="wtc4d.geo.wtc (site frame, memorial South Pool)",
    ),
    _lm(
        "statue_of_liberty_torch",
        "Statue of Liberty torch",
        40.689247,
        -74.044502,
        92.99,
        92.99,
        "statue",
        sigma_m=3.0,
        source="NPS: 305 ft 1 in from foundation base to torch",
    ),
    _lm(
        "empire_state_spire",
        "Empire State Building antenna tip",
        40.748441,
        -73.985664,
        458.0,
        443.2,
        "spire",
        sigma_m=4.0,
        source="1,454 ft to the antenna tip above ~14.8 m ground",
    ),
    _lm(
        "goldman_30_hudson_jc",
        "30 Hudson Street, Jersey City (built 2004; must NOT appear)",
        40.7125,
        -74.0325,
        241.0,
        238.0,
        existed_on_2001_09_11=False,
        source="Completed 2004",
    ),
]
LANDMARKS_BY_ID: dict[str, Landmark] = {lm.id: lm for lm in LANDMARKS}


# --- Environment on the day ----------------------------------------------------
WIND_NOTES = (
    "Surface wind from the NW at roughly 10-15 kt; smoke plumes drifted SE/SSE over "
    "Brooklyn. Aloft (at tower height) wind was more westerly. Verify with LGA/EWR METARs."
)
