"""Static scene prior: Lower Manhattan as it stood on 2001-09-11.

Everything in this package is expressed in the project world frame defined by
:mod:`wtc4d.world`: a local **East-North-Up** tangent frame in **metres** with
origin at ``wtc4d.world.WORLD_ORIGIN``.  ``+X`` east, ``+Y`` north, ``+Z`` up,
with ``Z = 0`` at mean sea level (NAVD88, which at The Battery agrees with
local MSL to about 0.1 m).

Public API
----------
``load_scene(year=2001)``      -> :class:`trimesh.Scene` of the city, ENU metres
``load_landmarks()``           -> merged landmark registry (JSON wins over ``world``)
``render_view(c2w, intr)``     -> ``dict(depth, instance_id, normals)``
``ground_elevation_m(lat,lon)``-> terrain height, metres above MSL

See ``wtc4d/geo/README.md`` for data sources, the accuracy budget and how to
regenerate the derived files in ``data/geo/``.
"""

from __future__ import annotations

from wtc4d.geo.landmarks import (
    landmark_enu_points,
    landmarks_existing_on_2001_09_11,
    load_landmarks,
)
from wtc4d.geo.render import render_view
from wtc4d.geo.scene import Building, load_buildings, load_scene, scene_metadata
from wtc4d.geo.terrain import ground_elevation_m, ground_elevation_m_enu
from wtc4d.geo.wtc import PLAZA_ELEVATION_M, wtc_complex_buildings, wtc_complex_meshes

__all__ = [
    "PLAZA_ELEVATION_M",
    "Building",
    "ground_elevation_m",
    "ground_elevation_m_enu",
    "landmark_enu_points",
    "landmarks_existing_on_2001_09_11",
    "load_buildings",
    "load_landmarks",
    "load_scene",
    "render_view",
    "scene_metadata",
    "wtc_complex_buildings",
    "wtc_complex_meshes",
]
