"""Tests for the parametric WTC complex meshes. Synthetic-only, no network."""

from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from wtc4d.geo import wtc  # noqa: E402
from wtc4d.world import WTC1, WTC2, latlon_to_enu  # noqa: E402


def test_site_frame_roundtrip():
    pts = np.array([[10.0, -30.0], [-120.0, 200.0], [0.0, 0.0]])
    enu = wtc.site_uv_to_enu(pts)
    back = wtc.enu_to_site_uv(enu)
    assert np.allclose(back, pts, atol=1e-6)


def test_wtc1_wtc2_footprints_match_world_towerspec():
    """wtc4d.geo.wtc must agree with the verified centres in wtc4d.world."""
    b1 = wtc.wtc_building("WTC1")
    c1 = b1.footprint().mean(axis=0)
    expected1 = latlon_to_enu(WTC1.center)[:2]
    assert np.linalg.norm(c1 - expected1) < 1.0

    b2 = wtc.wtc_building("WTC2")
    c2 = b2.footprint().mean(axis=0)
    expected2 = latlon_to_enu(WTC2.center)[:2]
    assert np.linalg.norm(c2 - expected2) < 1.0


def test_tower_side_length_and_height():
    b1 = wtc.wtc_building("WTC1")
    fp = b1.footprint()
    side = np.linalg.norm(fp[1] - fp[0])
    assert abs(side - 63.14) < 0.1
    assert abs(b1.height_m - 417.0) < 0.1
    assert abs(b1.top_elev_m - b1.roof_elev_m - (526.3 - 417.0)) < 0.1


def test_wtc_complex_meshes_watertight():
    meshes = wtc.wtc_complex_meshes()
    assert "WTC1" in meshes and "WTC2" in meshes and "WTC7" in meshes
    for name, m in meshes.items():
        if name.endswith("_MAST"):
            continue  # open cylinder caps are fine
        assert m.is_watertight, f"{name} is not watertight"
        assert m.volume > 0


def test_tower_mesh_bounds():
    m = wtc.tower_mesh("WTC1", with_mast=True)
    assert m.bounds[1][2] > wtc.wtc_building("WTC1").top_elev_m - 1.0


def test_towers_are_separated_and_not_overlapping():
    c1 = wtc.wtc_building("WTC1").footprint().mean(axis=0)
    c2 = wtc.wtc_building("WTC2").footprint().mean(axis=0)
    d = np.linalg.norm(c1 - c2)
    assert 100.0 < d < 200.0  # towers ~123 m apart centre to centre
