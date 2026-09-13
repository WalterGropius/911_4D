"""Tests for scene assembly, landmarks and terrain. Synthetic/committed data only."""

from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from wtc4d.geo.landmarks import (  # noqa: E402
    landmark_enu_points,
    landmarks_existing_on_2001_09_11,
    load_landmarks,
)
from wtc4d.geo.scene import Building, load_buildings, load_scene, scene_metadata  # noqa: E402
from wtc4d.geo.terrain import ground_elevation_m_enu  # noqa: E402
from wtc4d.geo.wtc import PLAZA_ELEVATION_M  # noqa: E402
from wtc4d.world import WORLD_ORIGIN, latlon_to_enu  # noqa: E402


def test_load_scene_is_enu_and_contains_towers():
    scene = load_scene()
    assert isinstance(scene, trimesh.Scene)
    names = set(scene.geometry.keys())
    assert "WTC1" in names and "WTC2" in names
    # tower roof corners must land near the world-frame origin (a few hundred m)
    verts = scene.geometry["WTC1"].vertices
    assert np.all(np.linalg.norm(verts[:, :2], axis=1) < 300.0)
    assert verts[:, 2].max() > 400.0  # roof above plaza


def test_scene_metadata_matches_geometry():
    scene = load_scene()
    for name, geom in scene.geometry.items():
        assert geom.metadata.get("id") == name
        assert "source" in geom.metadata or geom.metadata.get("id", "").endswith("_MAST")


def test_load_buildings_returns_pydantic_models():
    buildings = load_buildings()
    assert len(buildings) >= 5
    assert all(isinstance(b, Building) for b in buildings)
    ids = {b.id for b in buildings}
    assert "WTC1" in ids


def test_scene_summary_lists_towers_as_tallest():
    meta = scene_metadata()
    top_ids = {t["id"] for t in meta["tallest"]}
    assert "WTC1" in top_ids or "WTC1_MAST" in top_ids


def test_landmarks_merge_and_enu():
    lms = load_landmarks()
    assert len(lms) >= 6  # bootstrap list at minimum; data/geo/landmarks.json adds far more
    ids, pts = landmark_enu_points()
    assert pts.shape == (len(ids), 3)
    assert np.isfinite(pts).all()


def test_landmarks_2001_filter_excludes_future_buildings():
    lms = load_landmarks()
    by_id = {lm.id: lm for lm in lms}
    if "goldman_30_hudson_jc" in by_id:
        assert by_id["goldman_30_hudson_jc"].existed_on_2001_09_11 is False
    existing = landmarks_existing_on_2001_09_11()
    assert all(lm.existed_on_2001_09_11 for lm in existing)
    assert len(existing) <= len(lms)


def test_terrain_water_and_land():
    # far offshore in the harbor -> water
    assert ground_elevation_m_enu(-4000.0, -4000.0) == pytest.approx(0.0, abs=0.5)
    # WTC site -> land, roughly plaza-ish elevation, within a broad tolerance
    z = ground_elevation_m_enu(0.0, 0.0)
    assert 0.0 <= z < 30.0


def test_plaza_elevation_matches_world_origin_scale():
    # sanity: plaza is a few metres above MSL, not absurd
    assert 0.0 < PLAZA_ELEVATION_M < 20.0
    origin_enu = latlon_to_enu(WORLD_ORIGIN)
    assert np.allclose(origin_enu, [0.0, 0.0, 0.0], atol=1e-6)
