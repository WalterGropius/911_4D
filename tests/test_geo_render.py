"""Tests for render_view: synthetic scene, no network, CPU-only rasteriser."""

from __future__ import annotations

import math

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from wtc4d.geo.render import render_view  # noqa: E402
from wtc4d.schema.camera import CameraIntrinsics  # noqa: E402


def _looking_down_camera(height: float, width_px: int = 64, height_px: int = 64):
    """c2w for a camera at (0,0,height) looking straight down (-Z), OpenCV axes."""
    # camera +Z (forward) = world -Z; camera +X = world +X; camera +Y = world -Y
    m = np.eye(4)
    m[:3, 0] = [1.0, 0.0, 0.0]
    m[:3, 1] = [0.0, -1.0, 0.0]
    m[:3, 2] = [0.0, 0.0, -1.0]
    m[:3, 3] = [0.0, 0.0, height]
    fx = width_px  # ~45 deg-ish hfov for a small synthetic box
    intr = CameraIntrinsics(
        width=width_px, height=height_px, fx=fx, fy=fx, cx=width_px / 2.0, cy=height_px / 2.0
    )
    return m, intr


def _single_box_scene(size: float = 10.0, height: float = 20.0):
    box = trimesh.creation.box(extents=[size, size, height])
    box.apply_translation([0.0, 0.0, height / 2.0])
    scene = trimesh.Scene()
    scene.add_geometry(box, geom_name="TESTBOX")
    return scene


def test_render_view_hits_known_box_at_center_pixel():
    scene = _single_box_scene(size=10.0, height=20.0)
    m, intr = _looking_down_camera(height=100.0)
    res = render_view(m, intr, scene, backend="raster")
    cy, cx = intr.height // 2, intr.width // 2
    assert res["instance_id"][cy, cx] == 0
    assert math.isfinite(res["depth"][cy, cx])
    assert abs(res["depth"][cy, cx] - 80.0) < 1.0  # 100 - 20 (box top)


def test_render_view_misses_background_corner():
    scene = _single_box_scene(size=10.0, height=20.0)
    m, intr = _looking_down_camera(height=100.0)
    res = render_view(m, intr, scene, backend="raster")
    assert res["instance_id"][0, 0] == -1
    assert not math.isfinite(res["depth"][0, 0])


def test_raster_and_raycast_agree_on_hit_pixels():
    scene = _single_box_scene(size=10.0, height=20.0)
    m, intr = _looking_down_camera(height=100.0, width_px=32, height_px=32)
    raster = render_view(m, intr, scene, backend="raster")
    raycast = render_view(m, intr, scene, backend="raycast")
    hit_r = raster["instance_id"] >= 0
    hit_c = raycast["instance_id"] >= 0
    # allow a small mismatch at the silhouette edge (rasteriser pixel centers
    # vs. per-pixel ray casting can disagree by a border pixel)
    disagree = np.count_nonzero(hit_r != hit_c)
    assert disagree <= max(4, int(0.02 * hit_r.size))
    both = hit_r & hit_c
    assert np.allclose(raster["depth"][both], raycast["depth"][both], atol=1.0)


def test_render_view_on_real_scene_hits_wtc1():
    from wtc4d.geo.scene import load_scene
    from wtc4d.geo.wtc import wtc_building

    scene = load_scene()
    b = wtc_building("WTC1")
    centre = b.footprint().mean(axis=0)
    top = np.array([centre[0], centre[1], b.top_elev_m])
    eye = top + np.array([0.0, -400.0, 150.0])
    fwd = top - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, down, fwd, eye
    intr = CameraIntrinsics(width=128, height=128, fx=200.0, fy=200.0, cx=64.0, cy=64.0)
    res = render_view(m, intr, scene, backend="raster")
    assert (res["instance_id"] >= 0).sum() > 20  # tower fills a visible patch
