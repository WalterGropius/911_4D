"""Round-trip tests for wtc4d.camreg.conventions: the plumbing everything else
in camreg depends on."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wtc4d.camreg.conventions import (
    c2w_from_rt,
    c2w_from_w2c,
    camera_center,
    distort,
    invert_rigid,
    look_at,
    look_from_pose,
    opencv_to_opengl,
    opengl_to_opencv,
    pose_from_look,
    project_points,
    rodrigues,
    rodrigues_inverse,
    rt_from_c2w,
    undistort_normalized,
    w2c_from_c2w,
)
from wtc4d.schema.camera import CameraIntrinsics

RNG = np.random.default_rng(42)


def _random_rotation() -> np.ndarray:
    axis = RNG.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = RNG.uniform(-math.pi, math.pi)
    return rodrigues(axis * angle)


def _random_c2w() -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _random_rotation()
    m[:3, 3] = RNG.normal(scale=500.0, size=3)
    return m


@pytest.mark.parametrize("trial", range(20))
def test_rodrigues_round_trip(trial: int) -> None:
    axis = RNG.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = RNG.uniform(0.001, math.pi - 0.01)
    r0 = axis * angle
    R = rodrigues(r0)
    r1 = rodrigues_inverse(R)
    # axis-angle is only unique up to the rotation it represents: compare
    # rotation matrices, not raw vectors.
    np.testing.assert_allclose(rodrigues(r1), R, atol=1e-8)
    assert abs(np.linalg.det(R) - 1.0) < 1e-10
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)


def test_rodrigues_identity() -> None:
    np.testing.assert_allclose(rodrigues([0.0, 0.0, 0.0]), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(rodrigues_inverse(np.eye(3)), [0.0, 0.0, 0.0], atol=1e-10)


def test_rodrigues_near_180_degrees() -> None:
    # the branch that recovers the axis from (R + I)/2 near theta = pi
    axis = np.array([0.0, 0.6, 0.8])
    r0 = axis * (math.pi - 1e-4)
    R = rodrigues(r0)
    r1 = rodrigues_inverse(R)
    np.testing.assert_allclose(rodrigues(r1), R, atol=1e-6)


@pytest.mark.parametrize("trial", range(10))
def test_invert_rigid_is_involution(trial: int) -> None:
    m = _random_c2w()
    m_inv = invert_rigid(m)
    np.testing.assert_allclose(invert_rigid(m_inv), m, atol=1e-8)
    np.testing.assert_allclose(m @ m_inv, np.eye(4), atol=1e-8)


def test_w2c_c2w_round_trip() -> None:
    c2w = _random_c2w()
    w2c = w2c_from_c2w(c2w)
    np.testing.assert_allclose(c2w_from_w2c(w2c), c2w, atol=1e-8)


@pytest.mark.parametrize("trial", range(10))
def test_rt_c2w_round_trip(trial: int) -> None:
    c2w = _random_c2w()
    rvec, tvec = rt_from_c2w(c2w)
    recovered = c2w_from_rt(rvec, tvec)
    np.testing.assert_allclose(recovered, c2w, atol=1e-8)


def test_camera_center_matches_c2w_translation() -> None:
    c2w = _random_c2w()
    np.testing.assert_allclose(camera_center(c2w), c2w[:3, 3])


def test_opencv_opengl_round_trip() -> None:
    c2w = _random_c2w()
    gl = opencv_to_opengl(c2w)
    back = opengl_to_opencv(gl)
    np.testing.assert_allclose(back, c2w, atol=1e-10)
    # the flip only touches rotation; the camera centre is unaffected
    np.testing.assert_allclose(camera_center(gl), camera_center(c2w))


def test_opencv_opengl_flips_forward_and_down() -> None:
    c2w = np.eye(4)
    gl = opencv_to_opengl(c2w)
    # OpenCV +Z forward becomes OpenGL -Z; OpenCV +Y down becomes OpenGL +Y up
    np.testing.assert_allclose(gl[:3, 2], [0, 0, -1])
    np.testing.assert_allclose(gl[:3, 1], [0, -1, 0])
    np.testing.assert_allclose(gl[:3, 0], [1, 0, 0])


@pytest.mark.parametrize("az,el,roll", [(0, 0, 0), (90, 10, 0), (-45, 30, 15), (180, -20, -10)])
def test_look_from_pose_round_trip(az: float, el: float, roll: float) -> None:
    c2w = pose_from_look(np.array([10.0, -20.0, 30.0]), az, el, roll)
    az2, el2, roll2 = look_from_pose(c2w)
    assert abs(((az2 - az + 180) % 360) - 180) < 1e-6
    assert abs(el2 - el) < 1e-6
    assert abs(roll2 - roll) < 1e-6


def test_look_at_points_forward_axis_at_target() -> None:
    pos = np.array([0.0, 0.0, 0.0])
    target = np.array([100.0, 0.0, 0.0])  # due east
    c2w = look_at(pos, target)
    fwd = c2w[:3, 2]
    direction = (target - pos) / np.linalg.norm(target - pos)
    np.testing.assert_allclose(fwd, direction, atol=1e-10)


def test_distort_undistort_round_trip() -> None:
    xn = RNG.uniform(-0.3, 0.3, (50, 2))
    dist = [0.05, -0.01, 0.001, -0.001, 0.0]
    xd = distort(xn, dist)
    recovered = undistort_normalized(xd, dist)
    np.testing.assert_allclose(recovered, xn, atol=1e-6)


def test_distort_zero_is_identity() -> None:
    xn = RNG.uniform(-0.3, 0.3, (10, 2))
    np.testing.assert_allclose(distort(xn, []), xn)
    np.testing.assert_allclose(distort(xn, None), xn)


def test_project_points_center_pixel() -> None:
    """A point straight down the optical axis lands exactly on the principal point."""
    intr = CameraIntrinsics(width=640, height=480, fx=500.0, fy=500.0, cx=319.5, cy=239.5)
    c2w = np.eye(4)
    pt = np.array([[0.0, 0.0, 10.0]])  # +Z forward in OpenCV camera axes == world here
    uv, valid = project_points(pt, c2w, intr)
    assert valid[0]
    np.testing.assert_allclose(uv[0], [319.5, 239.5], atol=1e-8)


def test_project_points_behind_camera_is_invalid() -> None:
    intr = CameraIntrinsics(width=640, height=480, fx=500.0, fy=500.0, cx=319.5, cy=239.5)
    c2w = np.eye(4)
    pt = np.array([[0.0, 0.0, -10.0]])
    _, valid = project_points(pt, c2w, intr)
    assert not valid[0]


def test_project_points_consistent_with_manual_pinhole() -> None:
    intr = CameraIntrinsics(width=640, height=480, fx=800.0, fy=800.0, cx=320.0, cy=240.0)
    c2w = _random_c2w()
    pts = RNG.normal(scale=50.0, size=(30, 3)) + c2w[:3, 3] + c2w[:3, 2] * 200.0
    uv, valid = project_points(pts, c2w, intr)
    w2c = invert_rigid(c2w)
    cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
    expect_u = 800.0 * cam[:, 0] / cam[:, 2] + 320.0
    expect_v = 800.0 * cam[:, 1] / cam[:, 2] + 240.0
    np.testing.assert_allclose(uv[valid, 0], expect_u[valid], atol=1e-8)
    np.testing.assert_allclose(uv[valid, 1], expect_v[valid], atol=1e-8)
