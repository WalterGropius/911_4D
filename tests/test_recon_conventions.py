"""Convention round trips: ours <-> COLMAP (w2c quaternion) <-> nerfstudio (OpenGL)."""

from __future__ import annotations

import numpy as np
import pytest

from wtc4d.recon import conventions as C
from wtc4d.recon.colmap import read_colmap_model, write_colmap_model
from wtc4d.recon.data import scale_intrinsics
from wtc4d.recon.nerfstudio import read_transforms_json, write_transforms_json
from wtc4d.schema.camera import CameraIntrinsics, CameraPose


def _poses(n: int = 3) -> list[CameraPose]:
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        eye = rng.normal(scale=800.0, size=3) + np.array([0.0, 0.0, 300.0])
        c2w = C.look_at_c2w(eye, [0.0, 0.0, 200.0])
        intr = CameraIntrinsics(
            width=720,
            height=480,
            fx=900.0 + 10 * i,
            fy=900.0 + 10 * i,
            cx=360.0,
            cy=240.0,
            model="OPENCV",
            dist=[0.01, -0.002, 0.0, 0.0] if i == 1 else [],
        )
        out.append(
            CameraPose.from_matrix(
                c2w, shot_id=f"shot{i}", frame_idx=10 * i, intrinsics=intr, method="landmark_pnp"
            )
        )
    return out


def test_quaternion_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        r = C.quat_to_rotmat(q)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert pytest.approx(1.0, abs=1e-12) == np.linalg.det(r)
        q2 = C.rotmat_to_quat(r)
        assert np.allclose(C.quat_to_rotmat(q2), r, atol=1e-12)


def test_quaternion_round_trip_degenerate_traces():
    # Shepperd's method branches on which diagonal element dominates; hit all four.
    mats = [
        np.eye(3),
        np.diag([1.0, -1.0, -1.0]),
        np.diag([-1.0, 1.0, -1.0]),
        np.diag([-1.0, -1.0, 1.0]),
    ]
    for m in mats:
        q = C.rotmat_to_quat(m)
        assert np.allclose(C.quat_to_rotmat(q), m, atol=1e-12)


def test_c2w_colmap_round_trip():
    for pose in _poses():
        c2w = pose.matrix()
        q, t = C.c2w_to_colmap(c2w)
        assert np.allclose(C.colmap_to_c2w(q, t), c2w, atol=1e-9)


def test_colmap_translation_maps_world_point_into_camera():
    """x_cam = R(q) @ x_world + t must agree with inverting c2w directly."""
    c2w = C.look_at_c2w([1000.0, 200.0, 350.0], [0.0, 0.0, 200.0])
    q, t = C.c2w_to_colmap(c2w)
    x_world = np.array([12.0, -30.0, 210.0])
    x_cam = C.quat_to_rotmat(q) @ x_world + t
    expected = (C.invert_rigid(c2w) @ np.append(x_world, 1.0))[:3]
    assert np.allclose(x_cam, expected, atol=1e-9)
    assert x_cam[2] > 0  # in front of an OpenCV camera


def test_opengl_round_trip_and_axis_flip():
    c2w = C.look_at_c2w([0.0, -1200.0, 300.0], [0.0, 0.0, 200.0])
    gl = C.c2w_to_opengl(c2w)
    assert np.allclose(C.opengl_to_c2w(gl), c2w, atol=1e-12)
    # OpenGL +Y is up where OpenCV +Y is down; +Z points back along the view ray
    assert np.allclose(gl[:3, 1], -c2w[:3, 1])
    assert np.allclose(gl[:3, 2], -c2w[:3, 2])
    assert np.allclose(gl[:3, 3], c2w[:3, 3])


def test_look_at_is_right_handed_opencv():
    c2w = C.look_at_c2w([500.0, 0.0, 200.0], [0.0, 0.0, 200.0])
    right, down, forward = c2w[:3, 0], c2w[:3, 1], c2w[:3, 2]
    assert np.allclose(np.cross(right, down), forward, atol=1e-9)
    assert forward @ np.array([-1.0, 0.0, 0.0]) > 0.99  # looking back at the origin
    assert down[2] < 0  # +Y is down in world terms


def test_colmap_text_model_round_trip(tmp_path):
    poses = _poses()
    xyz = np.array([[0.0, 0.0, 100.0], [10.0, -5.0, 300.0]])
    rgb = np.array([[0.5, 0.25, 0.125], [1.0, 0.0, 0.0]])
    write_colmap_model(tmp_path / "sparse", poses, xyz, rgb)

    model = read_colmap_model(tmp_path / "sparse")
    assert len(model.images) == len(poses)
    back = model.to_poses()
    for a, b in zip(poses, back, strict=True):
        assert np.allclose(a.matrix(), b.matrix(), atol=1e-6)
        assert a.shot_id == b.shot_id
        assert a.frame_idx == b.frame_idx
        assert b.intrinsics.fx == pytest.approx(a.intrinsics.fx, rel=1e-9)
        assert b.intrinsics.cx == pytest.approx(a.intrinsics.cx, rel=1e-9)
    pts, cols = model.points_array()
    assert np.allclose(pts, xyz, atol=1e-6)
    assert np.allclose(cols, rgb, atol=2 / 255)


def test_colmap_distortion_selects_opencv_model(tmp_path):
    poses = _poses()
    write_colmap_model(tmp_path / "sparse", poses)
    model = read_colmap_model(tmp_path / "sparse")
    models = set(model.camera_models.values())
    assert models == {"PINHOLE", "OPENCV"}
    distorted = [i for i in model.cameras.values() if i.dist]
    assert distorted and distorted[0].dist[:2] == pytest.approx([0.01, -0.002])


def test_transforms_json_round_trip(tmp_path):
    poses = _poses()
    write_transforms_json(tmp_path / "transforms.json", poses)
    back, _ = read_transforms_json(tmp_path / "transforms.json")
    assert len(back) == len(poses)
    for a, b in zip(poses, back, strict=True):
        assert np.allclose(a.matrix(), b.matrix(), atol=1e-9)
        assert (a.shot_id, a.frame_idx) == (b.shot_id, b.frame_idx)
        assert b.intrinsics.width == a.intrinsics.width
        assert b.intrinsics.fy == pytest.approx(a.intrinsics.fy)


def test_transforms_json_carries_times(tmp_path):
    from wtc4d.schema.time import TimeEstimate, TimeMethod

    poses = _poses(1)
    times = {
        (poses[0].shot_id, poses[0].frame_idx): TimeEstimate(
            t=33000.0, sigma=0.75, method=TimeMethod.ONSCREEN_CLOCK
        )
    }
    write_transforms_json(tmp_path / "t.json", poses, times=times, epoch_id="E2")
    _, back = read_transforms_json(tmp_path / "t.json")
    te = back[(poses[0].shot_id, poses[0].frame_idx)]
    assert te.t == pytest.approx(33000.0)
    assert te.sigma == pytest.approx(0.75)
    assert te.method == TimeMethod.ONSCREEN_CLOCK


def test_transforms_json_rejects_reoriented_poses(tmp_path):
    import json

    poses = _poses(1)
    path = tmp_path / "t.json"
    write_transforms_json(path, poses)
    doc = json.loads(path.read_text())
    doc["applied_transform"] = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 5]]
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="applied_transform"):
        read_transforms_json(path)


def test_scale_intrinsics_keeps_pixel_centres():
    intr = CameraIntrinsics(width=720, height=480, fx=900.0, fy=900.0, cx=359.5, cy=239.5)
    half = scale_intrinsics(intr, 2.0)
    assert (half.width, half.height) == (360, 240)
    assert half.fx == pytest.approx(450.0)
    # the image centre must stay the image centre after downscaling
    assert half.cx == pytest.approx(half.width / 2 - 0.5)
    assert half.cy == pytest.approx(half.height / 2 - 0.5)
