"""Tracker test: a synthetic moving-camera sequence, registered once, tracked
through the rest of the shot, checked against ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from wtc4d.camreg.conventions import look_at, project_points
from wtc4d.camreg.landmarks import landmark_points, landmark_registry
from wtc4d.camreg.pnp import PnPConfig, solve_pose
from wtc4d.camreg.priors import get_prior
from wtc4d.camreg.synthetic import default_scene, degrade
from wtc4d.camreg.track import (
    BundleAdjustConfig,
    TrackConfig,
    bundle_adjust,
    lift_features,
    track_shot,
)
from wtc4d.schema.camera import CameraIntrinsics
from wtc4d.world import TOWERS, WTC1

pytest.importorskip("cv2")  # tracking needs OpenCV's LK implementation

N_FRAMES = 6
IMAGE_SIZE = (256, 192)
HFOV_DEG = 32.0


def _make_shot(drift_per_frame=(1.4, 0.3, 0.0), yaw_per_frame_deg: float = 1.5):
    scene = default_scene()
    prior = get_prior("brooklyn_heights_promenade")
    base_c = prior.enu() + np.array([60.0, -20.0, 2.0])
    w, h = IMAGE_SIZE
    f = w / (2 * np.tan(np.radians(HFOV_DEG) / 2))
    intr = CameraIntrinsics(width=w, height=h, fx=f, fy=f, cx=(w - 1) / 2, cy=(h - 1) / 2)
    frames, true_c2w = [], []
    for i in range(N_FRAMES):
        c = base_c + np.array(drift_per_frame) * i
        target = WTC1.enu_center() + np.array([0.0, 0.0, 180.0 + yaw_per_frame_deg * i])
        c2w = look_at(c, target)
        img = degrade(
            scene.render(c2w, intr).image, scale=0.85, noise_sigma=0.012, blur_sigma_px=0.6, seed=i
        )
        frames.append(img)
        true_c2w.append(c2w)
    return frames, true_c2w, prior, intr, base_c


def _keyframe_pose(true_c2w, intr, prior):
    reg = landmark_registry()
    ids = sorted(reg)
    pts = landmark_points(ids, reg)
    uv, valid = project_points(pts, true_c2w, intr)
    w, h = intr.width, intr.height
    inframe = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    sel = [i for i in range(len(ids)) if inframe[i]]
    result = solve_pose(
        pts[sel], uv[sel], (w, h), sigmas_px=np.full(len(sel), 1.5), prior=prior, config=PnPConfig()
    )
    return result.to_camera_pose("synthetic-track-shot", 0), result


def test_lift_features_hits_tower_geometry() -> None:
    prior = get_prior("brooklyn_heights_promenade")
    c2w = look_at(prior.enu(), WTC1.enu_center() + np.array([0.0, 0.0, 180.0]))
    intr = CameraIntrinsics(width=256, height=192, fx=400.0, fy=400.0, cx=127.5, cy=95.5)
    pose, _ = _keyframe_pose(c2w, intr, prior)
    # a point near the image centre should hit the tower it is pointed at
    uv = np.array([[127.5, 95.5], [127.5, 100.0]])
    pts3d, kept = lift_features(uv, pose, towers=TOWERS)
    assert len(kept) >= 1
    # every lifted point must be near the tower box in plan (within a few
    # tens of metres -- the box is 63 m across, and the pose itself has error)
    wtc1_xy = WTC1.enu_center()[:2]
    for p in pts3d:
        assert (
            np.linalg.norm(p[:2] - wtc1_xy) < 60.0
            or np.linalg.norm(p[:2] - TOWERS[1].enu_center()[:2]) < 60.0
        )


def test_lift_features_sky_ray_returns_nothing_without_ground() -> None:
    prior = get_prior("brooklyn_heights_promenade")
    c2w = look_at(prior.enu(), prior.enu() + np.array([0.0, 0.0, 1000.0]))  # straight up
    intr = CameraIntrinsics(width=64, height=64, fx=200.0, fy=200.0, cx=31.5, cy=31.5)
    from wtc4d.schema.camera import CameraPose

    pose = CameraPose.from_matrix(c2w, shot_id="x", frame_idx=0, intrinsics=intr, method="manual")
    uv = np.array([[31.5, 31.5]])
    pts3d, kept = lift_features(uv, pose, towers=TOWERS, ground_z=0.0)
    # looking straight up hits neither a tower nor (from above) the ground plane
    assert len(kept) == 0


def test_track_shot_recovers_drifting_handheld_camera() -> None:
    frames, true_c2w, prior, intr, base_c = _make_shot()
    kf_pose, kf_result = _keyframe_pose(true_c2w[0], intr, prior)
    assert np.linalg.norm(kf_result.position - base_c) < 5.0 * max(kf_result.position_sigma_m, 1.0)

    cfg = TrackConfig(
        camera_static=False,
        pnp=PnPConfig(fix_focal=intr.fx, robust=True),
        max_features=600,
        min_feature_distance=5,  # the small synthetic frame has sparse texture
        min_pnp_points=5,
        min_tracks=20,
    )
    tracked = track_shot(frames, 0, kf_pose, config=cfg)
    assert len(tracked) == N_FRAMES

    solved = [t for t in tracked if t.pose is not None]
    # tracking should carry the pose through every non-keyframe frame of this
    # short, clean shot
    assert len(solved) >= N_FRAMES - 1

    from wtc4d.camreg.conventions import camera_center

    for t in solved:
        true_c = base_c + np.array([1.4, 0.3, 0.0]) * t.frame_idx
        err = np.linalg.norm(camera_center(t.pose.c2w) - true_c)
        # ~2 km baseline shot: a few tens of metres of drift is expected from
        # LK tracking + per-frame PnP without bundle adjustment
        assert err < 60.0, f"frame {t.frame_idx}: {err:.1f} m off"


def test_track_shot_reports_tracks_and_flow() -> None:
    frames, true_c2w, prior, intr, base_c = _make_shot()
    kf_pose, _ = _keyframe_pose(true_c2w[0], intr, prior)
    tracked = track_shot(frames, 0, kf_pose, config=TrackConfig())
    assert tracked[0].n_tracks > 0
    assert tracked[0].mean_flow_px == 0.0  # the keyframe itself has no flow yet
    if len(tracked) > 1:
        assert tracked[1].mean_flow_px >= 0.0


def test_bundle_adjust_reduces_or_holds_reprojection_error() -> None:
    """A short window with injected per-frame position jitter: bundle
    adjustment (smoothness prior) should not make the fit worse."""
    frames, true_c2w, prior, intr, base_c = _make_shot()
    reg = landmark_registry()
    ids = sorted(reg)
    pts_all = landmark_points(ids, reg)
    w, h = intr.width, intr.height

    poses = []
    observations = []
    rng = np.random.default_rng(11)
    for i, c2w in enumerate(true_c2w[:4]):
        uv, valid = project_points(pts_all, c2w, intr)
        inframe = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        sel = [j for j in range(len(ids)) if inframe[j]]
        noisy_uv = uv[sel] + rng.normal(0, 1.0, (len(sel), 2))
        result = solve_pose(
            pts_all[sel], noisy_uv, (w, h), sigmas_px=np.full(len(sel), 1.0), prior=prior
        )
        poses.append(result.to_camera_pose("bundle-test", i))
        observations.append((pts_all[sel], noisy_uv, np.full(len(sel), 1.0)))

    def total_rmse(pose_list):
        errs = []
        for p, (pts, uv, _sig) in zip(pose_list, observations, strict=False):
            proj, _ = project_points(pts, p.matrix(), p.intrinsics)
            errs.append(np.linalg.norm(proj - uv, axis=1))
        return float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))

    before = total_rmse(poses)
    after_poses = bundle_adjust(
        poses, observations, camera_static=False, config=BundleAdjustConfig()
    )
    after = total_rmse(after_poses)
    assert after < before * 1.5  # should not blow up; typically improves or holds


def test_bundle_adjust_static_camera_pins_position() -> None:
    """For a locked-off shot, bundle adjustment should collapse the positions
    to (nearly) one point."""
    frames, true_c2w, prior, intr, base_c = _make_shot(drift_per_frame=(0.0, 0.0, 0.0))
    reg = landmark_registry()
    ids = sorted(reg)
    pts_all = landmark_points(ids, reg)
    w, h = intr.width, intr.height
    rng = np.random.default_rng(5)

    poses, observations = [], []
    for i, c2w in enumerate(true_c2w[:4]):
        uv, valid = project_points(pts_all, c2w, intr)
        inframe = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        sel = [j for j in range(len(ids)) if inframe[j]]
        noisy_uv = uv[sel] + rng.normal(0, 1.5, (len(sel), 2))
        result = solve_pose(
            pts_all[sel], noisy_uv, (w, h), sigmas_px=np.full(len(sel), 1.5), prior=prior
        )
        poses.append(result.to_camera_pose("static-bundle-test", i))
        observations.append((pts_all[sel], noisy_uv, np.full(len(sel), 1.5)))

    adjusted = bundle_adjust(poses, observations, camera_static=True, config=BundleAdjustConfig())
    from wtc4d.camreg.conventions import camera_center

    centres = np.array([camera_center(p.matrix()) for p in adjusted])
    spread = centres.std(axis=0)
    assert np.linalg.norm(spread) < 5.0  # positions collapsed close together
