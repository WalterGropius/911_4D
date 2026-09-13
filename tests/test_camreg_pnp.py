"""Synthetic PnP recovery tests: known camera -> degraded render -> click ->
solve -> check the recovered pose against ground truth and against the
reported uncertainty."""

from __future__ import annotations

import numpy as np
import pytest

from wtc4d.camreg.annotations import FrameAnnotation, LandmarkObservation
from wtc4d.camreg.conventions import look_at
from wtc4d.camreg.landmarks import landmark_registry, spread_metrics
from wtc4d.camreg.pnp import PnPConfig, solve_annotation, solve_pose
from wtc4d.camreg.priors import get_prior, load_priors
from wtc4d.camreg.validate import STANDARD_SCENARIOS, run_scenario
from wtc4d.world import WTC1

RNG = np.random.default_rng(7)


def test_priors_load_and_are_unique() -> None:
    priors = load_priors()
    assert len(priors) >= 20
    ids = [p.id for p in priors]
    assert len(ids) == len(set(ids))
    for p in priors:
        assert -90 <= p.location.lat <= 90
        assert -180 <= p.location.lon <= 180
        assert p.position_sigma_m > 0


def test_landmark_registry_has_both_towers_and_spread() -> None:
    reg = landmark_registry()
    assert any(k.startswith("wtc1_") for k in reg)
    assert any(k.startswith("wtc2_") for k in reg)
    assert "brooklyn_bridge_manhattan_tower" in reg
    # excludes landmarks that did not exist on the day
    assert "goldman_30_hudson_jc" not in reg
    full = landmark_registry(include_nonexistent=True)
    assert "goldman_30_hudson_jc" in full


def test_spread_metrics_planar_vs_volumetric() -> None:
    planar = np.column_stack([RNG.uniform(-100, 100, 50), RNG.uniform(-100, 100, 50), np.zeros(50)])
    m = spread_metrics(planar)
    assert m["planarity"] < 1e-6
    volumetric = RNG.uniform(-100, 100, (50, 3))
    m2 = spread_metrics(volumetric)
    assert m2["planarity"] > 0.3


@pytest.mark.parametrize("scenario", STANDARD_SCENARIOS, ids=lambda s: s.prior_id)
def test_synthetic_pnp_recovers_pose(scenario) -> None:
    """The headline validation: render, degrade, click, solve, check against
    both ground truth and the pose's own reported uncertainty."""
    row = run_scenario(scenario)
    assert row["n_pts"] >= 3
    # position within a generous multiple of the reported 1-sigma: this is an
    # honesty check on the covariance, not just an accuracy check
    assert row["pos_err_m"] < 5.0 * max(row["sigma_m"], 1.0)
    assert row["rot_err_deg"] < 1.5
    assert row["rmse_px"] < 6.0


def test_synthetic_pnp_tight_lens_is_accurate() -> None:
    """A near, wide-FOV, well-conditioned shot should recover position to a
    few metres -- this is the case with the least focal/depth ambiguity."""
    row = run_scenario(STANDARD_SCENARIOS[-1])  # west_street_vesey, hfov=70
    assert row["pos_err_m"] < 10.0
    assert row["sigma_m"] < 20.0


def test_solve_pose_rejects_too_few_points() -> None:
    pts = np.zeros((2, 3))
    uv = np.zeros((2, 2))
    with pytest.raises(ValueError):
        solve_pose(pts, uv, (640, 480))


def test_solve_pose_three_points_needs_prior() -> None:
    reg = landmark_registry()
    ids = ["wtc1_antenna", "wtc2_roof_ne", "brooklyn_bridge_manhattan_tower"]
    pts = np.array([reg[i].enu() for i in ids])
    with pytest.raises(ValueError):
        solve_pose(pts, np.zeros((3, 2)), (640, 480))


def test_solve_pose_three_points_with_prior_recovers_pose() -> None:
    """The minimal 3-point case, disambiguated by the vantage-point prior."""
    from wtc4d.camreg.conventions import project_points
    from wtc4d.schema.camera import CameraIntrinsics

    prior = get_prior("jc_exchange_place")
    true_c = prior.enu() + np.array([10.0, 5.0, 0.0])
    target = WTC1.enu_center() + np.array([0.0, 0.0, 200.0])
    c2w = look_at(true_c, target)
    w, h = 640, 480
    f = w / (2 * np.tan(np.radians(30) / 2))
    intr = CameraIntrinsics(width=w, height=h, fx=f, fy=f, cx=(w - 1) / 2, cy=(h - 1) / 2)

    reg = landmark_registry()
    ids = ["wtc1_antenna", "wtc1_roof_ne", "wtc2_roof_ne"]
    pts = np.array([reg[i].enu() for i in ids])
    uv, valid = project_points(pts, c2w, intr)
    assert valid.all()

    result = solve_pose(pts, uv, (w, h), sigmas_px=np.full(3, 1.0), prior=prior, config=PnPConfig())
    assert np.linalg.norm(result.position - true_c) < 4.0 * result.position_sigma_m


def test_fix_position_holds_camera_at_prior() -> None:
    from wtc4d.camreg.conventions import camera_center, project_points
    from wtc4d.schema.camera import CameraIntrinsics

    prior = get_prior("empire_state_observatory")
    c2w = look_at(prior.enu(), WTC1.enu_center() + np.array([0.0, 0.0, 200.0]))
    w, h = 640, 480
    f = w / (2 * np.tan(np.radians(15) / 2))
    intr = CameraIntrinsics(width=w, height=h, fx=f, fy=f, cx=(w - 1) / 2, cy=(h - 1) / 2)
    reg = landmark_registry()
    ids = sorted(reg)
    pts = np.array([reg[i].enu() for i in ids])
    uv, valid = project_points(pts, c2w, intr)
    inframe = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    sel = [i for i in range(len(ids)) if inframe[i]]
    assert len(sel) >= 3

    result = solve_pose(
        pts[sel],
        uv[sel],
        (w, h),
        sigmas_px=np.full(len(sel), 1.5),
        prior=prior,
        config=PnPConfig(fix_position=True),
    )
    np.testing.assert_allclose(camera_center(result.c2w), prior.enu(), atol=1e-6)


def test_solve_annotation_uses_named_prior() -> None:
    from wtc4d.camreg.conventions import project_points
    from wtc4d.schema.camera import CameraIntrinsics

    prior = get_prior("brooklyn_heights_promenade")
    c2w = look_at(
        prior.enu() + np.array([20.0, 5.0, 0.0]), WTC1.enu_center() + np.array([0, 0, 180.0])
    )
    w, h = 480, 320
    f = w / (2 * np.tan(np.radians(30) / 2))
    intr = CameraIntrinsics(width=w, height=h, fx=f, fy=f, cx=(w - 1) / 2, cy=(h - 1) / 2)
    reg = landmark_registry()
    ids = sorted(reg)
    pts = np.array([reg[i].enu() for i in ids])
    uv, valid = project_points(pts, c2w, intr)
    inframe = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    sel = [i for i in range(len(ids)) if inframe[i]][:8]

    ann = FrameAnnotation(
        shot_id="test-shot",
        frame_idx=0,
        image_width=w,
        image_height=h,
        camera_prior_id="brooklyn_heights_promenade",
        observations=[
            LandmarkObservation(
                landmark_id=ids[i], u=float(uv[i, 0]), v=float(uv[i, 1]), sigma_px=1.5
            )
            for i in sel
        ],
    )
    assert ann.check() == []
    result = solve_annotation(ann)
    pose = result.to_camera_pose(ann.shot_id, ann.frame_idx)
    assert pose.shot_id == "test-shot"
    assert pose.n_landmarks == len(sel)
    assert pose.reproj_rmse_px < 2.0


def test_annotation_check_flags_out_of_bounds() -> None:
    ann = FrameAnnotation(
        shot_id="x",
        frame_idx=0,
        image_width=100,
        image_height=100,
        observations=[
            LandmarkObservation(landmark_id="wtc1_roof_ne", u=500, v=50),
            LandmarkObservation(landmark_id="wtc2_roof_ne", u=50, v=50),
            LandmarkObservation(landmark_id="wtc1_antenna", u=60, v=60),
        ],
    )
    problems = ann.check()
    assert any("outside image width" in p for p in problems)


def test_annotation_rejects_unknown_landmark() -> None:
    ann = FrameAnnotation(
        shot_id="x",
        frame_idx=0,
        image_width=100,
        image_height=100,
        observations=[
            LandmarkObservation(landmark_id="not_a_real_landmark", u=10, v=10),
            LandmarkObservation(landmark_id="wtc1_roof_ne", u=20, v=20),
            LandmarkObservation(landmark_id="wtc2_roof_ne", u=30, v=30),
        ],
    )
    problems = ann.check()
    assert any("unknown landmark" in p for p in problems)
