"""Synthetic end-to-end validation: known cameras -> degraded renders -> recovered poses.

This is the machine-checkable evidence behind the accuracy claims in
``wtc4d/camreg/README.md``.  ``tests/test_camreg_pnp.py`` asserts tolerances on
a subset of this; :func:`run_synthetic_validation` (also wired to
``wtc4d camreg synthetic-report``) produces the full table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wtc4d.camreg.conventions import look_at, look_from_pose, project_points
from wtc4d.camreg.landmarks import landmark_points, landmark_registry
from wtc4d.camreg.pnp import PnPConfig, solve_pose
from wtc4d.camreg.priors import get_prior
from wtc4d.camreg.synthetic import default_scene, degrade
from wtc4d.schema.camera import CameraIntrinsics
from wtc4d.world import WTC1

__all__ = ["ScenarioSpec", "STANDARD_SCENARIOS", "run_synthetic_validation"]


@dataclass
class ScenarioSpec:
    prior_id: str
    offset_m: tuple[float, float, float]
    hfov_deg: float
    roll_deg: float = 0.0
    image_size: tuple[int, int] = (320, 240)
    click_sigma_px: float = 1.5
    seed: int = 0


STANDARD_SCENARIOS: list[ScenarioSpec] = [
    ScenarioSpec("brooklyn_heights_promenade", (60.0, -20.0, 2.0), 35.0, roll_deg=2.0),
    ScenarioSpec("jc_exchange_place", (-30.0, 40.0, 1.0), 28.0, roll_deg=-1.0, seed=1),
    ScenarioSpec("liberty_state_park", (100.0, 50.0, 1.0), 20.0, seed=2),
    ScenarioSpec("hoboken_waterfront", (0.0, 0.0, 1.0), 15.0, seed=3),
    ScenarioSpec("west_street_vesey", (0.0, 0.0, 0.0), 70.0, seed=4),
]


def run_scenario(spec: ScenarioSpec) -> dict:
    """Render one scenario, add SD-style degradation, click the true landmark
    pixels (simulating a careful annotator) and solve. Returns a result row.
    """
    scene = default_scene()
    prior = get_prior(spec.prior_id)
    true_c = prior.enu() + np.array(spec.offset_m)
    target = WTC1.enu_center() + np.array([0.0, 0.0, 180.0])
    c2w = look_at(true_c, target, roll_deg=spec.roll_deg)
    w, h = spec.image_size
    f = w / (2.0 * np.tan(np.radians(spec.hfov_deg) / 2.0))
    intr = CameraIntrinsics(width=w, height=h, fx=f, fy=f, cx=(w - 1) / 2.0, cy=(h - 1) / 2.0)

    # The degraded render is what an investigator would actually see and click
    # in wtc4d.camreg.annotate; it plays no further role here because this
    # scenario clicks the *known* projected pixel (plus noise) rather than
    # detecting it in the image, but rendering it is still worth doing: it
    # exercises the same synthetic pipeline real annotation would see and
    # keeps this validation honest about what "SD footage" looks like.
    render = scene.render(c2w, intr)
    degrade(render.image, scale=0.7, noise_sigma=0.02, blur_sigma_px=1.0, seed=spec.seed)

    registry = landmark_registry()
    ids = sorted(registry)
    pts = landmark_points(ids, registry)
    uv_true, valid = project_points(pts, c2w, intr)
    inframe = (
        valid
        & (uv_true[:, 0] >= 0)
        & (uv_true[:, 0] < w)
        & (uv_true[:, 1] >= 0)
        & (uv_true[:, 1] < h)
    )
    sel = [i for i in range(len(ids)) if inframe[i]]

    rng = np.random.default_rng(spec.seed)
    uv_clicked = uv_true[sel] + rng.normal(0.0, spec.click_sigma_px, (len(sel), 2))

    result = solve_pose(
        pts[sel],
        uv_clicked,
        (w, h),
        sigmas_px=np.full(len(sel), spec.click_sigma_px),
        prior=prior,
        landmark_ids=[ids[i] for i in sel],
        config=PnPConfig(),
    )
    az_t, el_t, roll_t = look_from_pose(c2w)
    az_e, el_e, roll_e = look_from_pose(result.c2w)
    rot_err = float(
        np.sqrt(
            ((az_e - az_t + 180) % 360 - 180) ** 2 + (el_e - el_t) ** 2 + (roll_e - roll_t) ** 2
        )
    )
    return {
        "prior": spec.prior_id,
        "hfov": spec.hfov_deg,
        "n_pts": len(sel),
        "pos_err_m": float(np.linalg.norm(result.position - true_c)),
        "sigma_m": result.position_sigma_m,
        "rot_err_deg": rot_err,
        "rmse_px": result.reproj_rmse_px,
        "planarity": result.planarity,
        "focal_depth_corr": result.focal_depth_correlation,
        "warnings": result.warnings,
    }


def run_synthetic_validation(scenarios: list[ScenarioSpec] | None = None) -> list[dict]:
    return [run_scenario(s) for s in (scenarios or STANDARD_SCENARIOS)]
