"""Pose + unknown focal length from 2D-3D landmark correspondences.

Problem
-------
A 2001 news frame gives us a handful of clicked points whose 3D positions we
know (tower corners, spires, bridge towers).  We want the camera-to-world pose
and the focal length, with an honest covariance.  Everything else about the
intrinsics is fixed by assumption:

* principal point at the image centre, ``(W-1)/2, (H-1)/2`` (OpenCV pixel
  convention: integer coordinates are pixel *centres*);
* square pixels, ``fx == fy`` -- true for a digitised 4:3 SD frame only if the
  frame has been resampled to square pixels.  A 720x480 NTSC DV frame has
  **non-square** pixels (~0.9 PAR); either resample before annotating or set
  ``PnPConfig.pixel_aspect`` so ``fy = fx / pixel_aspect``;
* distortion zero, or a single radial term ``k1`` when
  ``PnPConfig.estimate_k1`` is set and there are enough points to afford it.

Method
------
1. **Focal grid.**  Focal length and distance-along-the-optical-axis are
   strongly coupled for a distant subject, so we do not try to solve for the
   focal from a cold start.  We sweep a geometric grid of horizontal fields of
   view and, at each one, get an initial pose from ``cv2.solvePnPRansac``
   (``SOLVEPNP_EPNP``, or ``solveP3P`` for exactly three points).  When OpenCV
   is unavailable, or gives nothing, the camera prior supplies a look-at
   initialisation instead.
2. **Refinement.**  The best few initialisations are refined with
   ``scipy.optimize.least_squares`` (Levenberg-Marquardt) over
   ``[rvec, tvec, log f, (k1)]``, minimising reprojection error weighted by
   each observation's ``sigma_px``, plus a **soft prior** on the camera centre
   weighted by the prior's ``position_sigma_m`` / ``alt_sigma_m``.  The soft
   prior is what makes 3- and 4-point cases solvable at all, and what keeps
   the badly conditioned depth direction from running away.
3. **Covariance.**  From the Jacobian at the solution,
   ``cov = (J^T J)^-1 * max(chi2_red, 1)``.  The chi-square rescaling means an
   over-optimistic set of ``sigma_px`` cannot produce an over-confident pose;
   it can still produce an over-*confident* one if the annotations are biased
   rather than noisy, which no amount of arithmetic can detect.

What this cannot do
-------------------
If the landmarks are nearly coplanar or nearly collinear in 3D -- the normal
case for a far-away skyline shot, where every visible landmark is roughly the
same distance away -- then the range to the subject and the focal length trade
off almost exactly.  The solver reports this as ``planarity`` and
``focal_depth_correlation``; when the correlation is above ~0.95 the position
along the viewing direction is *the prior's*, not the image's, and
``position_sigma_m`` will (correctly) be large.  The fix is not a better
solver: it is a landmark at a different depth (a bridge tower, a near rooftop,
a street corner) or an independently known focal length.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from wtc4d.camreg.annotations import FrameAnnotation
from wtc4d.camreg.conventions import (
    c2w_from_rt,
    camera_center,
    distort,
    look_at,
    rodrigues,
    rt_from_c2w,
)
from wtc4d.camreg.landmarks import spread_metrics
from wtc4d.camreg.priors import CameraPriorRecord
from wtc4d.schema.camera import CameraIntrinsics, CameraPose

__all__ = ["PnPConfig", "PnPResult", "solve_pose", "solve_annotation"]


@dataclass
class PnPConfig:
    """Knobs for :func:`solve_pose`.  Defaults suit SD footage of the skyline."""

    fov_min_deg: float = 6.0
    """Narrowest horizontal field of view to consider (a long broadcast lens)."""
    fov_max_deg: float = 110.0
    """Widest horizontal field of view to consider (a handheld camcorder at
    full wide, or a consumer wide-angle adapter)."""
    n_focal: int = 28
    """Number of focal hypotheses, spaced geometrically in focal length."""
    n_refine: int = 6
    """How many of the best initialisations are refined."""
    ransac_reproj_px: float = 8.0
    """RANSAC inlier threshold for the OpenCV initialisation (pixels)."""
    ransac_iters: int = 2000
    robust: bool = True
    """Run a soft-L1 pass first and drop observations whose residual exceeds
    ``outlier_px`` before the final Gaussian refinement."""
    outlier_px: float = 12.0
    estimate_k1: bool = False
    """Solve one radial distortion coefficient.  Needs >= 6 well spread points;
    ignored otherwise."""
    fix_position: bool = False
    """Locked-off camera: hold the centre at the prior's location and solve
    only rotation (and focal).  Use for tripod/rooftop shots once the position
    has been established from a better-conditioned frame of the same shot."""
    fix_focal: float | None = None
    """Known focal length in pixels; skips the focal grid."""
    pixel_aspect: float = 1.0
    """``fx / fy``.  1.0 for square pixels."""
    prior_weight: float = 1.0
    """Multiplier on the soft prior residual.  0 disables the prior entirely
    (only sensible with many, well spread landmarks)."""
    max_nfev: int = 200


@dataclass
class PnPResult:
    """A solved pose with its uncertainty and the diagnostics behind it."""

    c2w: np.ndarray
    intrinsics: CameraIntrinsics
    n_landmarks: int
    n_inliers: int
    inlier_ids: list[str]
    outlier_ids: list[str]
    reproj_rmse_px: float
    reproj_max_px: float
    position_sigma_m: float
    rotation_sigma_deg: float
    focal_sigma_px: float
    covariance: np.ndarray
    """Parameter covariance in the order ``[rvec(3), tvec(3), log_f, (k1)]``,
    with rows/columns for held-fixed parameters omitted."""
    param_names: list[str]
    planarity: float
    """Smallest / largest PCA sigma of the 3D landmark set (see
    :func:`wtc4d.camreg.landmarks.spread_metrics`)."""
    focal_depth_correlation: float
    """|corr(log f, range along the optical axis)|.  Above ~0.95 the range is
    supplied by the prior, not by the image."""
    prior_offset_m: float | None
    converged: bool
    message: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def position(self) -> np.ndarray:
        return camera_center(self.c2w)

    @property
    def hfov_deg(self) -> float:
        return self.intrinsics.hfov_deg

    def to_camera_pose(
        self, shot_id: str, frame_idx: int, method: str = "landmark_pnp", notes: str = ""
    ) -> CameraPose:
        """Convert to the shared :class:`~wtc4d.schema.camera.CameraPose`."""
        extra = "; ".join(self.warnings)
        return CameraPose.from_matrix(
            self.c2w,
            shot_id=shot_id,
            frame_idx=frame_idx,
            intrinsics=self.intrinsics,
            method=method,
            position_sigma_m=float(self.position_sigma_m),
            rotation_sigma_deg=float(self.rotation_sigma_deg),
            reproj_rmse_px=float(self.reproj_rmse_px),
            n_landmarks=int(self.n_inliers),
            notes=(notes + ("; " if notes and extra else "") + extra),
        )

    def summary(self) -> str:
        p = self.position
        return (
            f"pos=({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) m  +-{self.position_sigma_m:.1f} m | "
            f"rot +-{self.rotation_sigma_deg:.2f} deg | hfov={self.hfov_deg:.1f} deg "
            f"(f={self.intrinsics.fx:.1f}+-{self.focal_sigma_px:.1f} px) | "
            f"rmse={self.reproj_rmse_px:.2f} px over {self.n_inliers}/{self.n_landmarks} points"
        )


# --- parameter packing --------------------------------------------------------
@dataclass
class _Problem:
    pts: np.ndarray  # (N, 3) world points
    uv: np.ndarray  # (N, 2) observations
    sig: np.ndarray  # (N,) pixel sigmas
    cx: float
    cy: float
    pixel_aspect: float
    estimate_k1: bool
    fix_position: bool
    fix_focal: float | None
    fixed_center: np.ndarray | None
    prior_center: np.ndarray | None
    prior_sigma: np.ndarray | None  # (3,) metres
    loss: str = "linear"

    @property
    def param_names(self) -> list[str]:
        names = ["rx", "ry", "rz"]
        if not self.fix_position:
            names += ["tx", "ty", "tz"]
        if self.fix_focal is None:
            names += ["log_f"]
        if self.estimate_k1:
            names += ["k1"]
        return names

    def pack(self, rvec, tvec, f: float, k1: float = 0.0) -> np.ndarray:
        x = list(np.asarray(rvec, dtype=np.float64).reshape(3))
        if not self.fix_position:
            x += list(np.asarray(tvec, dtype=np.float64).reshape(3))
        if self.fix_focal is None:
            x += [math.log(max(float(f), 1e-6))]
        if self.estimate_k1:
            x += [float(k1)]
        return np.array(x, dtype=np.float64)

    def unpack(self, x) -> tuple[np.ndarray, np.ndarray, float, float]:
        x = np.asarray(x, dtype=np.float64)
        i = 3
        rvec = x[:3]
        if self.fix_position:
            R = rodrigues(rvec)  # noqa: N806
            tvec = -R @ self.fixed_center
        else:
            tvec = x[i : i + 3]
            i += 3
        if self.fix_focal is None:
            f = math.exp(float(x[i]))
            i += 1
        else:
            f = float(self.fix_focal)
        k1 = float(x[i]) if self.estimate_k1 else 0.0
        return rvec, tvec, f, k1

    def project(self, x) -> np.ndarray:
        rvec, tvec, f, k1 = self.unpack(x)
        R = rodrigues(rvec)  # noqa: N806
        cam = self.pts @ R.T + tvec
        z = np.where(cam[:, 2] > 1e-6, cam[:, 2], 1e-6)
        xn = np.column_stack([cam[:, 0] / z, cam[:, 1] / z])
        if k1:
            xn = distort(xn, [k1])
        u = f * xn[:, 0] + self.cx
        v = (f / self.pixel_aspect) * xn[:, 1] + self.cy
        behind = cam[:, 2] <= 1e-6
        if behind.any():
            # push points behind the camera far away so the solver is repelled
            u = np.where(behind, self.cx + 1e4, u)
            v = np.where(behind, self.cy + 1e4, v)
        return np.column_stack([u, v])

    def residuals(self, x) -> np.ndarray:
        proj = self.project(x)
        r = ((proj - self.uv) / self.sig[:, None]).reshape(-1)
        if self.prior_center is not None and self.prior_sigma is not None:
            rvec, tvec, _, _ = self.unpack(x)
            c = -rodrigues(rvec).T @ tvec
            r = np.concatenate([r, (c - self.prior_center) / self.prior_sigma])
        return r

    def center(self, x) -> np.ndarray:
        rvec, tvec, _, _ = self.unpack(x)
        return -rodrigues(rvec).T @ tvec


# --- initialisation -----------------------------------------------------------
def _focal_grid(cfg: PnPConfig, width: int) -> list[float]:
    if cfg.fix_focal is not None:
        return [float(cfg.fix_focal)]
    f_min = width / (2.0 * math.tan(math.radians(cfg.fov_max_deg) / 2.0))
    f_max = width / (2.0 * math.tan(math.radians(cfg.fov_min_deg) / 2.0))
    return list(np.geomspace(f_min, f_max, max(2, cfg.n_focal)))


def _cv_init(prob: _Problem, f: float, cfg: PnPConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pose hypotheses from OpenCV for one focal length.  ``[]`` if unavailable."""
    try:
        import cv2
    except Exception:  # noqa: BLE001
        return []
    k = np.array(
        [[f, 0.0, prob.cx], [0.0, f / prob.pixel_aspect, prob.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    obj = prob.pts.astype(np.float64).reshape(-1, 1, 3)
    img = prob.uv.astype(np.float64).reshape(-1, 1, 2)
    zero = np.zeros(5)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    n = len(prob.pts)
    try:
        if n == 3:
            ok, rvecs, tvecs = cv2.solveP3P(obj, img, k, zero, flags=cv2.SOLVEPNP_AP3P)
            if ok:
                out += [
                    (np.asarray(r).reshape(3), np.asarray(t).reshape(3))
                    for r, t in zip(rvecs, tvecs, strict=False)
                ]
        else:
            flag = cv2.SOLVEPNP_EPNP if n >= 6 else cv2.SOLVEPNP_ITERATIVE
            ok, rvec, tvec, _ = cv2.solvePnPRansac(
                obj,
                img,
                k,
                zero,
                flags=flag,
                reprojectionError=cfg.ransac_reproj_px,
                iterationsCount=cfg.ransac_iters,
                confidence=0.999,
            )
            if ok:
                out.append((np.asarray(rvec).reshape(3), np.asarray(tvec).reshape(3)))
            ok2, rvec2, tvec2 = cv2.solvePnP(obj, img, k, zero, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok2:
                out.append((np.asarray(rvec2).reshape(3), np.asarray(tvec2).reshape(3)))
    except cv2.error:
        return out
    return [(r, t) for r, t in out if np.all(np.isfinite(r)) and np.all(np.isfinite(t))]


def _prior_init(prob: _Problem) -> list[tuple[np.ndarray, np.ndarray]]:
    """Look-at initialisation from the prior position towards the landmarks."""
    c = prob.fixed_center if prob.fixed_center is not None else prob.prior_center
    if c is None:
        return []
    target = prob.pts.mean(axis=0)
    out = []
    for roll in (0.0, -8.0, 8.0):
        c2w = look_at(c, target, roll_deg=roll)
        out.append(rt_from_c2w(c2w))
    return out


# --- refinement ---------------------------------------------------------------
def _refine(prob: _Problem, x0: np.ndarray, cfg: PnPConfig, loss: str = "linear"):
    from scipy.optimize import least_squares

    method = "lm" if loss == "linear" else "trf"
    kwargs = {}
    if method == "lm":
        kwargs["max_nfev"] = cfg.max_nfev * max(1, len(x0))
    else:
        kwargs["max_nfev"] = cfg.max_nfev
        kwargs["f_scale"] = 3.0
    return least_squares(
        prob.residuals, x0, method=method, loss=loss, xtol=1e-12, ftol=1e-12, **kwargs
    )


def _covariance(res, n_res: int, n_par: int) -> tuple[np.ndarray, float]:
    """``(J^T J)^-1`` scaled by the reduced chi-square (floored at 1)."""
    j = np.asarray(res.jac, dtype=np.float64)
    jtj = j.T @ j
    dof = max(n_res - n_par, 1)
    chi2_red = float(2.0 * res.cost / dof)
    scale = max(chi2_red, 1.0)
    try:
        cov = np.linalg.inv(jtj) * scale
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(jtj, rcond=1e-12) * scale
    if not np.all(np.isfinite(cov)):
        cov = np.full((n_par, n_par), np.nan)
    return cov, chi2_red


def _numeric_jacobian(fn, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    f0 = np.atleast_1d(fn(x))
    out = np.zeros((f0.size, x.size))
    for i in range(x.size):
        step = eps * max(1.0, abs(float(x[i])))
        xp = x.copy()
        xp[i] += step
        out[:, i] = (np.atleast_1d(fn(xp)) - f0) / step
    return out


# --- public API ---------------------------------------------------------------
def solve_pose(
    points_world,
    uv,
    image_size: tuple[int, int],
    sigmas_px=None,
    prior: CameraPriorRecord | None = None,
    config: PnPConfig | None = None,
    landmark_ids: list[str] | None = None,
) -> PnPResult:
    """Solve pose (and focal length) from 2D-3D correspondences.

    Parameters
    ----------
    points_world : (N, 3) world ENU metres.
    uv : (N, 2) pixel observations.
    image_size : ``(width, height)`` of the frame the pixels refer to.
    sigmas_px : (N,) per-observation 1-sigma click error; defaults to 2 px.
    prior : vantage-point prior used as a soft constraint on the camera centre
        (and, with ``config.fix_position``, as a hard one).
    landmark_ids : names for reporting inliers/outliers.

    Raises
    ------
    ValueError : fewer than three correspondences, or fewer than four without
        a prior (the problem is not determined).
    """
    cfg = config or PnPConfig()
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    obs = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if len(pts) != len(obs):
        raise ValueError(f"{len(pts)} world points but {len(obs)} observations")
    n = len(pts)
    if n < 3:
        raise ValueError(f"need at least 3 correspondences, got {n}")
    if n < 4 and prior is None:
        raise ValueError("3-point cases need a camera prior to disambiguate; pass prior=")
    sig = (
        np.full(n, 2.0)
        if sigmas_px is None
        else np.asarray(sigmas_px, dtype=np.float64).reshape(n).copy()
    )
    sig = np.clip(sig, 1e-3, None)
    ids = list(landmark_ids) if landmark_ids is not None else [f"p{i}" for i in range(n)]
    width, height = int(image_size[0]), int(image_size[1])

    warnings: list[str] = []
    spread = spread_metrics(pts)
    estimate_k1 = bool(cfg.estimate_k1 and n >= 6)
    if cfg.estimate_k1 and not estimate_k1:
        warnings.append("k1 not estimated: fewer than 6 correspondences")

    prior_center = prior.enu() if prior is not None else None
    prior_sigma = None
    if prior is not None and cfg.prior_weight > 0:
        s = np.array(
            [prior.position_sigma_m, prior.position_sigma_m, prior.alt_sigma_m], dtype=np.float64
        )
        prior_sigma = np.clip(s, 1e-3, None) / max(cfg.prior_weight, 1e-9)

    prob = _Problem(
        pts=pts,
        uv=obs,
        sig=sig,
        cx=(width - 1) / 2.0,
        cy=(height - 1) / 2.0,
        pixel_aspect=cfg.pixel_aspect,
        estimate_k1=estimate_k1,
        fix_position=cfg.fix_position,
        fix_focal=cfg.fix_focal,
        fixed_center=prior_center if cfg.fix_position else None,
        prior_center=None if cfg.fix_position else prior_center,
        prior_sigma=None if cfg.fix_position else prior_sigma,
    )
    if cfg.fix_position and prior_center is None:
        raise ValueError("fix_position=True requires a prior")

    # 1. candidate initialisations over the focal grid
    candidates: list[tuple[float, np.ndarray]] = []
    for f in _focal_grid(cfg, width):
        inits = _cv_init(prob, f, cfg) + _prior_init(prob)
        for rvec, tvec in inits:
            x0 = prob.pack(rvec, tvec, f)
            try:
                cost = float(np.sum(prob.residuals(x0) ** 2))
            except Exception:  # noqa: BLE001
                continue
            if math.isfinite(cost):
                candidates.append((cost, x0))
    if not candidates:
        raise ValueError("no usable initialisation (check the correspondences and the prior)")
    candidates.sort(key=lambda t: t[0])

    # 2. refine the best few, keep the best final cost
    best = None
    for _, x0 in candidates[: max(1, cfg.n_refine)]:
        try:
            res = _refine(prob, x0, cfg, loss="soft_l1" if cfg.robust else "linear")
        except Exception:  # noqa: BLE001 -- a bad init can make LM fail outright
            continue
        if best is None or res.cost < best.cost:
            best = res
    if best is None:
        raise ValueError("refinement failed from every initialisation")

    # 3. robust pass: drop gross outliers, then a clean Gaussian refinement
    inlier_mask = np.ones(n, dtype=bool)
    if cfg.robust:
        err = np.linalg.norm(prob.project(best.x) - obs, axis=1)
        inlier_mask = err <= max(cfg.outlier_px, 3.0 * float(np.median(err)) + 1e-9)
        if inlier_mask.sum() < max(3, min(n, 4)):
            inlier_mask = np.ones(n, dtype=bool)  # refuse to strip the problem bare
            warnings.append("outlier rejection disabled: too few surviving correspondences")
    prob_final = _Problem(
        pts=pts[inlier_mask],
        uv=obs[inlier_mask],
        sig=sig[inlier_mask],
        cx=prob.cx,
        cy=prob.cy,
        pixel_aspect=cfg.pixel_aspect,
        estimate_k1=estimate_k1 and int(inlier_mask.sum()) >= 6,
        fix_position=prob.fix_position,
        fix_focal=prob.fix_focal,
        fixed_center=prob.fixed_center,
        prior_center=prob.prior_center,
        prior_sigma=prob.prior_sigma,
    )
    x_start = best.x if prob_final.estimate_k1 == prob.estimate_k1 else best.x[:-1]
    final = _refine(prob_final, x_start, cfg, loss="linear")

    # 4. statistics
    proj_all = prob.project(
        np.concatenate([final.x, [0.0]])
        if (prob.estimate_k1 and not prob_final.estimate_k1)
        else final.x
    )
    err_all = np.linalg.norm(proj_all - obs, axis=1)
    err_in = err_all[inlier_mask]
    rmse = float(np.sqrt(np.mean(err_in**2))) if err_in.size else float("nan")

    n_par = len(final.x)
    cov, chi2_red = _covariance(final, final.fun.size, n_par)
    rvec, tvec, f, k1 = prob_final.unpack(final.x)
    c2w = c2w_from_rt(rvec, tvec)

    # camera-centre covariance by propagating through C(params)
    jc = _numeric_jacobian(prob_final.center, final.x)
    cov_c = jc @ cov @ jc.T
    position_sigma = float(np.sqrt(max(np.trace(cov_c), 0.0)))
    rotation_sigma = math.degrees(math.sqrt(max(float(np.trace(cov[:3, :3])), 0.0)))

    # focal sigma and the focal/range coupling
    focal_sigma = float("nan")
    focal_depth_corr = float("nan")
    names = prob_final.param_names
    if "log_f" in names:
        i_f = names.index("log_f")
        var_logf = float(cov[i_f, i_f])
        focal_sigma = f * math.sqrt(max(var_logf, 0.0))
        view_dir = c2w[:3, 2]
        j_depth = (view_dir @ jc).reshape(1, -1)
        var_depth = float(np.squeeze(j_depth @ cov @ j_depth.T))
        cov_fd = float(np.squeeze(j_depth @ cov[:, i_f]))
        denom = math.sqrt(max(var_logf * var_depth, 0.0))
        focal_depth_corr = abs(cov_fd / denom) if denom > 1e-30 else float("nan")

    intr = CameraIntrinsics(
        width=width,
        height=height,
        fx=float(f),
        fy=float(f / cfg.pixel_aspect),
        cx=prob.cx,
        cy=prob.cy,
        model="OPENCV" if k1 else "PINHOLE",
        dist=[float(k1), 0.0, 0.0, 0.0] if k1 else [],
    )

    prior_offset = (
        float(np.linalg.norm(camera_center(c2w) - prior_center))
        if prior_center is not None
        else None
    )
    if spread["planarity"] < 0.02:
        warnings.append(
            f"landmark set is nearly planar (planarity={spread['planarity']:.4f}): "
            "range along the optical axis is weakly observed"
        )
    if math.isfinite(focal_depth_corr) and focal_depth_corr > 0.95:
        warnings.append(
            f"focal/range correlation {focal_depth_corr:.3f}: the range is effectively "
            "the prior's, not the image's"
        )
    if chi2_red > 9.0:
        warnings.append(
            f"reduced chi-square {chi2_red:.1f}: residuals far exceed the stated sigma_px "
            "(wrong landmark, wrong frame, or optimistic click sigmas)"
        )
    if (
        prior_offset is not None
        and prior is not None
        and prior_offset > 4.0 * prior.position_sigma_m
    ):
        warnings.append(
            f"solution is {prior_offset:.0f} m from prior {prior.id} "
            f"({prior_offset / max(prior.position_sigma_m, 1e-6):.1f} sigma)"
        )
    if not final.success:
        warnings.append(f"optimiser did not converge cleanly: {final.message}")

    return PnPResult(
        c2w=c2w,
        intrinsics=intr,
        n_landmarks=n,
        n_inliers=int(inlier_mask.sum()),
        inlier_ids=[i for i, m in zip(ids, inlier_mask, strict=False) if m],
        outlier_ids=[i for i, m in zip(ids, inlier_mask, strict=False) if not m],
        reproj_rmse_px=rmse,
        reproj_max_px=float(err_in.max()) if err_in.size else float("nan"),
        position_sigma_m=position_sigma,
        rotation_sigma_deg=rotation_sigma,
        focal_sigma_px=focal_sigma,
        covariance=cov,
        param_names=names,
        planarity=float(spread["planarity"]),
        focal_depth_correlation=focal_depth_corr,
        prior_offset_m=prior_offset,
        converged=bool(final.success),
        message=str(final.message),
        warnings=warnings,
    )


def solve_annotation(
    ann: FrameAnnotation,
    prior: CameraPriorRecord | None = None,
    config: PnPConfig | None = None,
    registry: dict | None = None,
) -> PnPResult:
    """:func:`solve_pose` for a :class:`~wtc4d.camreg.annotations.FrameAnnotation`.

    When ``prior`` is omitted and the annotation names a ``camera_prior_id``,
    that prior is loaded from ``data/cameras/priors.yaml``.
    """
    if prior is None and ann.camera_prior_id:
        from wtc4d.camreg.priors import get_prior

        prior = get_prior(ann.camera_prior_id)
    return solve_pose(
        ann.world_points(registry),
        ann.uv(),
        (ann.image_width, ann.image_height),
        sigmas_px=ann.sigmas(),
        prior=prior,
        config=config,
        landmark_ids=ann.landmark_ids,
    )
