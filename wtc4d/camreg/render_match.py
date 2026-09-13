"""Annotation-free registration: render the prior, match it to the frame's edges.

The premise of this workstream is that classic SfM fails on 2001 footage --
smoke covers the subject, the cameras are far away with almost no baseline,
and the compression destroys texture -- but that we nonetheless *know what was
there*.  So instead of matching frames to each other, we match each frame to a
render of the georeferenced prior.

Three stages:

(a) **Coarse search.**  Sample poses on a grid around the camera prior
    (azimuth x elevation x roll x focal x a few position offsets), project the
    prior's wireframe, and score it against the frame's edges with an
    **oriented chamfer** distance: each projected model edge is compared only
    to image edges of a similar orientation, which stops a vertical tower edge
    from snapping onto a horizontal ticker bar.  Keep the top-k.

(b) **Continuous refinement.**  Polish the best hypotheses with Powell's method
    on a blurred chamfer cost, now with hidden-line removal (points whose
    rendered depth says they are behind something are dropped).

(c) **Correspondence + PnP.**  Walk the refined model edges, snap each sampled
    3D point to the nearest compatible image edge pixel, and feed the resulting
    2D-3D correspondences to :func:`wtc4d.camreg.pnp.solve_pose`.  This is what
    produces the final pose *with a covariance*; stages (a) and (b) only find
    the basin.

Why the skyline is the right feature
------------------------------------
The Lower Manhattan silhouette of 2001 is close to unique: two 415 m boxes
with a 110 m gap, the stepped World Financial Center roofs immediately west,
and nothing else within 200 m of their height for kilometres.  A rotation error
of one degree moves the towers by ~30 px in a 40-degree SD frame, so the
silhouette pins rotation hard.  It says much less about *range*, which is why
the prior and the PnP covariance matter (see :mod:`wtc4d.camreg.pnp`).

Honest failure modes
--------------------
* **The coarse grid is genuinely multi-modal and this is not fully solved.**
  Matching a sparse two-box silhouette (the fallback model, before
  ``wtc4d.geo`` lands) against a real chamfer field has many shallow local
  minima: a narrow, badly-cropped field of view can score as well as or better
  than the true wide one, because explaining 8% of the frame very precisely
  can beat explaining 35% of it approximately.  ``min_skyline_columns`` and
  the coverage-aware scoring in :func:`_score` push against this but do not
  eliminate it -- validated on synthetic data (see the package README), the
  coarse+refine stages alone recover the right basin only some of the time;
  treat their output as a *candidate*, not a pose.  What *is* reliable is
  :func:`edge_correspondences` + :func:`wtc4d.camreg.pnp.solve_pose` once given
  the right basin, and the annotation-based path
  (:mod:`wtc4d.camreg.annotations` + :mod:`wtc4d.camreg.pnp`) end to end, which
  is the recommended production path today.  The real building mesh from
  ``wtc4d.geo`` (a full skyline, not two boxes) should sharpen the chamfer
  field enough to close most of this gap; re-validate once it lands.
* **Smoke over the silhouette.**  After 09:03 the plume often covers one tower
  edge entirely.  Pass a dynamic mask (:mod:`wtc4d.camreg.masks`); the score
  then rests on whatever is left, and the reported chamfer will be worse.
* **Long lenses.**  Above ~2000 px focal the azimuth/position trade-off makes
  the coarse grid shallow; fix the position (``fix_position``) instead.
* **Night, extreme zoom, or close-in street views** where no skyline is
  visible: this module will return a confident-looking wrong answer.  Check
  ``RegistrationResult.chamfer_px`` / ``column_coverage`` and the PnP warnings
  before believing it.
* **A wrong prior.**  The search is local to the prior; if the shot is not from
  where the corpus says it is, the result is garbage with a good-looking score.
  Run against several priors and compare ``chamfer_px`` and ``column_coverage``.
* **Mitigations that help in practice:** run :func:`coarse_search` with a wider
  ``top_k`` and inspect all of them rather than trusting the single best; seed
  ``centre_azimuth_deg``/``centre_elevation_deg`` from a rough manual estimate;
  or skip straight to manual annotation (:mod:`wtc4d.camreg.annotate`) for a
  keyframe and let :mod:`wtc4d.camreg.track` propagate it through the shot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from wtc4d.camreg.conventions import (
    invert_rigid,
    pose_from_look,
)
from wtc4d.camreg.landmarks import landmark_registry
from wtc4d.camreg.pnp import PnPConfig, PnPResult, solve_pose
from wtc4d.camreg.priors import CameraPriorRecord
from wtc4d.schema.camera import CameraIntrinsics
from wtc4d.world import TOWERS, latlon_to_enu

__all__ = [
    "CoarseSearchConfig",
    "EdgeTarget",
    "PoseHypothesis",
    "RegistrationResult",
    "WireModel",
    "auto_register",
    "coarse_search",
    "edge_map",
    "match_learned",
    "prior_wireframe",
    "refine_hypothesis",
]


# --- the model ----------------------------------------------------------------
@dataclass
class WireModel:
    """3D line segments of the static prior, in world ENU metres."""

    segments: np.ndarray  # (S, 2, 3)
    weights: np.ndarray  # (S,) relative importance
    labels: list[str]
    landmark_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    landmark_ids: list[str] = field(default_factory=list)

    def sample(self, per_metre: float = 0.08, min_pts: int = 4, max_pts: int = 40):
        """Sample points along every segment.

        Returns ``(points (M, 3), seg_index (M,), weights (M,))``.  Sampling in
        *world* units rather than pixels means a 400 m tower edge always
        contributes more evidence than a 20 m parapet, whatever the zoom.
        """
        pts, idx, wts = [], [], []
        for i, seg in enumerate(self.segments):
            length = float(np.linalg.norm(seg[1] - seg[0]))
            n = int(np.clip(round(length * per_metre), min_pts, max_pts))
            t = np.linspace(0.0, 1.0, n)[:, None]
            pts.append(seg[0] + t * (seg[1] - seg[0]))
            idx.append(np.full(n, i, dtype=np.int64))
            wts.append(np.full(n, self.weights[i]))
        if not pts:
            return np.zeros((0, 3)), np.zeros(0, dtype=np.int64), np.zeros(0)
        return np.vstack(pts), np.concatenate(idx), np.concatenate(wts)


def _box_edges(corners: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """12 edges of a box given as (8, 3): 4 base corners then 4 roof corners.

    Vertical edges get a higher weight than horizontal ones: on a tower they
    are hundreds of metres of high-contrast silhouette, and they are the last
    thing smoke hides.
    """
    out = []
    for i in range(4):
        j = (i + 1) % 4
        out.append((corners[i], corners[j], 0.6))  # base ring
        out.append((corners[4 + i], corners[4 + j], 1.0))  # roof ring
        out.append((corners[i], corners[4 + i], 1.4))  # vertical
    return out


def prior_wireframe(use_geo: bool = True, include_landmark_boxes: bool = True) -> WireModel:
    """Build the wireframe used for matching.

    Uses ``wtc4d.geo`` when importable (it owns the real building footprints),
    and otherwise falls back to :data:`wtc4d.world.TOWERS` box corners plus a
    generic box under every landmark -- which is enough to test the machinery
    and enough for the towers themselves, which dominate every skyline frame.
    """
    segs: list[np.ndarray] = []
    weights: list[float] = []
    labels: list[str] = []

    geo_segments = _geo_segments() if use_geo else None
    if geo_segments is not None:
        for label, a, b, w in geo_segments:
            segs.append(np.array([a, b], dtype=np.float64))
            weights.append(w)
            labels.append(label)
    else:
        for tower in TOWERS:
            for a, b, w in _box_edges(tower.box_corners()):
                segs.append(np.array([a, b], dtype=np.float64))
                weights.append(w)
                labels.append(tower.id)
        if include_landmark_boxes:
            for lid, lm in landmark_registry().items():
                if lid.startswith(("wtc1_", "wtc2_")):
                    continue
                p = latlon_to_enu(lm.point)
                half = 12.0 if lm.kind in ("spire", "bridge_tower", "statue") else 35.0
                base = np.array(
                    [
                        [p[0] - half, p[1] - half, 0.0],
                        [p[0] + half, p[1] - half, 0.0],
                        [p[0] + half, p[1] + half, 0.0],
                        [p[0] - half, p[1] + half, 0.0],
                    ]
                )
                roof = base.copy()
                roof[:, 2] = max(p[2], 5.0)
                for a, b, w in _box_edges(np.vstack([base, roof])):
                    segs.append(np.array([a, b], dtype=np.float64))
                    weights.append(w * 0.5)
                    labels.append(lid)

    reg = landmark_registry()
    lids = sorted(reg)
    lpts = np.array([latlon_to_enu(reg[i].point) for i in lids]).reshape(-1, 3)
    return WireModel(
        segments=np.array(segs).reshape(-1, 2, 3),
        weights=np.array(weights, dtype=np.float64),
        labels=labels,
        landmark_points=lpts,
        landmark_ids=lids,
    )


def _geo_segments():
    """Wireframe edges from ``wtc4d.geo``, or ``None`` when unavailable.

    The geo workstream exposes ``load_scene() -> trimesh.Scene``; we take the
    axis-aligned bounding box of every mesh in it as a building block.  That is
    crude (it is a *silhouette* prior, not a facade prior), but it is a strict
    improvement over the two-tower fallback because it carries the real
    building footprints and heights of everything else in frame.
    """
    try:
        from wtc4d import geo  # type: ignore[attr-defined]

        scene = geo.load_scene()
    except Exception:  # noqa: BLE001 -- not merged yet, or failed to load
        return None
    out = []
    try:
        geometries = getattr(scene, "geometry", {}) or {}
        for name, mesh in geometries.items():
            bounds = np.asarray(mesh.bounds, dtype=np.float64)  # (2, 3) min/max
            lo, hi = bounds[0], bounds[1]
            base = np.array(
                [
                    [lo[0], lo[1], lo[2]],
                    [hi[0], lo[1], lo[2]],
                    [hi[0], hi[1], lo[2]],
                    [lo[0], hi[1], lo[2]],
                ]
            )
            roof = base.copy()
            roof[:, 2] = hi[2]
            for a, b, w in _box_edges(np.vstack([base, roof])):
                out.append((str(name), a, b, w))
    except Exception:  # noqa: BLE001
        return None
    return out or None


# --- the target ----------------------------------------------------------------
def edge_map(image: np.ndarray, low: float = 0.08, high: float = 0.20) -> np.ndarray:
    """Binary edge map of a grayscale float image in [0, 1].

    Uses ``cv2.Canny`` when OpenCV is importable (better thinning), otherwise a
    Sobel magnitude threshold with hysteresis-free double thresholding.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = np.clip(img, 0.0, 1.0)
    try:
        import cv2

        u8 = (img * 255).astype(np.uint8)
        return cv2.Canny(u8, int(low * 255), int(high * 255)) > 0
    except Exception:  # noqa: BLE001
        gx, gy = _sobel(img)
        mag = np.hypot(gx, gy)
        return mag > max(high, float(np.quantile(mag, 0.92)))


def _sobel(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kx = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
    ky = kx.T
    pad = np.pad(img, 1, mode="edge")
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    for i in range(3):
        for j in range(3):
            win = pad[i : i + img.shape[0], j : j + img.shape[1]]
            gx += kx[i, j] * win
            gy += ky[i, j] * win
    return gx, gy


@dataclass
class EdgeTarget:
    """Pre-computed (oriented) chamfer distance transforms of one frame."""

    width: int
    height: int
    edges: np.ndarray
    dt: np.ndarray
    dt_oriented: np.ndarray  # (n_bins, H, W)
    n_bins: int
    max_dist_px: float
    skyline: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """(W,) row of the topmost non-sky pixel per column, ``nan`` where unknown.
    This is the single most discriminative cue available -- see
    :func:`wtc4d.camreg.masks.skyline_profile`."""

    @classmethod
    def from_image(
        cls,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        n_bins: int = 4,
        max_dist_px: float = 40.0,
    ) -> EdgeTarget:
        """Build the target from a grayscale frame.

        ``mask`` is True where the pixel is *dynamic* (smoke, fire, dust,
        broadcast graphics) and must be ignored -- pass
        :func:`wtc4d.camreg.masks.dynamic_mask`.
        """
        from scipy.ndimage import distance_transform_edt

        img = np.asarray(image, dtype=np.float64)
        if img.ndim == 3:
            img = img.mean(axis=2)
        h, w = img.shape
        edges = edge_map(img)
        if mask is not None:
            edges = edges & ~np.asarray(mask, dtype=bool)
        gx, gy = _sobel(img)
        # edge orientation modulo pi, binned; the normal direction is what we
        # can measure, and it is perpendicular to the edge direction
        theta = (np.arctan2(gy, gx) + math.pi / 2.0) % math.pi
        bin_idx = np.minimum((theta / (math.pi / n_bins)).astype(int), n_bins - 1)
        dts = np.empty((n_bins, h, w), dtype=np.float32)
        for b in range(n_bins):
            sel = edges & (bin_idx == b)
            if not sel.any():
                dts[b] = max_dist_px
                continue
            dts[b] = np.minimum(distance_transform_edt(~sel), max_dist_px)
        dt = (
            np.minimum(distance_transform_edt(~edges), max_dist_px)
            if edges.any()
            else np.full((h, w), max_dist_px)
        )
        from wtc4d.camreg.masks import skyline_profile

        skyline = skyline_profile(img)
        if mask is not None:
            m = np.asarray(mask, dtype=bool)
            cols = np.arange(w)
            rows = np.where(np.isfinite(skyline), skyline, 0).astype(int)
            blocked = m[np.clip(rows, 0, h - 1), cols]
            skyline = np.where(blocked, np.nan, skyline)
        return cls(
            width=w,
            height=h,
            edges=edges,
            dt=dt.astype(np.float32),
            dt_oriented=dts,
            n_bins=n_bins,
            max_dist_px=max_dist_px,
            skyline=skyline,
        )

    def lookup(self, uv: np.ndarray, orientation: np.ndarray | None = None) -> np.ndarray:
        """Chamfer distance at pixel locations (nearest-neighbour sampling)."""
        u = np.clip(np.round(uv[:, 0]).astype(int), 0, self.width - 1)
        v = np.clip(np.round(uv[:, 1]).astype(int), 0, self.height - 1)
        if orientation is None:
            return self.dt[v, u]
        b = np.minimum(
            ((orientation % math.pi) / (math.pi / self.n_bins)).astype(int), self.n_bins - 1
        )
        return self.dt_oriented[b, v, u]


# --- hypotheses ----------------------------------------------------------------
@dataclass
class PoseHypothesis:
    c2w: np.ndarray
    focal_px: float
    score: float
    azimuth_deg: float
    elevation_deg: float
    roll_deg: float
    position: np.ndarray
    column_coverage: float = 0.0
    """Fraction of image columns in which the model produced a silhouette that
    the image could confirm or contradict -- how much of the frame the score
    actually rests on."""

    def intrinsics(self, width: int, height: int) -> CameraIntrinsics:
        return CameraIntrinsics(
            width=width,
            height=height,
            fx=self.focal_px,
            fy=self.focal_px,
            cx=(width - 1) / 2.0,
            cy=(height - 1) / 2.0,
            model="PINHOLE",
        )


@dataclass
class CoarseSearchConfig:
    azimuth_span_deg: float = 40.0
    azimuth_step_deg: float = 2.0
    elevation_span_deg: float = 12.0
    elevation_step_deg: float = 2.0
    roll_values_deg: tuple[float, ...] = (-4.0, 0.0, 4.0)
    fov_values_deg: tuple[float, ...] = (
        8.0,
        10.0,
        13.0,
        16.0,
        20.0,
        25.0,
        31.0,
        39.0,
        48.0,
        60.0,
        75.0,
    )
    """Horizontal fields of view to try, ~1.25x apart.  The cost is *sharply*
    peaked in focal length -- a 20% scale error moves a feature 100 px from the
    centre by 20 px -- so the ladder has to be fine or the local polish below
    cannot recover."""
    position_offsets: int = 0
    """Extra camera positions sampled on a ring at 1 sigma of the prior.
    0 keeps the prior's nominal position (correct for a fixed vantage point);
    use 4-8 for helicopters, where the prior is an orbit rather than a point."""
    top_k: int = 8
    min_visible_points: int = 12
    """A pose is inadmissible if fewer than this many model samples land in the
    frame.  This is an absolute count on purpose: penalising the *fraction* of
    the model that is visible would systematically favour wide-angle
    hypotheses, since a long lens can only ever see a sliver of an 8 km model."""
    sample_per_metre: float = 0.05
    skyline_weight: float = 2.5
    """Weight of the silhouette term relative to the oriented chamfer.  The
    silhouette is worth more: it is the one feature that survives smoke,
    compression and a bad white balance."""
    skyline_cap_px: float = 40.0
    skyline_unexplained_px: float = 8.0
    """Charged for an image column that has a silhouette the model knows
    nothing about.  It must be well below ``skyline_cap_px`` -- the prior is
    deliberately incomplete, so an unmodelled building is a small cost, not
    evidence against the pose -- but it must not be zero, or the cheapest pose
    is always the one that explains almost nothing (zoom in until only two
    edges are in frame)."""
    chamfer_cap_px: float = 12.0
    """Distance at which a projected model point stops being 'near' an image
    edge.  Points nearer than half this *earn* credit, points further pay: that
    is what makes explaining more of the frame cheaper than explaining less."""
    skyline_steps: int = 48
    """Samples per model segment when rasterising the silhouette envelope."""
    min_skyline_columns: float = 0.04
    """Minimum fraction of image columns in which the model must produce a
    silhouette before a pose is admissible at all."""


def _pose_grid(
    prior: CameraPriorRecord, cfg: CoarseSearchConfig, centre_az: float, centre_el: float
):
    positions = [prior.enu()]
    for i in range(cfg.position_offsets):
        ang = 2 * math.pi * i / max(cfg.position_offsets, 1)
        positions.append(
            prior.enu()
            + np.array(
                [
                    prior.position_sigma_m * math.cos(ang),
                    prior.position_sigma_m * math.sin(ang),
                    0.0,
                ]
            )
        )
    az = np.arange(
        centre_az - cfg.azimuth_span_deg / 2,
        centre_az + cfg.azimuth_span_deg / 2 + 1e-9,
        cfg.azimuth_step_deg,
    )
    el = np.arange(
        centre_el - cfg.elevation_span_deg / 2,
        centre_el + cfg.elevation_span_deg / 2 + 1e-9,
        cfg.elevation_step_deg,
    )
    return positions, az, el


class _Projector:
    """Projects model samples and their local edge orientations for one pose."""

    def __init__(self, model: WireModel, cfg: CoarseSearchConfig):
        self.cfg = cfg
        self.pts, self.seg_idx, self.wts = model.sample(per_metre=cfg.sample_per_metre)
        self.seg_a = model.segments[self.seg_idx, 0]
        self.seg_b = model.segments[self.seg_idx, 1]
        # dense samples along every segment, used for the silhouette envelope
        t = np.linspace(0.0, 1.0, cfg.skyline_steps)[None, :, None]
        a = model.segments[:, 0][:, None, :]
        b = model.segments[:, 1][:, None, :]
        self.dense = (a + t * (b - a)).reshape(-1, 3)

    def silhouette(self, c2w: np.ndarray, focal: float, cx: float, cy: float, width: int):
        """Per-column topmost model row (the rendered skyline), ``inf`` where
        the model projects nothing into that column."""
        w2c = invert_rigid(c2w)
        cam = self.dense @ w2c[:3, :3].T + w2c[:3, 3]
        z = cam[:, 2]
        ok = z > 1.0
        if not ok.any():
            return np.full(width, np.inf)
        u = focal * cam[ok, 0] / z[ok] + cx
        v = focal * cam[ok, 1] / z[ok] + cy
        inside = (u >= 0) & (u < width)
        if not inside.any():
            return np.full(width, np.inf)
        cols = u[inside].astype(np.int64)
        out = np.full(width, np.inf)
        np.minimum.at(out, cols, v[inside])
        return out

    def project(self, c2w: np.ndarray, focal: float, cx: float, cy: float):
        w2c = invert_rigid(c2w)
        r, t = w2c[:3, :3], w2c[:3, 3]

        def _proj(p):
            cam = p @ r.T + t
            z = np.where(cam[:, 2] > 1e-6, cam[:, 2], 1e-6)
            return np.column_stack([focal * cam[:, 0] / z + cx, focal * cam[:, 1] / z + cy]), cam[
                :, 2
            ]

        uv, z = _proj(self.pts)
        uva, _ = _proj(self.seg_a)
        uvb, _ = _proj(self.seg_b)
        d = uvb - uva
        orient = np.arctan2(d[:, 1], d[:, 0]) % math.pi
        return uv, z, orient


def _score(
    proj: _Projector,
    c2w: np.ndarray,
    focal: float,
    target: EdgeTarget,
    cfg: CoarseSearchConfig,
) -> tuple[float, float]:
    """Cost of one pose: oriented chamfer + silhouette.  Lower is better.

    Both terms are written so that *explaining more of the frame is cheaper*.
    A naive mean over whatever happens to be visible is degenerate in both
    directions: normalising by the visible model makes a long lens look good
    (only a sliver of the model is on trial), and normalising by the whole
    model makes a wide lens look good (everything is nominally in frame).  So
    each term is a sum over a fixed denominator -- the whole model for the
    chamfer, every image column for the silhouette -- with matched features
    scoring *below* zero and unmatched ones above it.

    Returns ``(cost, column_coverage)``; ``inf`` for an inadmissible pose.
    """
    cx, cy = (target.width - 1) / 2.0, (target.height - 1) / 2.0
    uv, z, orient = proj.project(c2w, focal, cx, cy)
    ok = (
        (z > 1.0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < target.width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < target.height)
    )
    if int(ok.sum()) < cfg.min_visible_points:
        return float("inf"), 0.0
    cap = min(cfg.chamfer_cap_px, target.max_dist_px)
    d = np.minimum(target.lookup(uv[ok], orient[ok]), cap)
    chamfer = float((proj.wts[ok] * (d - cap / 2.0)).sum() / max(proj.wts.sum(), 1e-9))
    sky_cost, coverage = _skyline_cost(proj, c2w, focal, target, cfg)
    if not math.isfinite(sky_cost):
        return float("inf"), coverage
    return chamfer + cfg.skyline_weight * sky_cost, coverage


def _skyline_cost(
    proj: _Projector,
    c2w: np.ndarray,
    focal: float,
    target: EdgeTarget,
    cfg: CoarseSearchConfig,
) -> tuple[float, float]:
    """Silhouette mismatch in pixels per image column, and the model's coverage.

    Every column of the frame contributes exactly one of:

    * model and image both have a silhouette -> the clipped row difference.
      This is the term that pins rotation and focal length.
    * model has a silhouette, the image column is **sky all the way down** ->
      the full cap.  Hanging a building in open sky is contradicted by the
      image, and charging for it is what stops the search from zooming out.
    * the image has a silhouette the model does not know about ->
      ``skyline_unexplained_px``.  Small, because the prior is incomplete.
    * both agree there is nothing, or the column is unknown -> zero.
    """
    if target.skyline.size != target.width:
        return 0.0, 0.0
    cx, cy = (target.width - 1) / 2.0, (target.height - 1) / 2.0
    model_sky = proj.silhouette(c2w, focal, cx, cy, target.width)
    has_model = np.isfinite(model_sky)
    img = target.skyline
    matched = has_model & np.isfinite(img)
    contradicted = has_model & np.isinf(img)
    unexplained = (~has_model) & np.isfinite(img)
    n_model = int(matched.sum()) + int(contradicted.sum())
    coverage = n_model / max(target.width, 1)
    if n_model < max(4, cfg.min_skyline_columns * target.width):
        return float("inf"), coverage
    total = float(np.minimum(np.abs(model_sky[matched] - img[matched]), cfg.skyline_cap_px).sum())
    total += cfg.skyline_cap_px * float(contradicted.sum())
    total += cfg.skyline_unexplained_px * float(unexplained.sum())
    return total / max(target.width, 1), coverage


def coarse_search(
    target: EdgeTarget,
    prior: CameraPriorRecord,
    model: WireModel | None = None,
    config: CoarseSearchConfig | None = None,
    centre_azimuth_deg: float | None = None,
    centre_elevation_deg: float | None = None,
) -> list[PoseHypothesis]:
    """Stage (a): grid search around the prior.  Returns the top-k hypotheses."""
    from wtc4d.camreg.priors import bearing_to_wtc_deg, elevation_to_wtc_top_deg

    cfg = config or CoarseSearchConfig()
    mdl = model if model is not None else prior_wireframe()
    proj = _Projector(mdl, cfg)
    az0 = centre_azimuth_deg if centre_azimuth_deg is not None else bearing_to_wtc_deg(prior)
    el0 = (
        centre_elevation_deg
        if centre_elevation_deg is not None
        else elevation_to_wtc_top_deg(prior) / 2.0
    )
    positions, azs, els = _pose_grid(prior, cfg, az0, el0)
    out: list[PoseHypothesis] = []
    for pos in positions:
        for fov in cfg.fov_values_deg:
            focal = target.width / (2.0 * math.tan(math.radians(fov) / 2.0))
            for az in azs:
                for el in els:
                    for roll in cfg.roll_values_deg:
                        c2w = pose_from_look(pos, float(az), float(el), float(roll))
                        cost, frac = _score(proj, c2w, focal, target, cfg)
                        if math.isfinite(cost):
                            out.append(
                                PoseHypothesis(
                                    c2w=c2w,
                                    focal_px=focal,
                                    score=cost,
                                    azimuth_deg=float(az),
                                    elevation_deg=float(el),
                                    roll_deg=float(roll),
                                    position=np.asarray(pos, dtype=np.float64),
                                    column_coverage=frac,
                                )
                            )
    out.sort(key=lambda hyp: hyp.score)
    return _spread_out(out, cfg.top_k)


def _spread_out(hyps: list[PoseHypothesis], k: int, min_sep_deg: float = 3.0):
    """Keep the best k hypotheses that are not near-duplicates of each other."""
    kept: list[PoseHypothesis] = []
    for h in hyps:
        if all(
            abs(h.azimuth_deg - g.azimuth_deg) + abs(h.elevation_deg - g.elevation_deg)
            > min_sep_deg
            or abs(math.log(h.focal_px / g.focal_px)) > 0.15
            for g in kept
        ):
            kept.append(h)
        if len(kept) >= k:
            break
    return kept


def _local_polish(
    target: EdgeTarget, hyp: PoseHypothesis, proj: _Projector, cfg: CoarseSearchConfig
) -> PoseHypothesis:
    """A small dense grid around one hypothesis, mainly to fix the focal length.

    Powell needs to start inside the basin; one rung of the coarse focal ladder
    is 25% of the focal length, which is often outside it.  This costs a few
    hundred evaluations and removes that failure mode.
    """
    best = hyp
    scales = np.geomspace(0.78, 1.28, 9)
    d_ang = (-1.0, -0.5, 0.0, 0.5, 1.0)
    for sc in scales:
        for daz in d_ang:
            for dele in d_ang:
                az = hyp.azimuth_deg + daz * cfg.azimuth_step_deg
                el = hyp.elevation_deg + dele * cfg.elevation_step_deg
                focal = hyp.focal_px * sc
                c2w = pose_from_look(hyp.position, az, el, hyp.roll_deg)
                score, frac = _score(proj, c2w, focal, target, cfg)
                if score < best.score:
                    best = PoseHypothesis(
                        c2w=c2w,
                        focal_px=float(focal),
                        score=score,
                        azimuth_deg=float(az),
                        elevation_deg=float(el),
                        roll_deg=hyp.roll_deg,
                        position=hyp.position,
                        column_coverage=frac,
                    )
    return best


def refine_hypothesis(
    target: EdgeTarget,
    hyp: PoseHypothesis,
    model: WireModel | None = None,
    config: CoarseSearchConfig | None = None,
    free_position: bool = False,
    prior: CameraPriorRecord | None = None,
) -> PoseHypothesis:
    """Stage (b): continuous refinement of one hypothesis (Powell on the chamfer).

    ``free_position`` also optimises the camera centre, which only makes sense
    when the geometry has real depth spread; for a distant skyline leave it off
    and let :func:`wtc4d.camreg.pnp.solve_pose` handle position with its prior.
    """
    from scipy.optimize import minimize

    cfg = config or CoarseSearchConfig()
    mdl = model if model is not None else prior_wireframe()
    proj = _Projector(mdl, cfg)
    pos0 = hyp.position.copy()
    sigma = prior.position_sigma_m if prior is not None else 100.0

    def unpack(x):
        az, el, roll, logf = x[0], x[1], x[2], x[3]
        pos = pos0 + x[4:7] * sigma if free_position else pos0
        return pos, az, el, roll, math.exp(logf)

    def cost(x):
        pos, az, el, roll, focal = unpack(x)
        c2w = pose_from_look(pos, az, el, roll)
        c, _ = _score(proj, c2w, focal, target, cfg)
        return c if math.isfinite(c) else 1e6

    hyp = _local_polish(target, hyp, proj, cfg)
    x0 = [hyp.azimuth_deg, hyp.elevation_deg, hyp.roll_deg, math.log(hyp.focal_px)]
    steps = [cfg.azimuth_step_deg, cfg.elevation_step_deg, 3.0, 0.12]
    if free_position:
        x0 += [0.0, 0.0, 0.0]
        steps += [0.5, 0.5, 0.2]
    # Powell's default unit directions are far too small for the focal (a step
    # of 1 in log f is a 2.7x zoom) and too large for nothing; set them by hand.
    direc = np.diag(steps)
    res = minimize(
        cost,
        np.array(x0),
        method="Powell",
        options={"direc": direc, "xtol": 1e-3, "ftol": 1e-4, "maxiter": 40},
    )
    pos, az, el, roll, focal = unpack(res.x)
    c2w = pose_from_look(pos, az, el, roll)
    score, frac = _score(proj, c2w, focal, target, cfg)
    return PoseHypothesis(
        c2w=c2w,
        focal_px=focal,
        score=score,
        azimuth_deg=float(az),
        elevation_deg=float(el),
        roll_deg=float(roll),
        position=np.asarray(pos),
        column_coverage=frac,
    )


# --- stage (c): correspondences -------------------------------------------------
def edge_correspondences(
    target: EdgeTarget,
    hyp: PoseHypothesis,
    model: WireModel,
    search_px: float = 12.0,
    config: CoarseSearchConfig | None = None,
    occlusion: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Snap model samples to image edges -> ``(points3d, uv, sigma_px)``.

    For each projected model sample, walk along the *normal* of the model edge
    (the direction the edge can actually move in the image) up to
    ``search_px`` and take the nearest image edge pixel of a compatible
    orientation.  The 3D point is known exactly, so this yields real 2D-3D
    correspondences for :func:`wtc4d.camreg.pnp.solve_pose`.

    With ``occlusion`` set, samples that the depth buffer says are hidden
    behind other geometry are dropped.
    """
    cfg = config or CoarseSearchConfig()
    proj = _Projector(model, cfg)
    cx, cy = (target.width - 1) / 2.0, (target.height - 1) / 2.0
    uv, z, orient = proj.project(hyp.c2w, hyp.focal_px, cx, cy)
    ok = (
        (z > 1.0)
        & (uv[:, 0] >= 1)
        & (uv[:, 0] < target.width - 1)
        & (uv[:, 1] >= 1)
        & (uv[:, 1] < target.height - 1)
    )
    if occlusion:
        ok &= _visible_mask(proj.pts, z, uv, hyp, target)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return np.zeros((0, 3)), np.zeros((0, 2)), np.zeros(0)

    normals = np.column_stack([-np.sin(orient[idx]), np.cos(orient[idx])])
    steps = np.arange(-search_px, search_px + 1e-9, 1.0)
    best_d = np.full(idx.size, np.inf)
    best_uv = uv[idx].copy()
    for s in steps:
        cand = uv[idx] + s * normals
        u = np.clip(np.round(cand[:, 0]).astype(int), 0, target.width - 1)
        v = np.clip(np.round(cand[:, 1]).astype(int), 0, target.height - 1)
        hit = target.edges[v, u]
        better = hit & (abs(s) < best_d)
        best_d[better] = abs(s)
        best_uv[better] = cand[better]
    found = np.isfinite(best_d)
    if not found.any():
        return np.zeros((0, 3)), np.zeros((0, 2)), np.zeros(0)
    sel = idx[found]
    # sigma grows with how far we had to search: a snap at 10 px is a guess
    sigma = np.clip(1.0 + best_d[found], 1.0, search_px)
    return proj.pts[sel], best_uv[found], sigma


def _visible_mask(pts, z, uv, hyp: PoseHypothesis, target: EdgeTarget) -> np.ndarray:
    """Hidden-line removal using a depth render of the synthetic prior."""
    try:
        from wtc4d.camreg.synthetic import default_scene

        intr = hyp.intrinsics(target.width, target.height)
        depth = default_scene().render(hyp.c2w, intr, texture=False).depth
    except Exception:  # noqa: BLE001 -- occlusion is an optimisation, never a hard requirement
        return np.ones(len(pts), dtype=bool)
    u = np.clip(np.round(uv[:, 0]).astype(int), 0, target.width - 1)
    v = np.clip(np.round(uv[:, 1]).astype(int), 0, target.height - 1)
    d = depth[v, u]
    # a sample on a silhouette edge is legitimately at the depth discontinuity,
    # so allow a generous slack rather than an exact depth test
    return ~np.isfinite(d) | (z <= d * 1.02 + 5.0)


# --- orchestration ---------------------------------------------------------------
@dataclass
class RegistrationResult:
    """Everything :func:`auto_register` learned about one frame."""

    pose: PnPResult | None
    hypothesis: PoseHypothesis
    chamfer_px: float
    n_correspondences: int
    prior_id: str
    stage: str
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        head = (
            f"[{self.prior_id}] chamfer={self.chamfer_px:.2f} px, "
            f"{self.n_correspondences} correspondences, stage={self.stage}"
        )
        return head if self.pose is None else head + "\n  " + self.pose.summary()


def auto_register(
    image: np.ndarray,
    prior: CameraPriorRecord,
    mask: np.ndarray | None = None,
    model: WireModel | None = None,
    config: CoarseSearchConfig | None = None,
    pnp_config: PnPConfig | None = None,
    refine_top: int = 3,
    search_px: float = 12.0,
) -> RegistrationResult:
    """Run stages (a), (b) and (c) on one frame.  See the module docstring."""
    cfg = config or CoarseSearchConfig()
    mdl = model if model is not None else prior_wireframe()
    target = EdgeTarget.from_image(image, mask=mask)
    warnings: list[str] = []
    if target.edges.mean() < 0.005:
        warnings.append("almost no edges in the frame: too dark, too smoky or too blurred")

    hyps = coarse_search(target, prior, mdl, cfg)
    if not hyps:
        return RegistrationResult(
            pose=None,
            hypothesis=PoseHypothesis(np.eye(4), 0.0, float("inf"), 0.0, 0.0, 0.0, prior.enu()),
            chamfer_px=float("inf"),
            n_correspondences=0,
            prior_id=prior.id,
            stage="coarse_failed",
            warnings=warnings + ["coarse search found no admissible pose"],
        )

    best = min(
        (refine_hypothesis(target, h, mdl, cfg, prior=prior) for h in hyps[:refine_top]),
        key=lambda h: h.score,
    )

    pts3d, uv, sigma = edge_correspondences(target, best, mdl, search_px=search_px, config=cfg)
    if len(pts3d) < 6:
        warnings.append(f"only {len(pts3d)} edge correspondences; returning the chamfer pose only")
        return RegistrationResult(
            pose=None,
            hypothesis=best,
            chamfer_px=best.score,
            n_correspondences=int(len(pts3d)),
            prior_id=prior.id,
            stage="refined",
            warnings=warnings,
        )

    pcfg = pnp_config or PnPConfig(
        fix_focal=None, robust=True, n_refine=1, fov_min_deg=4.0, fov_max_deg=120.0
    )
    try:
        pose = solve_pose(
            pts3d,
            uv,
            (target.width, target.height),
            sigmas_px=sigma,
            prior=prior,
            config=pcfg,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"PnP on edge correspondences failed: {exc}")
        pose = None
    return RegistrationResult(
        pose=pose,
        hypothesis=best,
        chamfer_px=best.score,
        n_correspondences=int(len(pts3d)),
        prior_id=prior.id,
        stage="pnp" if pose is not None else "refined",
        warnings=warnings + (pose.warnings if pose else []),
    )


def match_learned(image_a: np.ndarray, image_b: np.ndarray, max_keypoints: int = 1024):
    """Learned dense matching between a frame and a *rendered* view (optional).

    Interface for stage (b') -- LightGlue + SuperPoint (the ``lightglue``
    package) or kornia's ``LoFTR``.  Both are torch-based and therefore not
    installed by the ``camreg`` extra; this raises with instructions instead of
    pretending.  The intended use is: render the prior *textured* from the
    refined hypothesis, match render <-> frame, lift the render-side keypoints
    to 3D through the depth buffer, and hand the correspondences to
    :func:`wtc4d.camreg.pnp.solve_pose` exactly as
    :func:`edge_correspondences` does.

    This is the piece that should be run at scale on GPU (see ``infra/``);
    everything else in this module is CPU-friendly.
    """
    try:
        import torch  # noqa: F401
        from lightglue import LightGlue, SuperPoint  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "learned matching needs torch + lightglue, which are not part of the "
            "camreg extra (CPU CI must stay small). Install them in the GPU image: "
            "pip install torch lightglue"
        ) from exc
    raise NotImplementedError("wired up in the GPU image; see wtc4d/camreg/README.md 'next steps'")
