"""Propagate a registered keyframe's pose through the rest of its shot.

One annotated (or auto-registered) frame gives a pose.  A shot is many
frames; re-running PnP-from-landmarks on every frame would need an annotation
on every frame, which does not scale.  Instead:

1. **Lift.**  Detect 2D features in the keyframe, keep the static ones (reject
   the dynamic mask), and back-project each one to 3D by ray casting against
   the prior -- the tower boxes (or ``wtc4d.geo``'s mesh, when available) --
   using the keyframe's own pose.  A feature that does not hit anything (sky,
   or a ray that grazes past every box) is dropped: we cannot use a point whose
   depth we do not know.
2. **Track.**  Follow those 2D points frame-to-frame with pyramidal
   Lucas-Kanade optical flow (``cv2.calcOpticalFlowPyrLK``), re-masking dynamic
   content and dropping tracks that fail the forward-backward consistency
   check.  New features are re-detected and re-lifted whenever too many are
   lost, so a long shot does not run out of points.
3. **Solve.**  Every frame with enough surviving tracks gets its own PnP
   solve (:func:`wtc4d.camreg.pnp.solve_pose`) against the lifted 3D points,
   which is already a big improvement over pure 2D tracking because it is
   metric and self-correcting (accumulated flow drift shows up as reprojection
   error, not as silent pose drift).
4. **Smooth.**  A small sparse windowed bundle adjustment over the keyframes'
   poses only (not the 3D points, which are fixed to the prior) applies one of
   two priors depending on the shot's ``camera_motion``:

   * **handheld / pan_tilt / zoom** -- a smoothness prior: the second
     difference of camera *position* and of camera *orientation* (as a
     rotation-vector delta) is penalised, which damps tracking jitter without
     erasing an intentional pan.
   * **static (tripod / rooftop)** -- a fixed-position prior: position is
     pinned to its mean across the window and only rotation (+ focal, for a
     zoom) vary frame to frame.

What this cannot do
--------------------
Optical flow drifts on low-contrast, smoky or heavily compressed footage; the
per-frame PnP re-solve bounds the *pose* error (bad points show up as outliers
and get RANSAC'd out) but cannot invent points where none survive.  A shot that
loses the skyline behind smoke for many seconds will lose tracking there --
this is exactly where a second annotated keyframe on the far side, tracked
backwards, should meet the forward track partway.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from wtc4d.camreg.conventions import camera_center, rodrigues
from wtc4d.camreg.masks import HeuristicSegmenter, MaskSegmenter
from wtc4d.camreg.pnp import PnPConfig, PnPResult, solve_pose
from wtc4d.schema.camera import CameraPose
from wtc4d.world import TOWERS, TowerSpec

__all__ = [
    "BundleAdjustConfig",
    "TrackConfig",
    "TrackedFrame",
    "bundle_adjust",
    "detect_features",
    "lift_features",
    "ray_box_intersect",
    "track_shot",
    "track_optical_flow",
]


# --- geometry: ray casting against the prior -----------------------------------
def ray_box_intersect(origin: np.ndarray, direction: np.ndarray, box_corners: np.ndarray):
    """Nearest intersection of a ray with an axis-*unaligned* box (8 corners:
    4 base then 4 roof, as returned by :meth:`wtc4d.world.TowerSpec.box_corners`).

    Implemented as a slab test in the box's own (possibly rotated) frame, so it
    works for the towers' true orientation, not just axis-aligned boxes.
    Returns the world-space hit point, or ``None``.
    """
    base = box_corners[:4]
    centre = base.mean(axis=0)
    ex = box_corners[1] - box_corners[0]
    ey = box_corners[3] - box_corners[0]
    ez = np.array([0.0, 0.0, box_corners[4, 2] - box_corners[0, 2]])
    half = np.array([np.linalg.norm(ex), np.linalg.norm(ey), np.linalg.norm(ez)]) / 2.0
    axes = np.column_stack(
        [ex / max(np.linalg.norm(ex), 1e-12), ey / max(np.linalg.norm(ey), 1e-12), [0, 0, 1.0]]
    )
    local_o = axes.T @ (origin - (centre + np.array([0, 0, half[2]])))
    local_d = axes.T @ direction
    tmin, tmax = -np.inf, np.inf
    for i in range(3):
        if abs(local_d[i]) < 1e-12:
            if abs(local_o[i]) > half[i]:
                return None
            continue
        t1 = (-half[i] - local_o[i]) / local_d[i]
        t2 = (half[i] - local_o[i]) / local_d[i]
        t1, t2 = min(t1, t2), max(t1, t2)
        tmin, tmax = max(tmin, t1), min(tmax, t2)
        if tmin > tmax:
            return None
    t = tmin if tmin > 1e-6 else tmax
    if t <= 1e-6:
        return None
    return origin + t * direction


def _ray_ground_intersect(origin: np.ndarray, direction: np.ndarray, z: float = 0.0):
    if abs(direction[2]) < 1e-9:
        return None
    t = (z - origin[2]) / direction[2]
    if t <= 1e-6:
        return None
    return origin + t * direction


def lift_features(
    uv: np.ndarray,
    pose: CameraPose,
    towers: list[TowerSpec] | None = None,
    scene=None,
    ground_z: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project 2D keyframe points to 3D by ray casting against the prior.

    Tries, per point: the tower boxes (or ``scene``, a ``trimesh.Scene`` from
    ``wtc4d.geo.load_scene()``, when given -- nearer geometry wins), then a
    ground-plane fallback at ``ground_z``.  Sky rays hit nothing and are
    dropped.

    Returns ``(points_world (M, 3), kept_idx (M,))``: ``kept_idx`` indexes the
    input ``uv`` array so the caller can align the lifted points with their
    original pixel locations and any per-feature bookkeeping (ids, descriptors).
    """
    c2w = pose.matrix()
    origin = camera_center(c2w)
    intr = pose.intrinsics
    k_inv = np.linalg.inv(intr.K())
    dirs_cam = (k_inv @ np.column_stack([uv, np.ones(len(uv))]).T).T
    dirs_world = dirs_cam @ c2w[:3, :3].T
    dirs_world = dirs_world / np.linalg.norm(dirs_world, axis=1, keepdims=True)

    boxes = [t.box_corners() for t in (towers if towers is not None else TOWERS)]
    pts: list[np.ndarray] = []
    idx: list[int] = []
    for i, d in enumerate(dirs_world):
        best = None
        best_t = np.inf
        if scene is not None:
            hit = _scene_intersect(scene, origin, d)
            if hit is not None:
                t = float(np.linalg.norm(hit - origin))
                best, best_t = hit, t
        for box in boxes:
            hit = ray_box_intersect(origin, d, box)
            if hit is not None:
                t = float(np.linalg.norm(hit - origin))
                if t < best_t:
                    best, best_t = hit, t
        if best is None:
            best = _ray_ground_intersect(origin, d, ground_z)
            if best is not None:
                best_t = float(np.linalg.norm(best - origin))
        if best is not None and np.isfinite(best_t) and best_t < 20000.0:
            pts.append(best)
            idx.append(i)
    if not pts:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    return np.array(pts), np.array(idx, dtype=np.int64)


def _scene_intersect(scene, origin: np.ndarray, direction: np.ndarray):
    """First hit of a ray against a ``trimesh.Scene``, or ``None``."""
    try:
        mesh = scene.dump(concatenate=True) if hasattr(scene, "dump") else scene
        locations, _, _ = mesh.ray.intersects_location(
            ray_origins=origin[None, :], ray_directions=direction[None, :]
        )
    except Exception:  # noqa: BLE001 -- ray casting against an arbitrary mesh is best-effort
        return None
    if len(locations) == 0:
        return None
    d = np.linalg.norm(locations - origin, axis=1)
    return locations[int(np.argmin(d))]


# --- 2D feature detection and tracking -----------------------------------------
def detect_features(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    max_features: int = 400,
    min_distance: int = 10,
) -> np.ndarray:
    """Good-Features-To-Track corners outside ``mask``.  (N, 2) pixel coords."""
    import cv2

    gray = np.asarray(image, dtype=np.float64)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    cv_mask = None
    if mask is not None:
        cv_mask = (~np.asarray(mask, dtype=bool)).astype(np.uint8) * 255
    corners = cv2.goodFeaturesToTrack(
        u8,
        maxCorners=max_features,
        qualityLevel=0.01,
        minDistance=min_distance,
        mask=cv_mask,
    )
    if corners is None:
        return np.zeros((0, 2))
    return corners.reshape(-1, 2)


def track_optical_flow(
    prev_gray_u8: np.ndarray,
    next_gray_u8: np.ndarray,
    prev_pts: np.ndarray,
    fb_threshold_px: float = 1.5,
):
    """Pyramidal LK flow with a forward-backward consistency check.

    Returns ``(next_pts (M, 2), kept_idx (M,))``, ``kept_idx`` indexing
    ``prev_pts``.
    """
    import cv2

    if len(prev_pts) == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=np.int64)
    p0 = prev_pts.reshape(-1, 1, 2).astype(np.float32)
    lk = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray_u8, next_gray_u8, p0, None, **lk)
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(next_gray_u8, prev_gray_u8, p1, None, **lk)
    st1 = st1.reshape(-1).astype(bool)
    st2 = st2.reshape(-1).astype(bool)
    fb_err = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
    good = st1 & st2 & (fb_err < fb_threshold_px)
    idx = np.flatnonzero(good)
    return p1.reshape(-1, 2)[idx], idx


# --- orchestration --------------------------------------------------------------
@dataclass
class TrackConfig:
    max_features: int = 400
    min_feature_distance: int = 10
    """Minimum pixel spacing between detected corners (``cv2.goodFeaturesToTrack``'s
    ``minDistance``).  Lower this for a small or sparsely textured frame, where
    the default otherwise leaves too few candidates to survive lifting."""
    min_tracks: int = 30
    """Re-detect new features once the surviving count drops below this."""
    min_pnp_points: int = 6
    fb_threshold_px: float = 1.5
    ground_z: float = 0.0
    camera_static: bool = False
    pnp: PnPConfig = field(default_factory=PnPConfig)
    segmenter: MaskSegmenter = field(default_factory=HeuristicSegmenter)


@dataclass
class TrackedFrame:
    frame_idx: int
    pose: PnPResult | None
    n_tracks: int
    mean_flow_px: float


def _to_u8(image: np.ndarray) -> np.ndarray:
    g = np.asarray(image, dtype=np.float64)
    if g.ndim == 3:
        g = g.mean(axis=2)
    return np.clip(g * 255.0, 0, 255).astype(np.uint8)


def track_shot(
    frames: list[np.ndarray],
    keyframe_idx: int,
    keyframe_pose: CameraPose,
    frame_indices: list[int] | None = None,
    config: TrackConfig | None = None,
    towers: list[TowerSpec] | None = None,
    scene=None,
) -> list[TrackedFrame]:
    """Propagate ``keyframe_pose`` forward and backward through ``frames``.

    ``frames[keyframe_idx]`` must be the frame ``keyframe_pose`` was solved on.
    Returns one :class:`TrackedFrame` per input frame, in the same order,
    covering both directions from the keyframe.
    """
    cfg = config or TrackConfig()
    n = len(frames)
    idx_out = frame_indices if frame_indices is not None else list(range(n))
    results: dict[int, TrackedFrame] = {}

    def _run(order: list[int]) -> None:
        pose = keyframe_pose
        pts3d = np.zeros((0, 3))
        pts2d = np.zeros((0, 2))
        prev_u8 = _to_u8(frames[order[0]])
        for step, i in enumerate(order):
            img = frames[i]
            u8 = _to_u8(img)
            mask = cfg.segmenter(np.asarray(img, dtype=np.float64), camera_static=cfg.camera_static)
            if step == 0:
                pts2d = detect_features(img, mask, cfg.max_features, cfg.min_feature_distance)
                pts3d, kept = lift_features(pts2d, pose, towers, scene, cfg.ground_z)
                pts2d = pts2d[kept]
                results[i] = TrackedFrame(i, None, len(pts2d), 0.0)
                prev_u8 = u8
                continue
            new2d, kept = track_optical_flow(prev_u8, u8, pts2d, cfg.fb_threshold_px)
            pts3d = pts3d[kept]
            in_mask = mask[
                np.clip(new2d[:, 1].astype(int), 0, mask.shape[0] - 1),
                np.clip(new2d[:, 0].astype(int), 0, mask.shape[1] - 1),
            ]
            keep2 = ~in_mask
            new2d, pts3d = new2d[keep2], pts3d[keep2]
            flow = (
                float(np.mean(np.linalg.norm(new2d - pts2d[kept][keep2], axis=1)))
                if len(new2d)
                else 0.0
            )

            solved = None
            if len(pts3d) >= cfg.min_pnp_points:
                try:
                    solved = solve_pose(
                        pts3d,
                        new2d,
                        (img.shape[1], img.shape[0]),
                        prior=None,
                        config=cfg.pnp,
                    )
                    pose = solved.to_camera_pose(pose.shot_id, i, method="tracking").matrix()
                    pose = keyframe_pose.model_copy(
                        update={"c2w": [float(x) for x in np.asarray(pose).reshape(16)]}
                    )
                except ValueError:
                    solved = None
            results[i] = TrackedFrame(i, solved, len(new2d), flow)
            pts2d, prev_u8 = new2d, u8

            if len(pts2d) < cfg.min_tracks:
                fresh = detect_features(
                    img, mask, cfg.max_features - len(pts2d), cfg.min_feature_distance
                )
                if len(fresh):
                    lifted, kept_f = lift_features(fresh, pose, towers, scene, cfg.ground_z)
                    if len(lifted):
                        pts2d = np.vstack([pts2d, fresh[kept_f]])
                        pts3d = np.vstack([pts3d, lifted])

    fwd = [i for i in range(keyframe_idx, n)]
    bwd = [i for i in range(keyframe_idx, -1, -1)]
    if fwd:
        _run(fwd)
    if len(bwd) > 1:
        _run(bwd)
    return [results[i] for i in idx_out if i in results]


# --- windowed bundle adjustment --------------------------------------------------
@dataclass
class BundleAdjustConfig:
    smoothness_weight: float = 4.0
    """Weight on the second-difference (jerk) prior for a moving camera."""
    fixed_position_weight: float = 200.0
    """Weight pinning position to the window mean for a static camera.  Large
    on purpose: a tripod/rooftop camera's position should barely move at all,
    and with dozens of reprojection residuals per frame (each weighted
    ``1/sigma_px``) a soft prior needs real strength to dominate them."""
    max_nfev: int = 300


def bundle_adjust(
    poses: list[CameraPose],
    observations: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    camera_static: bool = False,
    config: BundleAdjustConfig | None = None,
) -> list[CameraPose]:
    """Refine a short window of keyframe poses jointly.

    ``observations[i]`` is ``(points3d, uv, sigma_px)`` for ``poses[i]`` -- the
    3D points stay fixed (they come from the prior via :func:`lift_features`);
    only camera parameters move.  Reprojection is weighted by ``sigma_px`` as
    in :mod:`wtc4d.camreg.pnp`; in addition, consecutive cameras are coupled by
    a smoothness or fixed-position prior depending on ``camera_static``.

    Needs at least 2 poses; returns them unchanged (with a note) if scipy's
    solver fails to improve on the input.
    """
    from scipy.optimize import least_squares

    if len(poses) < 2:
        return list(poses)
    cfg = config or BundleAdjustConfig()
    n = len(poses)
    rt = [_pose_to_azel_vec(p) for p in poses]  # (rvec(3), t(3), log_f) per frame
    x0 = np.concatenate(rt)
    mean_c = np.mean(
        [rodrigues(r[:3]).T @ (-r[3:6]) for r in rt], axis=0
    )  # mean camera centre across the window, used by the fixed-position prior

    def unpack(x):
        return [x[7 * i : 7 * i + 7] for i in range(n)]

    def residuals(x):
        blocks = unpack(x)
        res = []
        for i, (rvec_t_f, obs) in enumerate(zip(blocks, observations, strict=False)):
            rvec, t, logf = rvec_t_f[:3], rvec_t_f[3:6], rvec_t_f[6]
            pts, uv, sig = obs
            if len(pts):
                R = rodrigues(rvec)  # noqa: N806
                cam = pts @ R.T + t
                z = np.where(cam[:, 2] > 1e-6, cam[:, 2], 1e-6)
                f = math.exp(logf)
                cx, cy = poses[i].intrinsics.cx, poses[i].intrinsics.cy
                u = f * cam[:, 0] / z + cx
                v = f * cam[:, 1] / z + cy
                res.append(((np.column_stack([u, v]) - uv) / sig[:, None]).reshape(-1))
        centres = [
            rodrigues(b[:3]).T @ (-b[3:6]) for b in blocks
        ]  # actual camera centre C = -R^T t, not the raw w2c translation t
        if camera_static:
            for c in centres:
                res.append(cfg.fixed_position_weight * (c - mean_c))
        else:
            for i in range(1, n - 1):
                r0, r1, r2 = blocks[i - 1][:3], blocks[i][:3], blocks[i + 1][:3]
                res.append(
                    cfg.smoothness_weight * (centres[i + 1] - 2 * centres[i] + centres[i - 1])
                )
                res.append(cfg.smoothness_weight * (r2 - 2 * r1 + r0))
        return np.concatenate(res) if res else np.zeros(1)

    result = least_squares(
        residuals, x0, method="lm", max_nfev=cfg.max_nfev * n, xtol=1e-10, ftol=1e-10
    )
    blocks = unpack(result.x)
    out = []
    for i, p in enumerate(poses):
        rvec, t, logf = blocks[i][:3], blocks[i][3:6], blocks[i][6]
        R = rodrigues(rvec)  # noqa: N806
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t
        f = math.exp(logf)
        new_intr = p.intrinsics.model_copy(
            update={"fx": f, "fy": f * p.intrinsics.fy / p.intrinsics.fx}
        )
        out.append(
            p.model_copy(
                update={
                    "c2w": [float(x) for x in c2w.reshape(16)],
                    "intrinsics": new_intr,
                    "method": "tracking+bundle",
                }
            )
        )
    return out


def _pose_to_azel_vec(pose: CameraPose) -> np.ndarray:
    from wtc4d.camreg.pnp import rt_from_c2w  # local import: avoids a cycle at module load

    rvec, tvec = rt_from_c2w(pose.matrix())
    return np.concatenate([rvec, tvec, [math.log(max(pose.intrinsics.fx, 1e-6))]])
