"""Shot boundary detection, camera-motion heuristic, and per-shot quality.

Scene cuts come from PySceneDetect's ``ContentDetector``. Camera motion is
classified from dense optical-flow statistics sampled across each shot: this
is a heuristic (documented thresholds below), not a trained classifier --
good enough to triage "is this shot worth registering a camera for", not a
ground truth label.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from wtc4d.corpus.quality import quality_score
from wtc4d.schema import Shot, TimeEstimate

logger = logging.getLogger(__name__)

MAX_SAMPLES_PER_SHOT = 8  # for quality scoring (spread across the shot)
FLOW_BURST_LEN = 10  # consecutive frames for optical flow (must be *adjacent*)
FLOW_RESIZE_WIDTH = 160  # downscale before optical flow, for speed

# Motion-classification thresholds (heuristic; see module docstring).
STATIC_MAG_THRESH = 0.4  # px/frame at FLOW_RESIZE_WIDTH scale
RADIAL_THRESH = 0.35  # |mean radial dot product|, zoom indicator
ANGULAR_CIRCVAR_THRESH = 0.35  # direction consistency: below = "coherent"
MAG_CV_THRESH = 0.6  # spatial magnitude coefficient of variation: below = "uniform"


def _sample_frame_indices(start_frame: int, end_frame: int, max_samples: int) -> list[int]:
    n = max(1, min(max_samples, end_frame - start_frame))
    if end_frame - start_frame <= 1:
        return [start_frame]
    return sorted(set(np.linspace(start_frame, end_frame - 1, num=n, dtype=int).tolist()))


def _consecutive_burst_start(start_frame: int, end_frame: int, burst_len: int) -> tuple[int, int]:
    """Pick a short *contiguous* run of frames near the shot's middle.

    Optical flow needs true adjacent frames (small displacement); sampling
    widely-spaced frames for speed (as quality scoring does) breaks flow
    estimation on any real inter-frame motion.
    """
    shot_len = max(1, end_frame - start_frame)
    n = min(burst_len, shot_len)
    mid = start_frame + shot_len // 2
    burst_start = max(start_frame, min(end_frame - n, mid - n // 2))
    return burst_start, n


def _read_consecutive_gray_frames(
    cap: cv2.VideoCapture, start_frame: int, count: int
) -> list[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(count):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape[1] > FLOW_RESIZE_WIDTH:
            scale = FLOW_RESIZE_WIDTH / gray.shape[1]
            gray = cv2.resize(gray, (FLOW_RESIZE_WIDTH, max(1, int(gray.shape[0] * scale))))
        frames.append(gray)
    return frames


def _read_gray_frames(cap: cv2.VideoCapture, indices: list[int]) -> list[np.ndarray]:
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape[1] > FLOW_RESIZE_WIDTH:
            scale = FLOW_RESIZE_WIDTH / gray.shape[1]
            gray = cv2.resize(gray, (FLOW_RESIZE_WIDTH, max(1, int(gray.shape[0] * scale))))
        frames.append(gray)
    return frames


def _flow_stats(prev: np.ndarray, nxt: np.ndarray) -> dict | None:
    flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    fx, fy = flow[..., 0], flow[..., 1]
    mag = np.sqrt(fx**2 + fy**2)
    avg_mag = float(mag.mean())
    if avg_mag < 1e-6:
        return {"avg_mag": 0.0, "mag_cv": 0.0, "ang_circvar": 1.0, "radial_dot": 0.0}
    mag_cv = float(mag.std() / (avg_mag + 1e-9))

    angles = np.arctan2(fy, fx)
    weights = mag
    r_x = float((weights * np.cos(angles)).sum())
    r_y = float((weights * np.sin(angles)).sum())
    resultant = np.hypot(r_x, r_y)
    ang_circvar = 1.0 - resultant / (weights.sum() + 1e-9)

    h, w = prev.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    rx, ry = (xx - cx), (yy - cy)
    r_norm = np.hypot(rx, ry) + 1e-9
    radial_unit_x, radial_unit_y = rx / r_norm, ry / r_norm
    flow_unit_x, flow_unit_y = fx / (mag + 1e-9), fy / (mag + 1e-9)
    dot = radial_unit_x * flow_unit_x + radial_unit_y * flow_unit_y
    radial_dot = float((dot * weights).sum() / (weights.sum() + 1e-9))

    return {
        "avg_mag": avg_mag,
        "mag_cv": mag_cv,
        "ang_circvar": float(ang_circvar),
        "radial_dot": radial_dot,
    }


def classify_motion(stats: dict) -> str:
    """Map aggregated flow stats (see :func:`_flow_stats`) to a motion label."""
    if stats["avg_mag"] < STATIC_MAG_THRESH:
        return "static"
    if abs(stats["radial_dot"]) > RADIAL_THRESH:
        return "zoom"
    if stats["ang_circvar"] < ANGULAR_CIRCVAR_THRESH:
        return "pan_tilt" if stats["mag_cv"] < MAG_CV_THRESH else "aerial"
    return "handheld"


def _shot_motion_and_quality(
    cap: cv2.VideoCapture, start_frame: int, end_frame: int
) -> tuple[str, float | None]:
    indices = _sample_frame_indices(start_frame, end_frame, MAX_SAMPLES_PER_SHOT)
    frames = _read_gray_frames(cap, indices)
    if not frames:
        return "unknown", None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or frames[0].shape[1]
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frames[0].shape[0]
    quality = float(np.mean([quality_score(f, width=width, height=height) for f in frames]))

    burst_start, burst_len = _consecutive_burst_start(start_frame, end_frame, FLOW_BURST_LEN)
    burst_frames = _read_consecutive_gray_frames(cap, burst_start, burst_len)
    if len(burst_frames) < 2:
        return "unknown", quality

    all_stats = [
        s
        for s in (_flow_stats(a, b) for a, b in zip(burst_frames, burst_frames[1:], strict=False))
        if s
    ]
    if not all_stats:
        return "unknown", quality
    avg_stats = {k: float(np.mean([s[k] for s in all_stats])) for k in all_stats[0]}
    return classify_motion(avg_stats), quality


def detect_shots(
    video_path: Path,
    source_id: str,
    *,
    source_time_hint: TimeEstimate | None = None,
) -> list[Shot]:
    """Detect shot boundaries in ``video_path`` and classify each one."""
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector

    video = open_video(str(video_path))
    fps = video.frame_rate

    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not scene_list:
        scene_list = [(0, max(1, total_frames))]
    else:
        scene_list = [(start.get_frames(), end.get_frames()) for start, end in scene_list]

    shots = []
    try:
        for idx, (start_frame, end_frame) in enumerate(scene_list):
            motion, quality = _shot_motion_and_quality(cap, start_frame, end_frame)
            time_est = None
            if source_time_hint is not None:
                time_est = TimeEstimate(
                    t=source_time_hint.t + start_frame / fps,
                    sigma=source_time_hint.sigma,
                    method=source_time_hint.method,
                    evidence=f"propagated from source {source_id} time_hint + frame offset",
                    derived_from=[source_id],
                )
            shots.append(
                Shot(
                    id=f"{source_id}-shot{idx:04d}",
                    source_id=source_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    fps=fps,
                    camera_motion=motion,
                    quality_score=quality,
                    time=time_est,
                )
            )
    finally:
        cap.release()
    logger.info("detect_shots(%s): %d shots", source_id, len(shots))
    return shots
