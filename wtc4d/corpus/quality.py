"""Per-frame quality metrics: sharpness, blockiness, and a combined score.

All functions take a grayscale ``numpy.ndarray`` (uint8) frame. No heavy
dependency beyond OpenCV/numpy (already required by ``shots.py``).
"""

from __future__ import annotations

import cv2
import numpy as np


def laplacian_variance(gray: np.ndarray) -> float:
    """Sharpness proxy: variance of the Laplacian. Low -> blurry/soft focus."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def blockiness_score(gray: np.ndarray, block_size: int = 8) -> float:
    """0..1 estimate of 8x8 DCT blocking artifacts (heavy MPEG re-encoding).

    Measures the mean absolute pixel discontinuity *at* block boundaries
    relative to discontinuity *within* blocks; a value near 1 means edges
    line up suspiciously well with the block grid (compression artifact), a
    value near 0 means no excess energy at block boundaries.
    """
    h, w = gray.shape
    if h < block_size * 2 or w < block_size * 2:
        return 0.0
    img = gray.astype(np.float64)

    col_diff = np.abs(np.diff(img, axis=1))
    boundary_cols = np.arange(block_size, w, block_size) - 1
    boundary_cols = boundary_cols[boundary_cols < col_diff.shape[1]]
    if len(boundary_cols) == 0:
        return 0.0
    at_boundary = col_diff[:, boundary_cols].mean()
    overall = col_diff.mean()

    row_diff = np.abs(np.diff(img, axis=0))
    boundary_rows = np.arange(block_size, h, block_size) - 1
    boundary_rows = boundary_rows[boundary_rows < row_diff.shape[0]]
    at_boundary_r = row_diff[boundary_rows, :].mean() if len(boundary_rows) else 0.0
    overall_r = row_diff.mean()

    boundary = (at_boundary + at_boundary_r) / 2.0
    inner = (overall + overall_r) / 2.0
    if inner < 1e-6:
        return 0.0
    ratio = boundary / inner
    # ratio ~1 => no excess boundary energy; >1 => blocking. Squash to 0..1.
    return float(np.clip((ratio - 1.0), 0.0, 1.0))


def quality_score(
    gray: np.ndarray, *, width: int, height: int, sharpness_ref: float = 200.0
) -> float:
    """Combine resolution, sharpness and blockiness into a single 0..1 score.

    Heuristic, not calibrated against human judgments: resolution term
    saturates at 1080p, sharpness term saturates around ``sharpness_ref``
    (typical Laplacian variance for a reasonably sharp SD/HD frame), and
    blockiness directly subtracts.
    """
    res_term = min(1.0, (width * height) / (1920 * 1080))
    sharp = laplacian_variance(gray)
    sharp_term = min(1.0, sharp / sharpness_ref)
    block_term = blockiness_score(gray)
    score = 0.4 * res_term + 0.5 * sharp_term - 0.2 * block_term
    return float(np.clip(score, 0.0, 1.0))
