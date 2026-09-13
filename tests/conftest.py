"""Shared fixtures. Synthetic-video helpers avoid any dependency on ffmpeg
being installed on the test host: they build frames directly with
``cv2.VideoWriter`` (mp4v), which is always available once opencv is."""

from __future__ import annotations

import pytest


def _write_video(path, frames, fps):
    import cv2

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


@pytest.fixture
def static_video(tmp_path):
    """A short video where every frame is identical (no motion)."""
    pytest.importorskip("cv2")
    import numpy as np

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    path = tmp_path / "static.mp4"
    _write_video(path, [frame] * 20, fps=10)
    return path


@pytest.fixture
def pan_video(tmp_path):
    """A short video panning across a random-noise background (uniform flow)."""
    pytest.importorskip("cv2")
    import numpy as np

    w, h, fps, n = 160, 120, 15, 30
    rng = np.random.default_rng(1)
    big = rng.integers(0, 255, size=(h, w * 3, 3), dtype=np.uint8)
    frames = [big[:, i * 2 : i * 2 + w] for i in range(n)]
    path = tmp_path / "pan.mp4"
    _write_video(path, frames, fps=fps)
    return path
