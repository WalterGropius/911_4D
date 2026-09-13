import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from wtc4d.corpus import quality  # noqa: E402


def test_laplacian_variance_flat_is_zero():
    flat = np.full((64, 64), 128, dtype=np.uint8)
    assert quality.laplacian_variance(flat) == pytest.approx(0.0, abs=1e-6)


def test_laplacian_variance_sharp_edge_is_higher_than_blurred():
    sharp = np.zeros((64, 64), dtype=np.uint8)
    sharp[:, 32:] = 255
    blurred = cv2.GaussianBlur(sharp, (15, 15), 5)
    assert quality.laplacian_variance(sharp) > quality.laplacian_variance(blurred)


def test_blockiness_score_bounds():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
    score = quality.blockiness_score(noise)
    assert 0.0 <= score <= 1.0


def test_blockiness_score_small_image_returns_zero():
    tiny = np.zeros((4, 4), dtype=np.uint8)
    assert quality.blockiness_score(tiny) == 0.0


def test_quality_score_bounds_and_resolution_sensitivity():
    rng = np.random.default_rng(0)
    small = rng.integers(0, 255, size=(64, 64), dtype=np.uint8).astype(np.uint8)
    large = rng.integers(0, 255, size=(1080, 1920), dtype=np.uint8).astype(np.uint8)
    q_small = quality.quality_score(small, width=64, height=64)
    q_large = quality.quality_score(large, width=1920, height=1080)
    assert 0.0 <= q_small <= 1.0
    assert 0.0 <= q_large <= 1.0
    assert q_large >= q_small
