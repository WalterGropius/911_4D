"""Sanity tests for wtc4d.camreg.masks against the synthetic renderer's ground
truth (sky/object ids), plus targeted unit tests for the individual cues."""

from __future__ import annotations

import numpy as np

from wtc4d.camreg.conventions import look_at
from wtc4d.camreg.masks import (
    HeuristicSegmenter,
    MaskConfig,
    dynamic_mask,
    fire_mask,
    rgb_to_hsv,
    sky_mask,
    skyline_profile,
    smoke_mask,
    static_graphics_mask,
    temporal_variance_mask,
)
from wtc4d.camreg.priors import get_prior
from wtc4d.camreg.synthetic import default_scene, degrade
from wtc4d.schema.camera import CameraIntrinsics
from wtc4d.world import WTC1


def _render(hfov_deg: float = 35.0, size=(240, 180)):
    scene = default_scene()
    prior = get_prior("brooklyn_heights_promenade")
    c2w = look_at(prior.enu(), WTC1.enu_center() + np.array([0.0, 0.0, 180.0]))
    w, h = size
    f = w / (2 * np.tan(np.radians(hfov_deg) / 2))
    intr = CameraIntrinsics(width=w, height=h, fx=f, fy=f, cx=(w - 1) / 2, cy=(h - 1) / 2)
    return scene.render(c2w, intr), intr


def test_sky_mask_matches_render_ground_truth() -> None:
    render, _ = _render()
    img = degrade(render.image, scale=0.8, noise_sigma=0.02, blur_sigma_px=0.8)
    pred = sky_mask(img)
    true_sky = render.sky
    iou = (pred & true_sky).sum() / max((pred | true_sky).sum(), 1)
    assert iou > 0.85


def test_skyline_profile_matches_render_within_a_few_pixels() -> None:
    render, _ = _render()
    img = degrade(render.image, scale=0.8, noise_sigma=0.02, blur_sigma_px=0.8)
    pred = skyline_profile(img)
    true_row = np.where(
        render.sky.all(axis=0), np.inf, np.argmax(~render.sky, axis=0).astype(float)
    )
    both = np.isfinite(pred) & np.isfinite(true_row)
    assert both.sum() > 0.3 * len(pred)
    err = np.abs(pred[both] - true_row[both])
    assert np.median(err) < 6.0


def test_skyline_profile_reports_inf_for_open_sky_columns() -> None:
    """A hand-built image: the left half has a silhouette, the right half is
    open sky top to bottom.  ``skyline_profile`` must say ``inf`` for the sky
    half, not merely "unknown" (``nan``) -- the distinction is what lets
    :mod:`wtc4d.camreg.render_match` charge a pose for hanging a building in
    empty sky (see ``EdgeTarget``/``_skyline_cost``)."""
    h, w = 100, 100
    img = np.full((h, w), 0.9)  # bright sky everywhere
    img[40:, :50] = 0.2  # a dark, textured "building" filling the lower-left
    rng = np.random.default_rng(3)
    img[40:, :50] += rng.normal(0, 0.03, (h - 40, 50))
    pred = skyline_profile(img)
    assert np.all(np.isinf(pred[50:]))  # right half: sky all the way down
    assert np.all(np.isfinite(pred[:50]))  # left half: silhouette detected


def test_rgb_to_hsv_primaries() -> None:
    rgb = np.array([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]])
    hsv = rgb_to_hsv(rgb)
    np.testing.assert_allclose(hsv[0, 0], [0.0, 1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(hsv[1, 0], [120.0, 1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(hsv[2, 0], [240.0, 1.0, 1.0], atol=1e-6)


def test_rgb_to_hsv_grayscale_has_zero_saturation() -> None:
    rgb = np.full((5, 5, 3), 0.5)
    hsv = rgb_to_hsv(rgb)
    np.testing.assert_allclose(hsv[..., 1], 0.0, atol=1e-9)


def test_smoke_mask_flags_bright_flat_region_not_sky() -> None:
    h, w = 80, 80
    img = np.full((h, w), 0.15)
    img[10:60, 10:60] = 0.65  # a bright, flat "plume" against a dark background
    img[0:5, :] = 0.95  # a bright, flat sky strip at the very top
    mask = smoke_mask(img)
    assert mask[35, 35]  # centre of the plume region
    assert not mask[2, 40]  # sky, reached from the top edge, must not be "smoke"


def test_fire_mask_flags_orange_not_blue_sky() -> None:
    img = np.zeros((10, 10, 3))
    img[2:5, 2:5] = [0.95, 0.45, 0.05]  # saturated orange flame
    img[7:9, 7:9] = [0.5, 0.7, 0.95]  # blue sky
    mask = fire_mask(img)
    assert mask[3, 3]
    assert not mask[8, 8]


def test_fire_mask_grayscale_input_is_all_false() -> None:
    img = np.random.default_rng(0).uniform(0, 1, (20, 20))
    assert not fire_mask(img).any()


def test_temporal_variance_mask_flags_changing_pixels() -> None:
    rng = np.random.default_rng(1)
    stack = np.full((8, 20, 20), 0.4) + rng.normal(0, 0.005, (8, 20, 20))
    stack[:, 5:10, 5:10] += np.linspace(-0.3, 0.3, 8)[:, None, None]  # a moving/changing patch
    mask = temporal_variance_mask(stack)
    assert mask[7, 7]
    assert not mask[15, 15]


def test_static_graphics_mask_flags_frozen_sharp_bug() -> None:
    rng = np.random.default_rng(2)
    h, w, t = 60, 60, 10
    stack = 0.4 + 0.02 * rng.normal(size=(t, h, w))  # noisy scene content
    # a station bug with internal structure (irregular, like text/a logo),
    # not a flat block or a period-2 checkerboard: real graphics have edges
    # throughout, and a period-2 pattern makes central-difference gradients
    # cancel (x[i-1] == x[i+1] everywhere), which would defeat the test
    # without saying anything about the mask itself.
    bug = (rng.uniform(size=(10, 10)) > 0.5).astype(np.float64)
    for i in range(t):
        stack[i, 45:55, 5:15] = bug  # perfectly constant across every frame
    mask = static_graphics_mask(stack)
    assert mask[45:55, 5:15].mean() > 0.5  # most of the bug is flagged
    assert not mask[30, 30]


def test_dynamic_mask_combines_cues_and_dilates() -> None:
    render, _ = _render()
    img = degrade(render.image, scale=0.8, noise_sigma=0.02, blur_sigma_px=0.8)
    mask = dynamic_mask(img, config=MaskConfig(dilate_radius=1))
    assert mask.shape == img.shape
    assert mask.dtype == bool
    # a clean rendered skyline has no smoke/fire: the mask should be sparse
    assert mask.mean() < 0.05


def test_heuristic_segmenter_matches_dynamic_mask() -> None:
    render, _ = _render()
    img = degrade(render.image, scale=0.8, noise_sigma=0.02, blur_sigma_px=0.8)
    seg = HeuristicSegmenter()
    np.testing.assert_array_equal(seg(img), dynamic_mask(img))
