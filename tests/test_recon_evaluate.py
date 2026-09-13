"""Metrics and per-gaussian coverage."""

from __future__ import annotations

import torch

from wtc4d.recon.conventions import look_at_c2w
from wtc4d.recon.data import FrameSample
from wtc4d.recon.evaluate import coverage, evaluate_dataset, psnr, ssim
from wtc4d.recon.gaussians import Gaussians
from wtc4d.schema.camera import CameraIntrinsics

W, H = 20, 15
INTR = CameraIntrinsics(width=W, height=H, fx=25.0, fy=25.0, cx=W / 2, cy=H / 2)


def _sample(shot_id="s0", frame_idx=0, image=None, mask=None, radius=900.0, t=None) -> FrameSample:
    c2w = torch.as_tensor(look_at_c2w([radius, 0.0, 200.0], [0.0, 0.0, 200.0]), dtype=torch.float32)
    return FrameSample(
        index=0,
        shot_id=shot_id,
        frame_idx=frame_idx,
        c2w=c2w,
        intrinsics=INTR,
        width=W,
        height=H,
        image=image,
        dynamic_mask=mask,
        t=t,
    )


def test_psnr_perfect_and_zero():
    img = torch.rand(H, W, 3)
    assert psnr(img, img) >= 99.0
    noisy = (img + 1.0).clamp(0, 1)
    assert psnr(noisy, img) < psnr(img, img)


def test_psnr_respects_weight_mask():
    gt = torch.zeros(H, W, 3)
    pred = torch.zeros(H, W, 3)
    pred[:, : W // 2] = 1.0  # only the left half is wrong
    weight = torch.zeros(H, W, 1)
    weight[:, W // 2 :] = 1.0  # supervise only the correct half
    assert psnr(pred, gt, weight) >= 99.0
    assert psnr(pred, gt) < 20.0


def test_psnr_weight_all_zero_is_nan():
    img = torch.rand(H, W, 3)
    weight = torch.zeros(H, W, 1)
    assert psnr(img, img, weight) != psnr(img, img, weight)  # NaN != NaN


def test_ssim_identical_images_is_one():
    img = torch.rand(H, W, 3)
    assert float(ssim(img, img)) == 1.0 or abs(float(ssim(img, img)) - 1.0) < 1e-4


def test_ssim_decreases_with_noise():
    torch.manual_seed(0)
    img = torch.rand(H, W, 3)
    noisy = (img + torch.randn(H, W, 3) * 0.3).clamp(0, 1)
    assert float(ssim(noisy, img)) < float(ssim(img, img))


def test_ssim_handles_tiny_images():
    img = torch.rand(3, 3, 3)
    val = ssim(img, img)
    assert torch.isfinite(val)


def test_evaluate_dataset_reports_psnr_and_per_shot():
    g = Gaussians.from_points([[0.0, 0.0, 200.0]], [[1.0, 0.0, 0.0]], scale_m=15.0, opacity=0.95)
    with torch.no_grad():
        from wtc4d.recon.cpu_raster import render as render_cpu

        rendered = render_cpu(g, _sample().c2w, INTR, W, H, bg_color=(0.2, 0.2, 0.2)).rgb
    ds = [_sample(shot_id="s0", image=rendered), _sample(shot_id="s1", image=rendered)]
    report = evaluate_dataset(g, ds, bg_color=(0.2, 0.2, 0.2), backend="cpu")
    assert len(report.frames) == 2
    assert report.psnr_mean > 40.0
    assert set(report.per_shot_psnr) == {"s0", "s1"}
    assert report.n_gaussians == 1


def test_evaluate_dataset_skips_frames_without_images():
    g = Gaussians.from_points([[0.0, 0.0, 200.0]], scale_m=10.0)
    ds = [_sample(image=None)]
    report = evaluate_dataset(g, ds, backend="cpu")
    assert report.frames == []
    assert report.psnr_mean == 0.0


def test_evaluate_dataset_masks_dynamic_pixels_by_default():
    g = Gaussians.from_points([[0.0, 0.0, 200.0]], [[1.0, 0.0, 0.0]], scale_m=15.0, opacity=0.95)
    from wtc4d.recon.cpu_raster import render as render_cpu

    truth = render_cpu(g, _sample().c2w, INTR, W, H, bg_color=(0.2, 0.2, 0.2)).rgb
    corrupted = truth.clone()
    corrupted[:, : W // 2] = 1.0  # a fake "smoke" region that disagrees badly
    mask = torch.zeros(H, W, 1)
    mask[:, : W // 2] = 1.0

    masked = evaluate_dataset(
        g, [_sample(image=corrupted, mask=mask)], bg_color=(0.2, 0.2, 0.2), backend="cpu"
    )
    unmasked = evaluate_dataset(
        g,
        [_sample(image=corrupted, mask=mask)],
        bg_color=(0.2, 0.2, 0.2),
        backend="cpu",
        use_masks=False,
    )
    assert masked.psnr_mean > unmasked.psnr_mean


def test_coverage_counts_frames_and_shots_per_gaussian():
    near = Gaussians.from_points([[0.0, 0.0, 200.0]], [[1.0, 0.0, 0.0]], scale_m=20.0, opacity=0.9)
    far = Gaussians.from_points([[0.0, 0.0, 200.0]], [[0.0, 1.0, 0.0]], scale_m=20.0, opacity=0.9)
    g = Gaussians.cat([near, far])
    # near is visible from every camera below; far is placed behind one of them
    ds = [
        _sample(shot_id="s0", frame_idx=0, radius=900.0),
        _sample(shot_id="s0", frame_idx=1, radius=900.0),
        _sample(shot_id="s1", frame_idx=0, radius=900.0),
    ]
    cov = coverage(g, ds, backend="cpu", bg_color=(0.1, 0.1, 0.1))
    assert cov["n_frames"].shape == (2,)
    assert (cov["n_frames"] > 0).all()
    assert (cov["n_shots"] <= 2).all()
    assert cov["weight"].sum() > 0
