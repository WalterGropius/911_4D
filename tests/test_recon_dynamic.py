"""4D prototype: canonical set + deformation field on the CPU toy "collapse"."""

from __future__ import annotations

import pytest
import torch

from wtc4d.recon.dynamic import (
    DeformationConfig,
    DeformationField,
    DynamicGaussians,
    DynamicTrainConfig,
    PositionEncoding,
    TimeEncoding,
    train_dynamic,
)
from wtc4d.recon.gaussians import Gaussians
from wtc4d.recon.synthetic import toy_dynamic_scene


def test_time_encoding_is_periodic_and_bounded():
    enc = TimeEncoding(n_freq=3)
    out = enc(torch.tensor([0.0, 0.5, 1.0]))
    assert out.shape == (3, enc.dim)
    assert torch.isfinite(out).all()


def test_position_encoding_shape():
    enc = PositionEncoding(n_freq=4, scale=500.0)
    out = enc(torch.randn(10, 3) * 100.0)
    assert out.shape == (10, enc.dim)


def test_deformation_field_starts_at_identity():
    """Zero-initialised heads: at construction the field must predict zero
    deformation everywhere, so training starts from the static solution."""
    field = DeformationField(DeformationConfig())
    means = torch.randn(20, 3) * 50.0
    out = field(means, 0.3)
    assert torch.allclose(out["dmean"], torch.zeros_like(out["dmean"]), atol=1e-6)
    assert torch.allclose(out["dlog_scale"], torch.zeros_like(out["dlog_scale"]), atol=1e-6)


def test_deformation_field_translation_is_bounded():
    cfg = DeformationConfig(max_translation_m=50.0)
    field = DeformationField(cfg)
    with torch.no_grad():
        field.head_mean.weight.normal_(std=10.0)
        field.head_mean.bias.normal_(std=10.0)
    out = field(torch.randn(30, 3) * 100.0, 0.7)
    assert float(out["dmean"].detach().abs().max()) <= 50.0 + 1e-4


def test_dynamic_gaussians_at_t_start_is_near_canonical_before_training():
    canonical = Gaussians.from_points([[0.0, 0.0, 200.0]], scale_m=5.0)
    model = DynamicGaussians(canonical, t_start=0.0, t_end=10.0)
    state = model.at(0.0)
    assert torch.allclose(state.means, canonical.means, atol=1e-5)


def test_dynamic_gaussians_render_matches_backend_signature():
    from wtc4d.recon.conventions import look_at_c2w
    from wtc4d.schema.camera import CameraIntrinsics

    canonical = Gaussians.from_points(
        [[0.0, 0.0, 200.0]], [[1.0, 0.5, 0.2]], scale_m=20.0, opacity=0.9
    )
    model = DynamicGaussians(canonical, t_start=0.0, t_end=10.0)
    c2w = torch.as_tensor(look_at_c2w([900.0, 0.0, 200.0], [0.0, 0.0, 200.0]), dtype=torch.float32)
    intr = CameraIntrinsics(width=16, height=12, fx=25.0, fy=25.0, cx=8.0, cy=6.0)
    out = model.render(3.0, c2w, intr, 16, 12, backend="cpu")
    assert out.rgb.shape == (12, 16, 3)


def test_export_timestep_is_detached_static_snapshot():
    canonical = Gaussians.from_points([[0.0, 0.0, 200.0]], scale_m=10.0)
    model = DynamicGaussians(canonical, t_start=0.0, t_end=10.0)
    snap = model.export_timestep(5.0)
    assert isinstance(snap, Gaussians)
    assert not snap.means.requires_grad


def _train_cfg(tmp_path, **overrides) -> DynamicTrainConfig:
    cfg = DynamicTrainConfig(
        device="cpu",
        backend="cpu",
        iterations=250,
        out_dir=str(tmp_path),
        name="toy",
        log_interval=10,
        seed=0,
        deformation=DeformationConfig(hidden=32, n_layers=3),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_psnr_improves_on_toy_dynamic_scene(tmp_path):
    scene = toy_dynamic_scene(n_cameras=6, n_times=8, width=28, height=20, n_points=90, seed=0)
    cfg = _train_cfg(tmp_path, iterations=300)
    result = train_dynamic(cfg, scene.dataset, scene.init, iterations=300)

    assert len(result.history) > 1
    first_psnr = result.history[0]["psnr"]
    last_psnr = result.history[-1]["psnr"]
    assert last_psnr > first_psnr + 3.0
    assert result.stats["n_frames"] == len(scene.dataset)
    assert result.stats_path.exists()


def test_train_dynamic_requires_timed_frames(tmp_path):
    from wtc4d.recon.synthetic import InMemoryDataset

    canonical = Gaussians.from_points([[0.0, 0.0, 200.0]], scale_m=10.0)
    cfg = _train_cfg(tmp_path)
    with pytest.raises(ValueError, match="absolute times"):
        train_dynamic(cfg, InMemoryDataset([]), canonical)


def test_learned_time_window_fades_a_gaussian_in_and_out(tmp_path):
    """With ``learn_time_window`` the blob's temporal opacity should end up
    concentrated inside the window rather than flat (i.e. it learns to turn on
    and off, not just to exist everywhere)."""
    scene = toy_dynamic_scene(n_cameras=6, n_times=8, width=24, height=18, n_points=70, seed=1)
    cfg = _train_cfg(tmp_path, iterations=300, learn_time_window=True)
    result = train_dynamic(cfg, scene.dataset, scene.init, iterations=300)

    model = result.model
    assert model.t_center is not None
    sigmas = torch.exp(model.t_log_scale.detach())
    span = model.t_end - model.t_start
    assert float(sigmas.min()) < span  # at least one gaussian localised in time


def test_freeze_canonical_iters_prevents_early_canonical_drift(tmp_path):
    scene = toy_dynamic_scene(n_cameras=5, n_times=6, width=24, height=18, n_points=60, seed=2)
    cfg = _train_cfg(tmp_path, iterations=40, freeze_canonical_iters=40)
    result = train_dynamic(cfg, scene.dataset, scene.init.clone(), iterations=40)
    assert torch.allclose(result.model.means.detach(), scene.init.means, atol=1e-5)


def test_rigidity_regulariser_reduces_deformation_magnitude(tmp_path):
    scene = toy_dynamic_scene(n_cameras=5, n_times=6, width=24, height=18, n_points=60, seed=3)

    cfg_reg = _train_cfg(tmp_path / "reg", iterations=150, rigidity_weight=0.05)
    result_reg = train_dynamic(cfg_reg, scene.dataset, scene.init.clone(), iterations=150)
    with torch.no_grad():
        d_reg = result_reg.model.field(
            result_reg.model.means, result_reg.model.t_norm(scene.times[-1])
        )["dmean"]

    cfg_plain = _train_cfg(tmp_path / "plain", iterations=150, rigidity_weight=0.0)
    result_plain = train_dynamic(cfg_plain, scene.dataset, scene.init.clone(), iterations=150)
    with torch.no_grad():
        d_plain = result_plain.model.field(
            result_plain.model.means, result_plain.model.t_norm(scene.times[-1])
        )["dmean"]

    assert float(d_reg.pow(2).mean()) < float(d_plain.pow(2).mean())
