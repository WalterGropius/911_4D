"""Static training: the CPU toy scene must actually improve, and the pieces
around the core loop (appearance, masks, densification, pose refinement,
config round trip) must do what they claim."""

from __future__ import annotations

import pytest

from wtc4d.recon.synthetic import toy_static_scene
from wtc4d.recon.train_static import (
    DensifyConfig,
    OptimConfig,
    OutputConfig,
    TrainStaticConfig,
    train_static,
)


def _cfg(tmp_path, **overrides) -> TrainStaticConfig:
    cfg = TrainStaticConfig(
        device="cpu",
        backend="cpu",
        optim=OptimConfig(iterations=250, seed=0),
        densify=DensifyConfig(enable=False),
        output=OutputConfig(out_dir=str(tmp_path), name="run", ckpt_interval=0, export_ply=True),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_psnr_improves_on_toy_static_scene(tmp_path):
    scene = toy_static_scene(n_cameras=6, width=32, height=24, n_points=150, seed=0)
    train_ds, val_ds = scene.dataset.split(holdout_every=6)
    cfg = _cfg(tmp_path, optim=OptimConfig(iterations=300, seed=0))
    result = train_static(cfg, dataset=train_ds, val_dataset=val_ds, gaussians=scene.init)

    assert len(result.history) > 0
    first_psnr = result.history[0]["psnr"]
    last_psnr = result.history[-1]["psnr"]
    assert last_psnr > first_psnr + 3.0
    assert result.stats["psnr_mean"] is not None
    assert result.ply_path is not None and result.ply_path.exists()
    assert result.npz_path.exists()
    assert result.stats_path.exists()


def test_appearance_model_absorbs_exposure_jitter(tmp_path):
    """Without an appearance model, per-camera exposure jitter caps achievable
    PSNR on the *training* images (there is only one radiance field to fit six
    different exposures to); with one, each image gets its own gain/bias and
    the ceiling should be substantially higher.  (Held-out shots are a
    separate concern -- an unseen broadcast's grade cannot be recovered from
    its image alone; that is exercised by ``test_pose_refinement_...`` style
    tests operating on training data instead.)"""
    scene = toy_static_scene(
        n_cameras=6, width=28, height=20, n_points=120, seed=1, appearance_jitter=0.35
    )
    train_ds, _ = scene.dataset.split(holdout_every=1)  # no holdout: evaluate on training images

    cfg_on = _cfg(tmp_path / "on", optim=OptimConfig(iterations=350, seed=1))
    cfg_on.appearance.mode = "affine"
    result_on = train_static(
        cfg_on, dataset=train_ds, val_dataset=train_ds, gaussians=scene.init.clone()
    )

    cfg_off = _cfg(tmp_path / "off", optim=OptimConfig(iterations=350, seed=1))
    cfg_off.appearance.mode = "none"
    result_off = train_static(
        cfg_off, dataset=train_ds, val_dataset=train_ds, gaussians=scene.init.clone()
    )

    assert result_on.stats["psnr_mean"] > result_off.stats["psnr_mean"] + 1.0


def test_masked_pixels_do_not_supervise_static_layers(tmp_path):
    scene = toy_static_scene(
        n_cameras=6, width=28, height=20, n_points=120, seed=2, with_masks=True
    )
    train_ds, val_ds = scene.dataset.split(holdout_every=6)
    cfg = _cfg(tmp_path, optim=OptimConfig(iterations=200, seed=2))
    cfg.loss.mask_mode = "ignore"
    result = train_static(cfg, dataset=train_ds, val_dataset=val_ds, gaussians=scene.init)

    # every training sample carries a mask; the reported PSNR must be computed
    # only over unmasked pixels (evaluate_dataset defaults to use_masks=True)
    assert result.report is not None
    assert all(f.masked_fraction > 0 for f in result.report.frames)
    assert result.stats["psnr_mean"] > 15.0


def test_anchor_loss_pulls_means_back_toward_init(tmp_path):
    scene = toy_static_scene(n_cameras=4, width=24, height=18, n_points=80, seed=3)
    train_ds, val_ds = scene.dataset.split(holdout_every=4)
    init_means = scene.init.means.clone()

    cfg = _cfg(tmp_path, optim=OptimConfig(iterations=150, seed=3))
    cfg.loss.anchor_weight = 5.0
    cfg.loss.anchor_decay_iters = 100_000  # effectively constant over this short run
    result = train_static(cfg, dataset=train_ds, val_dataset=val_ds, gaussians=scene.init.clone())
    drift_with_anchor = (result.gaussians.means - init_means).norm(dim=-1).mean()

    cfg2 = _cfg(tmp_path / "noanchor", optim=OptimConfig(iterations=150, seed=3))
    cfg2.loss.anchor_weight = 0.0
    result2 = train_static(cfg2, dataset=train_ds, val_dataset=val_ds, gaussians=scene.init.clone())
    drift_without_anchor = (result2.gaussians.means - init_means).norm(dim=-1).mean()

    assert float(drift_with_anchor) < float(drift_without_anchor)


def test_densification_grows_the_gaussian_count(tmp_path):
    scene = toy_static_scene(n_cameras=6, width=28, height=20, n_points=60, seed=4)
    train_ds, val_ds = scene.dataset.split(holdout_every=6)
    cfg = _cfg(tmp_path, optim=OptimConfig(iterations=400, seed=4))
    cfg.densify = DensifyConfig(
        enable=True, start_iter=20, stop_iter=380, interval=40, grad_threshold=1e-6
    )
    n_before = scene.init.n
    result = train_static(cfg, dataset=train_ds, val_dataset=val_ds, gaussians=scene.init.clone())
    assert (
        result.gaussians.n >= n_before
    )  # threshold near zero: everything qualifies to clone/split
    assert result.gaussians.n > n_before


def test_pose_refinement_stays_within_configured_bounds(tmp_path):
    scene = toy_static_scene(n_cameras=5, width=24, height=18, n_points=70, seed=5)
    train_ds, val_ds = scene.dataset.split(holdout_every=5)
    cfg = _cfg(tmp_path, optim=OptimConfig(iterations=150, seed=5))
    cfg.pose_refine.enable = True
    cfg.pose_refine.max_translation_m = 10.0
    cfg.pose_refine.max_rotation_deg = 2.0

    from wtc4d.recon.train_static import StaticTrainer

    trainer = StaticTrainer(cfg, train_ds, gaussians=scene.init, val_dataset=val_ds)
    for _ in range(150):
        trainer.step()

    for sample in train_ds:
        refined, _ = trainer.camera_for(sample)
        delta_t = (refined[:3, 3] - sample.c2w[:3, 3]).norm().detach()
        assert float(delta_t) <= cfg.pose_refine.max_translation_m + 1e-3


def test_config_yaml_round_trip(tmp_path):
    cfg = _cfg(tmp_path)
    path = cfg.to_yaml(tmp_path / "cfg.yaml")
    back = TrainStaticConfig.from_yaml(path)
    assert back.optim.iterations == cfg.optim.iterations
    assert back.output.name == cfg.output.name
    assert back.device == cfg.device


def test_train_static_requires_poses_path_without_explicit_dataset():
    cfg = TrainStaticConfig()
    with pytest.raises(ValueError, match="poses_path"):
        train_static(cfg)
