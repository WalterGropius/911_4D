"""Static (per-epoch) gaussian splat training.

What makes this problem different from a normal 3DGS capture, and what this
module does about it:

*Cameras are already metric.*  ``camreg`` registers frames against the city
prior, so poses are **fixed by default**; only a small, bounded refinement is
optional (``PoseRefineConfig``).  Nothing here is allowed to rescale or
re-centre the world -- the world frame is the shared contract.

*Every broadcast is graded differently.*  A 2001 network feed, a VHS
off-air recording and a DV camcorder disagree wildly on exposure and white
balance, and a single radiance field cannot satisfy them all.  Each image
therefore gets an appearance transform (gain/bias per channel, or a tiny MLP)
that is applied to the render before the loss, exactly as in NeRF-W and
"VastGaussian"-style decoupled appearance modelling.

*Most pixels are dynamic.*  Smoke, fire, dust and station graphics are masked
out; masked pixels do not supervise the static layers at all.  Optionally the
masked pixels supervise a separate, time-tagged "smoke" gaussian set
(``LossConfig.mask_mode="supervise_smoke"``).

*There is a strong prior.*  The init comes from the city mesh, so an anchor
loss (distance from a gaussian to where the prior put it) and an optional
depth loss against the prior's own depth render keep facades on the building
instead of floating in front of it.  The anchor weight decays: the prior is a
starting point, not ground truth.

*The sky is a huge, textureless part of every frame.*  It gets a learned
background colour rather than a cloud of sky gaussians.

Run it with ``wtc4d recon train-static --config cfg.yaml``.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, Field
from torch import Tensor

from wtc4d import timeline

from .backends import render as render_backend
from .backends import resolve as resolve_backend
from .evaluate import EvalReport, evaluate_dataset, psnr, ssim
from .gaussians import Gaussians, Layer

__all__ = [
    "AppearanceConfig",
    "AppearanceModel",
    "DataConfig",
    "DensifyConfig",
    "InitConfig",
    "LossConfig",
    "OptimConfig",
    "OutputConfig",
    "PoseRefineConfig",
    "StaticTrainer",
    "TrainStaticConfig",
    "train_static",
]


# --------------------------------------------------------------------- config
class DataConfig(BaseModel):
    poses_path: str | None = None
    frames_dir: str | None = None
    masks_dir: str | None = None
    times_path: str | None = None
    downscale: float = 1.0
    mask_is_static: bool = False
    epoch_id: str | None = None
    t_start: float | None = None
    t_end: float | None = None
    holdout_every: int = 8
    keep_untimed: bool = True


class InitConfig(BaseModel):
    mesh_points: int = 100_000
    procedural_t: float | None = None
    procedural_ply: str | None = None
    colmap_path: str | None = None
    ply_path: str | None = Field(default=None, description="resume from an existing splat")
    dedup_radius: float = 0.5
    sh_degree: int = 0
    opacity: float = 0.1
    seed: int = 0


class LossConfig(BaseModel):
    l1_weight: float = 0.8
    ssim_weight: float = 0.2
    mask_mode: str = "ignore"  # ignore | supervise_smoke | none
    opacity_reg: float = 0.0
    scale_reg: float = 1e-3
    max_scale_m: float = 40.0
    anisotropy_max: float = 10.0
    anchor_weight: float = 0.0
    anchor_decay_iters: int = 5_000
    depth_weight: float = 0.0
    depth_from_prior: bool = False
    smoke_time_sigma_s: float = 8.0


class AppearanceConfig(BaseModel):
    mode: str = "affine"  # none | affine | mlp
    lr: float = 1e-3
    embed_dim: int = 4
    hidden: int = 16
    per_shot: bool = False  # share one transform across all frames of a shot


class PoseRefineConfig(BaseModel):
    enable: bool = False
    lr_rotation: float = 1e-5
    lr_translation: float = 1e-3
    refine_focal: bool = False
    lr_focal: float = 1e-4
    max_translation_m: float = 25.0
    max_rotation_deg: float = 1.0


class DensifyConfig(BaseModel):
    enable: bool = True
    start_iter: int = 500
    stop_iter: int = 15_000
    interval: int = 100
    grad_threshold: float = 2e-4
    percent_dense: float = 0.01
    min_opacity: float = 0.005
    max_gaussians: int = 3_000_000
    reset_opacity_interval: int = 3_000
    reset_opacity_value: float = 0.01
    prune_big_scale_m: float | None = 80.0


class OptimConfig(BaseModel):
    iterations: int = 30_000
    lr_means: float = 1.6e-4  # multiplied by the scene extent, as in 3DGS
    lr_means_final_factor: float = 0.01
    lr_log_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_sh_dc: float = 2.5e-3
    lr_sh_rest_factor: float = 0.05
    lr_opacity: float = 5e-2
    lr_background: float = 2.5e-3
    random_order: bool = True
    seed: int = 0


class OutputConfig(BaseModel):
    out_dir: str = "runs/recon"
    name: str = "static"
    ckpt_interval: int = 5_000
    eval_interval: int = 2_000
    log_interval: int = 100
    export_ply: bool = True
    residuals: bool = False
    with_lpips: bool = False
    coverage: bool = True


class TrainStaticConfig(BaseModel):
    """Full configuration for one static training run (YAML-friendly)."""

    epoch_id: str | None = None
    device: str = "cpu"
    backend: str | None = None  # None/auto -> gsplat when available
    background: str = "learned"  # learned | none | fixed
    background_color: tuple[float, float, float] = (0.55, 0.68, 0.85)
    data: DataConfig = Field(default_factory=DataConfig)
    init: InitConfig = Field(default_factory=InitConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    appearance: AppearanceConfig = Field(default_factory=AppearanceConfig)
    pose_refine: PoseRefineConfig = Field(default_factory=PoseRefineConfig)
    densify: DensifyConfig = Field(default_factory=DensifyConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainStaticConfig:
        import yaml

        return cls.model_validate(yaml.safe_load(Path(path).read_text()) or {})

    def to_yaml(self, path: str | Path) -> Path:
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(json.loads(self.model_dump_json()), sort_keys=False))
        return path


# ---------------------------------------------------------------- sub-models
class AppearanceModel(nn.Module):
    """Per-image (or per-shot) photometric transform applied to the render.

    ``affine``: ``rgb * exp(gain) + bias`` per channel -- exposure and white
    balance, three parameters each, initialised to the identity.
    ``mlp``: a two-layer network on ``[rgb, embedding]``, which can also absorb
    a broadcast's gamma/knee.  ``none`` is the identity.
    """

    def __init__(self, cfg: AppearanceConfig, keys: list[str]) -> None:
        super().__init__()
        self.mode = cfg.mode
        self.keys = list(keys)
        self.index = {k: i for i, k in enumerate(self.keys)}
        n = max(len(self.keys), 1)
        if self.mode == "affine":
            self.gain = nn.Parameter(torch.zeros(n, 3))
            self.bias = nn.Parameter(torch.zeros(n, 3))
        elif self.mode == "mlp":
            self.embed = nn.Parameter(torch.zeros(n, cfg.embed_dim))
            self.net = nn.Sequential(
                nn.Linear(3 + cfg.embed_dim, cfg.hidden),
                nn.ReLU(),
                nn.Linear(cfg.hidden, 3),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        elif self.mode != "none":
            raise ValueError(f"unknown appearance mode {self.mode!r}")

    def key_for(self, sample, per_shot: bool) -> str:
        return sample.shot_id if per_shot else f"{sample.shot_id}:{sample.frame_idx}"

    def forward(self, rgb: Tensor, key: str) -> Tensor:
        """Apply the transform for ``key``.

        An **unknown** key (a held-out frame whose shot was never trained on)
        falls back to the mean of the learned transforms -- "the average
        broadcast grade" -- rather than to an arbitrary image's.  Evaluation on
        such frames therefore measures geometry plus a generic colour
        mismatch; hold out frames, not whole shots, when that matters.
        """
        if self.mode == "none":
            return rgb
        i = self.index.get(key)
        if self.mode == "affine":
            gain = self.gain[i] if i is not None else self.gain.mean(dim=0)
            bias = self.bias[i] if i is not None else self.bias.mean(dim=0)
            return rgb * torch.exp(gain) + bias
        embed = self.embed[i] if i is not None else self.embed.mean(dim=0)
        e = embed.expand(*rgb.shape[:-1], self.embed.shape[1])
        return rgb + self.net(torch.cat([rgb, e], dim=-1))


class PoseRefiner(nn.Module):
    """Small per-image pose (and optional focal) correction.

    Parameterised as a **camera-frame** perturbation ``c2w' = c2w @ exp(xi)``
    with ``xi`` a bounded se(3) vector, so the refinement cannot drift the
    scene: the caps in :class:`PoseRefineConfig` are hard (``tanh``), defaulting
    to 25 m and 1 degree -- the order of ``camreg``'s own sigmas.
    """

    def __init__(self, cfg: PoseRefineConfig, n_images: int) -> None:
        super().__init__()
        self.cfg = cfg
        n = max(n_images, 1)
        self.delta_rot = nn.Parameter(torch.zeros(n, 3))
        self.delta_trans = nn.Parameter(torch.zeros(n, 3))
        self.log_focal = nn.Parameter(torch.zeros(n))

    def forward(self, c2w: Tensor, index: int) -> Tensor:
        cfg = self.cfg
        rot = torch.tanh(self.delta_rot[index]) * math.radians(cfg.max_rotation_deg)
        trans = torch.tanh(self.delta_trans[index]) * cfg.max_translation_m
        delta = torch.eye(4, device=c2w.device, dtype=c2w.dtype)
        delta = torch.cat(
            [
                torch.cat([_so3_exp(rot), trans.reshape(3, 1)], dim=1),
                delta[3:4],
            ],
            dim=0,
        )
        return c2w @ delta

    def focal_scale(self, index: int) -> Tensor:
        if not self.cfg.refine_focal:
            return torch.ones((), device=self.log_focal.device, dtype=self.log_focal.dtype)
        return torch.exp(torch.tanh(self.log_focal[index]) * 0.1)  # +-10 %


def _so3_exp(w: Tensor) -> Tensor:
    """Rodrigues: axis-angle (3,) -> rotation (3, 3), differentiable at 0."""
    theta = w.norm().clamp_min(1e-12)
    k = torch.zeros(3, 3, device=w.device, dtype=w.dtype)
    kx = torch.stack(
        [
            torch.stack([k[0, 0], -w[2], w[1]]),
            torch.stack([w[2], k[0, 0], -w[0]]),
            torch.stack([-w[1], w[0], k[0, 0]]),
        ]
    )
    eye = torch.eye(3, device=w.device, dtype=w.dtype)
    a = torch.sin(theta) / theta
    b = (1 - torch.cos(theta)) / (theta * theta)
    return eye + a * kx + b * (kx @ kx)


# --------------------------------------------------------------------- result
@dataclass
class TrainResult:
    gaussians: Gaussians
    stats: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    report: EvalReport | None = None
    ply_path: Path | None = None
    npz_path: Path | None = None
    stats_path: Path | None = None


# -------------------------------------------------------------------- trainer
class StaticTrainer:
    """The training loop, usable step-by-step (tests) or via :meth:`run`."""

    def __init__(
        self,
        cfg: TrainStaticConfig,
        dataset,
        gaussians: Gaussians | None = None,
        val_dataset=None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.backend = resolve_backend(cfg.backend)
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.gaussians = (gaussians if gaussians is not None else self._build_init()).to(
            device=self.device
        )
        self._as_parameters()

        self.scene_extent = _scene_extent(self.gaussians.means.detach())
        self.anchor = self.gaussians.means.detach().clone()
        self.anchor_valid = torch.ones(self.gaussians.n, dtype=torch.bool, device=self.device)

        keys = _appearance_keys(dataset, cfg.appearance.per_shot)
        self.appearance = AppearanceModel(cfg.appearance, keys).to(self.device)
        self.pose_refiner = (
            PoseRefiner(cfg.pose_refine, len(dataset)).to(self.device)
            if cfg.pose_refine.enable
            else None
        )
        bg = torch.tensor(cfg.background_color, dtype=torch.float32, device=self.device)
        self.background = (
            nn.Parameter(torch.logit(bg.clamp(1e-3, 1 - 1e-3)))
            if cfg.background == "learned"
            else bg
        )

        self.optimizer = self._build_optimizer()
        self.aux_optimizer = self._build_aux_optimizer()
        self.iteration = 0
        self.history: list[dict] = []
        self._xy_grad = torch.zeros(self.gaussians.n, device=self.device)
        self._denom = torch.zeros(self.gaussians.n, device=self.device)
        self._rng = np.random.default_rng(cfg.optim.seed)
        self._order: list[int] = []

    # ------------------------------------------------------------ setup bits
    def _build_init(self) -> Gaussians:
        from .init import init_gaussians

        ic = self.cfg.init
        if ic.ply_path:
            return Gaussians.load_ply(ic.ply_path, device=self.cfg.device)
        return init_gaussians(
            mesh_points=ic.mesh_points,
            epoch_id=self.cfg.epoch_id or self.cfg.data.epoch_id,
            procedural_t=ic.procedural_t,
            procedural_ply=ic.procedural_ply,
            colmap_path=ic.colmap_path,
            dedup_radius=ic.dedup_radius,
            sh_degree=ic.sh_degree,
            seed=ic.seed,
            device=self.cfg.device,
        )

    def _as_parameters(self) -> None:
        g = self.gaussians
        for name in ("means", "log_scales", "quats", "sh_dc", "sh_rest", "logit_opacities"):
            t = getattr(g, name)
            if t is not None:
                setattr(g, name, nn.Parameter(t.detach().clone()))

    def _param_groups(self) -> list[dict]:
        g, o = self.gaussians, self.cfg.optim
        groups = [
            {"params": [g.means], "lr": o.lr_means * self.scene_extent, "name": "means"},
            {"params": [g.log_scales], "lr": o.lr_log_scales, "name": "log_scales"},
            {"params": [g.quats], "lr": o.lr_quats, "name": "quats"},
            {"params": [g.sh_dc], "lr": o.lr_sh_dc, "name": "sh_dc"},
            {"params": [g.logit_opacities], "lr": o.lr_opacity, "name": "logit_opacities"},
        ]
        if g.sh_rest is not None:
            groups.append(
                {
                    "params": [g.sh_rest],
                    "lr": o.lr_sh_dc * o.lr_sh_rest_factor,
                    "name": "sh_rest",
                }
            )
        return groups

    def _build_optimizer(self) -> torch.optim.Adam:
        return torch.optim.Adam(self._param_groups(), eps=1e-15)

    def _build_aux_optimizer(self) -> torch.optim.Adam | None:
        groups: list[dict] = []
        if self.cfg.appearance.mode != "none":
            groups.append(
                {"params": list(self.appearance.parameters()), "lr": self.cfg.appearance.lr}
            )
        if self.pose_refiner is not None:
            pr = self.cfg.pose_refine
            groups.append({"params": [self.pose_refiner.delta_rot], "lr": pr.lr_rotation})
            groups.append({"params": [self.pose_refiner.delta_trans], "lr": pr.lr_translation})
            if pr.refine_focal:
                groups.append({"params": [self.pose_refiner.log_focal], "lr": pr.lr_focal})
        if isinstance(self.background, nn.Parameter):
            groups.append({"params": [self.background], "lr": self.cfg.optim.lr_background})
        groups = [g for g in groups if g["params"]]
        return torch.optim.Adam(groups) if groups else None

    # --------------------------------------------------------------- helpers
    @property
    def bg_color(self) -> Tensor | None:
        if self.cfg.background == "none":
            return None
        if isinstance(self.background, nn.Parameter):
            return torch.sigmoid(self.background)
        return self.background

    def camera_for(self, sample):
        """``(c2w, intrinsics)`` for a sample, with pose refinement applied."""
        from .cpu_raster import intrinsics_to_K

        c2w = sample.c2w
        k = intrinsics_to_K(sample.intrinsics, device=self.device, dtype=torch.float32)
        if self.pose_refiner is not None:
            c2w = self.pose_refiner(c2w, sample.index)
            scale = self.pose_refiner.focal_scale(sample.index)
            k = k.clone()
            k = torch.stack(
                [
                    torch.stack([k[0, 0] * scale, k[0, 1], k[0, 2]]),
                    torch.stack([k[1, 0], k[1, 1] * scale, k[1, 2]]),
                    k[2],
                ]
            )
        return c2w, k

    def render_sample(self, sample, gaussians: Gaussians | None = None, t: float | None = None):
        c2w, k = self.camera_for(sample)
        return render_backend(
            gaussians if gaussians is not None else self.gaussians,
            c2w,
            k,
            sample.width,
            sample.height,
            t=t,
            bg_color=self.bg_color,
            backend=self.backend,
        )

    def _lr_means_at(self, it: int) -> float:
        o = self.cfg.optim
        frac = min(max(it / max(o.iterations, 1), 0.0), 1.0)
        return o.lr_means * self.scene_extent * (o.lr_means_final_factor**frac)

    def _next_index(self) -> int:
        if not self.cfg.optim.random_order:
            return self.iteration % len(self.dataset)
        if not self._order:
            self._order = list(self._rng.permutation(len(self.dataset)))
        return int(self._order.pop())

    # ------------------------------------------------------------------ step
    def step(self) -> dict:
        cfg = self.cfg
        sample = self.dataset[self._next_index()]
        if sample.image is None:
            self.iteration += 1
            return {"iter": self.iteration, "skipped": True}

        for group in self.optimizer.param_groups:
            if group.get("name") == "means":
                group["lr"] = self._lr_means_at(self.iteration)

        out = self.render_sample(sample)
        means2d = out.meta.get("means2d")
        if isinstance(means2d, Tensor) and means2d.requires_grad:
            means2d.retain_grad()

        key = self.appearance.key_for(sample, cfg.appearance.per_shot)
        rgb = self.appearance(out.rgb, key)
        rgb_dynamic = None
        if self._use_smoke_supervision(sample):
            out_dyn = self.render_sample(sample, gaussians=self._smoke_render_set(), t=sample.t)
            rgb_dynamic = self.appearance(out_dyn.rgb, key)
        loss, parts = self._loss(rgb, out, sample, rgb_dynamic)

        self.optimizer.zero_grad(set_to_none=True)
        if self.aux_optimizer is not None:
            self.aux_optimizer.zero_grad(set_to_none=True)
        loss.backward()

        self._accumulate_densify_stats(out)
        self.optimizer.step()
        if self.aux_optimizer is not None:
            self.aux_optimizer.step()

        self.iteration += 1
        if cfg.densify.enable:
            self._maybe_densify()

        parts["iter"] = self.iteration
        parts["loss"] = float(loss.detach())
        parts["n_gaussians"] = self.gaussians.n
        return parts

    def _dynamic_mask_layers(self) -> Tensor:
        return self.gaussians.layer_mask(Layer.SMOKE, Layer.DEBRIS)

    def _use_smoke_supervision(self, sample) -> bool:
        """True when masked pixels should train a separate smoke layer."""
        if self.cfg.loss.mask_mode != "supervise_smoke" or sample.dynamic_mask is None:
            return False
        return bool(self._dynamic_mask_layers().any())

    def _smoke_render_set(self) -> Gaussians:
        """Smoke/debris gaussians (live) in front of the static scene (detached).

        Detaching the static part is what keeps the promise that masked pixels
        never supervise static geometry: they still occlude correctly, but no
        gradient reaches the facades through them.
        """
        dyn = self._dynamic_mask_layers()
        static_part = self.gaussians.select(~dyn).detach()
        smoke_part = self.gaussians.select(dyn)
        return Gaussians.cat([static_part, smoke_part])

    def _loss(
        self, rgb: Tensor, out, sample, rgb_dynamic: Tensor | None = None
    ) -> tuple[Tensor, dict]:
        cfg = self.cfg.loss
        gt = sample.image
        rgb_c = rgb.clamp(0.0, 1.0)
        weight = None
        if cfg.mask_mode != "none" and sample.dynamic_mask is not None:
            weight = sample.static_weight
        if weight is not None and float(weight.sum()) < 1.0:
            weight = None

        if weight is None:
            l1 = (rgb - gt).abs().mean()
        else:
            l1 = ((rgb - gt).abs() * weight).sum() / (weight.sum() * 3).clamp_min(1e-6)
        loss = cfg.l1_weight * l1
        parts = {"l1": float(l1.detach())}

        if cfg.ssim_weight > 0:
            s = ssim(rgb_c, gt, weight)
            loss = loss + cfg.ssim_weight * (1.0 - s)
            parts["ssim"] = float(s.detach())

        if rgb_dynamic is not None:
            dyn_w = sample.dynamic_mask
            if float(dyn_w.sum()) > 0:
                l1_dyn = ((rgb_dynamic - gt).abs() * dyn_w).sum() / (dyn_w.sum() * 3).clamp_min(
                    1e-6
                )
                loss = loss + cfg.l1_weight * l1_dyn
                parts["l1_dynamic"] = float(l1_dyn.detach())

        if cfg.opacity_reg > 0:
            loss = loss + cfg.opacity_reg * self.gaussians.opacities().mean()
        if cfg.scale_reg > 0:
            scales = self.gaussians.scales()
            big = torch.relu(scales.max(dim=-1).values - cfg.max_scale_m).mean()
            ratio = scales.max(dim=-1).values / scales.min(dim=-1).values.clamp_min(1e-6)
            aniso = torch.relu(ratio - cfg.anisotropy_max).mean()
            loss = loss + cfg.scale_reg * (big + aniso)
        if cfg.anchor_weight > 0 and bool(self.anchor_valid.any()):
            decay = max(0.0, 1.0 - self.iteration / max(cfg.anchor_decay_iters, 1))
            if decay > 0:
                d = self.gaussians.means[self.anchor_valid] - self.anchor[self.anchor_valid]
                loss = loss + cfg.anchor_weight * decay * d.pow(2).sum(-1).mean()
        if cfg.depth_weight > 0 and getattr(sample, "prior_depth", None) is not None:
            valid = (out.alpha.detach() > 0.5) & (sample.prior_depth > 0)
            if weight is not None:
                valid = valid & (weight > 0.5)
            if bool(valid.any()):
                d = (out.depth - sample.prior_depth).abs()[valid].mean()
                loss = loss + cfg.depth_weight * d
                parts["depth_l1"] = float(d.detach())

        with torch.no_grad():
            parts["psnr"] = psnr(rgb_c, gt, weight)
        return loss, parts

    # ---------------------------------------------------------- densification
    def _accumulate_densify_stats(self, out) -> None:
        means2d = out.meta.get("means2d")
        if not isinstance(means2d, Tensor) or means2d.grad is None:
            return
        grad = means2d.grad.detach()
        if grad.shape[0] != self.gaussians.n:
            return
        visible = out.meta.get("visible")
        norm = grad.norm(dim=-1)
        if visible is not None and visible.shape[0] == self.gaussians.n:
            norm = norm * visible.to(norm.dtype)
            self._denom += visible.to(self._denom.dtype)
        else:
            self._denom += 1.0
        self._xy_grad += norm

    def _maybe_densify(self) -> None:
        d = self.cfg.densify
        it = self.iteration
        if it < d.start_iter or it > d.stop_iter:
            return
        if d.reset_opacity_interval and it % d.reset_opacity_interval == 0:
            self._reset_opacity()
            return
        if it % d.interval != 0:
            return
        self._densify_and_prune()

    @torch.no_grad()
    def _reset_opacity(self) -> None:
        value = math.log(
            self.cfg.densify.reset_opacity_value / (1 - self.cfg.densify.reset_opacity_value)
        )
        self.gaussians.logit_opacities.clamp_(max=value)

    @torch.no_grad()
    def _densify_and_prune(self) -> None:
        cfg = self.cfg.densify
        g = self.gaussians
        n = g.n
        grads = self._xy_grad / self._denom.clamp_min(1.0)
        scales = g.scales()
        max_scale = scales.max(dim=-1).values
        size_threshold = cfg.percent_dense * self.scene_extent

        selected = grads > cfg.grad_threshold
        room = max(cfg.max_gaussians - n, 0)
        clone_mask = selected & (max_scale <= size_threshold)
        split_mask = selected & (max_scale > size_threshold)
        if room == 0:
            clone_mask = torch.zeros_like(clone_mask)
            split_mask = torch.zeros_like(split_mask)
        elif int(clone_mask.sum() + split_mask.sum()) > room:
            # keep the highest-gradient candidates only
            cand = torch.nonzero(clone_mask | split_mask).reshape(-1)
            keep = cand[torch.argsort(grads[cand], descending=True)[:room]]
            mask = torch.zeros(n, dtype=torch.bool, device=g.means.device)
            mask[keep] = True
            clone_mask = clone_mask & mask
            split_mask = split_mask & mask

        keep_mask = g.opacities() > cfg.min_opacity
        if cfg.prune_big_scale_m is not None:
            keep_mask = keep_mask & (max_scale < cfg.prune_big_scale_m)
        if not bool(keep_mask.any()):
            keep_mask = torch.ones(n, dtype=torch.bool, device=g.means.device)

        src_parts = [torch.nonzero(keep_mask).reshape(-1)]
        new_parts: list[dict] = []

        if bool(clone_mask.any()):
            idx = torch.nonzero(clone_mask & keep_mask).reshape(-1)
            if idx.numel():
                jitter = torch.randn(idx.numel(), 3, device=g.means.device) * scales[idx] * 0.5
                new_parts.append({"src": idx, "means": g.means[idx] + jitter, "scale_div": 1.0})
        if bool(split_mask.any()):
            idx = torch.nonzero(split_mask & keep_mask).reshape(-1)
            if idx.numel():
                for _ in range(2):
                    offset = torch.randn(idx.numel(), 3, device=g.means.device) * scales[idx]
                    new_parts.append({"src": idx, "means": g.means[idx] + offset, "scale_div": 1.6})

        if not new_parts and bool(keep_mask.all()):
            self._xy_grad.zero_()
            self._denom.zero_()
            return

        def gather(name: str) -> Tensor | None:
            base = getattr(g, name)
            if base is None:
                return None
            chunks = [base[src_parts[0]]]
            for part in new_parts:
                v = base[part["src"]]
                if name == "log_scales" and part["scale_div"] != 1.0:
                    v = v - math.log(part["scale_div"])
                if name == "means":
                    v = part["means"]
                chunks.append(v)
            return torch.cat(chunks, dim=0)

        index_map = torch.cat(
            [src_parts[0]] + [torch.full_like(p["src"], -1) for p in new_parts], dim=0
        )
        new_g = Gaussians(
            means=nn.Parameter(gather("means").detach().clone()),
            log_scales=nn.Parameter(gather("log_scales").detach().clone()),
            quats=nn.Parameter(gather("quats").detach().clone()),
            sh_dc=nn.Parameter(gather("sh_dc").detach().clone()),
            sh_rest=None if g.sh_rest is None else nn.Parameter(gather("sh_rest").detach().clone()),
            logit_opacities=nn.Parameter(gather("logit_opacities").detach().clone()),
            layer=gather("layer"),
            t_center=gather("t_center"),
            t_log_scale=gather("t_log_scale"),
        )
        self._transfer_optimizer_state(new_g, index_map)
        self.gaussians = new_g
        anchor = torch.zeros(new_g.n, 3, device=self.device)
        valid = index_map >= 0
        anchor[valid] = self.anchor[index_map[valid]]
        self.anchor = anchor
        self.anchor_valid = valid & self.anchor_valid[index_map.clamp_min(0)]
        self._xy_grad = torch.zeros(new_g.n, device=self.device)
        self._denom = torch.zeros(new_g.n, device=self.device)

    def _transfer_optimizer_state(self, new_g: Gaussians, index_map: Tensor) -> None:
        """Rebuild Adam, carrying moments for surviving gaussians (zeros for new).

        Dropping the moments entirely (the easy option) makes every
        densification step a small shock to the optimiser; carrying them keeps
        the loss curve smooth across the 100-iteration densification cadence.
        """
        old_state = {}
        for group in self.optimizer.param_groups:
            p = group["params"][0]
            st = self.optimizer.state.get(p)
            if st is not None and "exp_avg" in st:
                old_state[group["name"]] = (st["exp_avg"], st["exp_avg_sq"], st.get("step"))
        self.gaussians = new_g
        self.optimizer = self._build_optimizer()
        valid = index_map >= 0
        src = index_map.clamp_min(0)
        for group in self.optimizer.param_groups:
            name = group["name"]
            if name not in old_state:
                continue
            exp_avg, exp_avg_sq, step = old_state[name]
            p = group["params"][0]
            new_avg = torch.zeros_like(p)
            new_avg_sq = torch.zeros_like(p)
            new_avg[valid] = exp_avg[src][valid]
            new_avg_sq[valid] = exp_avg_sq[src][valid]
            self.optimizer.state[p] = {
                "step": step if step is not None else torch.tensor(0.0),
                "exp_avg": new_avg,
                "exp_avg_sq": new_avg_sq,
            }

    # ------------------------------------------------------------------- run
    def run(self, iterations: int | None = None) -> TrainResult:
        cfg = self.cfg
        total = iterations if iterations is not None else cfg.optim.iterations
        out_dir = Path(cfg.output.out_dir) / cfg.output.name
        out_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()

        for _ in range(total):
            stats = self.step()
            if cfg.output.log_interval and self.iteration % cfg.output.log_interval == 0:
                self.history.append(stats)
            if cfg.output.ckpt_interval and self.iteration % cfg.output.ckpt_interval == 0:
                self.save(out_dir, tag=f"{self.iteration:06d}")

        result = self.finalize(out_dir)
        result.stats["wall_seconds"] = round(time.time() - started, 2)
        if result.stats_path:
            result.stats_path.write_text(json.dumps(result.stats, indent=2) + "\n")
        return result

    def finalize(self, out_dir: str | Path) -> TrainResult:
        cfg = self.cfg
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        g = self.gaussians.detach()

        report = None
        eval_ds = (
            self.val_dataset if self.val_dataset is not None and len(self.val_dataset) else None
        )
        if eval_ds is not None:
            report = evaluate_dataset(
                g,
                eval_ds,
                appearance=self._appearance_fn(),
                bg_color=self.bg_color.detach() if self.bg_color is not None else None,
                backend=self.backend,
                residual_dir=(out_dir / "residuals") if cfg.output.residuals else None,
                with_lpips=cfg.output.with_lpips,
            )

        npz_path = g.save_npz(out_dir / f"{cfg.output.name}.npz")
        ply_path = g.save_ply(out_dir / f"{cfg.output.name}.ply") if cfg.output.export_ply else None

        stats = {
            "name": cfg.output.name,
            "epoch_id": cfg.epoch_id,
            "backend": self.backend,
            "iterations": self.iteration,
            "n_gaussians": g.n,
            "n_train_frames": len(self.dataset),
            "n_val_frames": len(eval_ds) if eval_ds is not None else 0,
            "scene_extent_m": self.scene_extent,
            "background": (
                [round(float(x), 5) for x in self.bg_color.detach().cpu()]
                if self.bg_color is not None
                else None
            ),
            "psnr_mean": report.psnr_mean if report else None,
            "ssim_mean": report.ssim_mean if report else None,
            "lpips_mean": report.lpips_mean if report else None,
            "per_shot_psnr": report.per_shot_psnr if report else {},
            "history": self.history[-200:],
            "config": json.loads(cfg.model_dump_json()),
        }
        if cfg.output.coverage and eval_ds is not None:
            from .evaluate import coverage as coverage_fn

            cov = coverage_fn(
                g,
                self.dataset,
                backend=self.backend,
                bg_color=self.bg_color.detach() if self.bg_color is not None else None,
            )
            np.savez_compressed(out_dir / f"{cfg.output.name}_coverage.npz", **cov)
            stats["coverage"] = {
                "mean_frames_per_gaussian": float(cov["n_frames"].mean()),
                "mean_shots_per_gaussian": float(cov["n_shots"].mean()),
                "fraction_single_shot": float((cov["n_shots"] <= 1).mean()),
            }
        stats_path = out_dir / f"{cfg.output.name}_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2) + "\n")
        if report is not None:
            report.save(out_dir / f"{cfg.output.name}_eval.json")

        return TrainResult(
            gaussians=g,
            stats=stats,
            history=self.history,
            report=report,
            ply_path=ply_path,
            npz_path=npz_path,
            stats_path=stats_path,
        )

    def _appearance_fn(self):
        per_shot = self.cfg.appearance.per_shot
        if self.cfg.appearance.mode == "none":
            return None

        def fn(rgb: Tensor, sample) -> Tensor:
            return self.appearance(rgb, self.appearance.key_for(sample, per_shot))

        return fn

    def save(self, out_dir: str | Path, tag: str = "latest") -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.cfg.output.name}_{tag}.npz"
        self.gaussians.detach().save_npz(path)
        return path


# ------------------------------------------------------------------ entrypoint
def _scene_extent(means: Tensor) -> float:
    """Radius of the scene, used to scale the position learning rate."""
    if means.numel() == 0:
        return 1.0
    centre = means.mean(dim=0)
    return float((means - centre).norm(dim=-1).max().clamp_min(1.0))


def _appearance_keys(dataset, per_shot: bool) -> list[str]:
    keys = []
    for sample in dataset:
        key = sample.shot_id if per_shot else f"{sample.shot_id}:{sample.frame_idx}"
        if key not in keys:
            keys.append(key)
    return keys


def build_dataset(cfg: TrainStaticConfig):
    """Build the train/val datasets described by ``cfg.data``."""
    from .data import ReconDataset

    dc = cfg.data
    if not dc.poses_path:
        raise ValueError("data.poses_path is required (or pass a dataset to train_static)")
    ds = ReconDataset.from_paths(
        dc.poses_path,
        dc.frames_dir,
        dc.masks_dir,
        dc.times_path,
        downscale=dc.downscale,
        device=cfg.device,
        mask_is_static=dc.mask_is_static,
    )
    epoch_id = cfg.epoch_id or dc.epoch_id
    if dc.t_start is not None and dc.t_end is not None:
        ds = ds.filter_time(dc.t_start, dc.t_end, keep_untimed=dc.keep_untimed)
    elif epoch_id:
        ep = timeline.EPOCHS_BY_ID[epoch_id]
        ds = ds.filter_time(ep.t_start, ep.t_end, keep_untimed=dc.keep_untimed)
    return ds.split(dc.holdout_every)


def attach_prior_depth(dataset, prior: Gaussians, *, backend: str | None = None, bg_color=None):
    """Render the frozen prior's depth into every sample as ``prior_depth``.

    This is what ``LossConfig.depth_weight`` supervises against: it pins the
    splat to the prior's surface where the prior is trustworthy (facades of
    surveyed buildings) without forbidding it to move.
    """
    for sample in dataset:
        with torch.no_grad():
            out = render_backend(
                prior,
                sample.c2w,
                sample.intrinsics,
                sample.width,
                sample.height,
                bg_color=bg_color,
                backend=backend,
            )
            sample.prior_depth = torch.where(
                out.alpha > 0.5, out.depth, torch.zeros_like(out.depth)
            )
    return dataset


def train_static(
    cfg: TrainStaticConfig,
    dataset=None,
    val_dataset=None,
    gaussians: Gaussians | None = None,
    iterations: int | None = None,
) -> TrainResult:
    """Train one static splat.  ``dataset`` overrides ``cfg.data`` (tests, jobs)."""
    torch.manual_seed(cfg.optim.seed)
    if dataset is None:
        dataset, val_dataset = build_dataset(cfg)
    trainer = StaticTrainer(cfg, dataset, gaussians=gaussians, val_dataset=val_dataset)
    if cfg.loss.depth_weight > 0 and cfg.loss.depth_from_prior:
        attach_prior_depth(
            dataset,
            trainer.gaussians.detach(),
            backend=trainer.backend,
            bg_color=trainer.bg_color,
        )
    return trainer.run(iterations)


def smoke_layer_from(
    g: Gaussians, t_center: float, t_sigma: float = 8.0, layer: Layer = Layer.SMOKE
) -> Gaussians:
    """Tag a gaussian set as a time-windowed smoke layer centred on ``t_center``."""
    out = g.detach().clone()
    n = out.n
    out.layer = torch.full((n,), int(layer), dtype=torch.int64, device=out.means.device)
    out.t_center = torch.full((n,), float(t_center), device=out.means.device)
    out.t_log_scale = torch.full((n,), math.log(max(t_sigma, 1e-3)), device=out.means.device)
    return out
