"""True-4D prototype: canonical gaussians + a time-conditioned deformation field.

The design survey, with the trade-offs for *this* dataset, is in
``docs/recon_4d_design.md``.  Summary of the choice implemented here:

* **Deformation field** (4DGS, Wu et al. 2024): one canonical gaussian set
  plus ``F(x, t) -> (dx, dscale, drot, dopacity)``.  Compact, temporally
  smooth by construction, and it degrades gracefully when a camera's time is
  off by a second -- which ours are (``sync`` gives sigma ~0.5-1 s).
* **Temporal opacity** (Spacetime Gaussians, Li et al. 2024): each gaussian
  may carry ``t_center``/``t_log_scale``, so material that only exists for a
  moment (a dust puff, a falling panel) can appear and vanish instead of
  being dragged through the whole window by the deformation field.

Both are in this module because they are complementary: the field carries
coherent motion, the temporal window carries birth and death.  What is *not*
here (per-timestep independent sets) is discussed in the design doc; with
10-30 SD cameras per window it overfits immediately.

The prototype is deliberately small: no densification, no appearance MLP by
default, one deformation MLP.  It exists to validate the interfaces and the
CPU test loop before GPU time is spent.
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

from .backends import render as render_backend
from .backends import resolve as resolve_backend
from .evaluate import psnr, ssim
from .gaussians import Gaussians
from .train_static import AppearanceConfig, AppearanceModel, _appearance_keys

__all__ = [
    "DeformationField",
    "DynamicGaussians",
    "DynamicTrainConfig",
    "TimeEncoding",
    "train_dynamic",
]


class TimeEncoding(nn.Module):
    """Sinusoidal encoding of a scalar, ``[t, sin(2^k pi t), cos(2^k pi t)]``."""

    def __init__(self, n_freq: int = 4) -> None:
        super().__init__()
        self.n_freq = n_freq
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq).float() * math.pi)

    @property
    def dim(self) -> int:
        return 1 + 2 * self.n_freq

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape(-1, 1) if x.dim() <= 1 else x
        scaled = x * self.freqs.reshape(1, -1)
        return torch.cat([x, torch.sin(scaled), torch.cos(scaled)], dim=-1)


class PositionEncoding(nn.Module):
    """Sinusoidal encoding of a 3D position, normalised by the scene radius."""

    def __init__(self, n_freq: int = 4, scale: float = 500.0) -> None:
        super().__init__()
        self.n_freq = n_freq
        self.scale = scale
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq).float() * math.pi)

    @property
    def dim(self) -> int:
        return 3 + 6 * self.n_freq

    def forward(self, x: Tensor) -> Tensor:
        xn = x / self.scale
        scaled = xn.unsqueeze(-1) * self.freqs.reshape(1, 1, -1)
        return torch.cat([xn, torch.sin(scaled).flatten(-2), torch.cos(scaled).flatten(-2)], dim=-1)


class DeformationConfig(BaseModel):
    pos_freqs: int = 4
    time_freqs: int = 4
    hidden: int = 64
    n_layers: int = 3
    predict_scale: bool = True
    predict_rotation: bool = True
    predict_opacity: bool = True
    max_translation_m: float = 400.0  # a tower's worth of fall
    max_log_scale_delta: float = 1.0
    scene_scale_m: float = 500.0


class DeformationField(nn.Module):
    """``F(x_canonical, t) -> (dmean, dlog_scale, dquat, dlogit_opacity)``.

    Outputs are bounded (``tanh`` on the translation) so that a bad gradient
    early in training cannot fling gaussians across Lower Manhattan, and the
    last layer is zero-initialised so training starts from the static solution
    -- which is the right prior here: most of the scene *is* static even inside
    a collapse window.
    """

    def __init__(self, cfg: DeformationConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.pos_enc = PositionEncoding(cfg.pos_freqs, cfg.scene_scale_m)
        self.time_enc = TimeEncoding(cfg.time_freqs)
        in_dim = self.pos_enc.dim + self.time_enc.dim
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(max(cfg.n_layers - 1, 1)):
            layers += [nn.Linear(dim, cfg.hidden), nn.ReLU()]
            dim = cfg.hidden
        self.trunk = nn.Sequential(*layers)
        self.head_mean = nn.Linear(dim, 3)
        self.head_scale = nn.Linear(dim, 3) if cfg.predict_scale else None
        self.head_rot = nn.Linear(dim, 4) if cfg.predict_rotation else None
        self.head_opacity = nn.Linear(dim, 1) if cfg.predict_opacity else None
        for head in (self.head_mean, self.head_scale, self.head_rot, self.head_opacity):
            if head is not None:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def forward(self, means: Tensor, t_norm: float | Tensor) -> dict[str, Tensor]:
        n = means.shape[0]
        tt = torch.as_tensor(t_norm, device=means.device, dtype=means.dtype).reshape(1, 1)
        feat = torch.cat([self.pos_enc(means), self.time_enc(tt).expand(n, -1)], dim=-1)
        h = self.trunk(feat)
        out = {"dmean": torch.tanh(self.head_mean(h)) * self.cfg.max_translation_m}
        if self.head_scale is not None:
            out["dlog_scale"] = torch.tanh(self.head_scale(h)) * self.cfg.max_log_scale_delta
        if self.head_rot is not None:
            out["dquat"] = self.head_rot(h)
        if self.head_opacity is not None:
            out["dlogit_opacity"] = self.head_opacity(h).squeeze(-1)
        return out


class DynamicGaussians(nn.Module):
    """Canonical gaussians + deformation field + optional temporal opacity.

    ``render(t, camera)`` matches the static backend signature so the viewer
    export, the evaluation code and the CPU tests are shared.
    """

    def __init__(
        self,
        canonical: Gaussians,
        t_start: float,
        t_end: float,
        cfg: DeformationConfig | None = None,
        learn_time_window: bool = False,
    ) -> None:
        super().__init__()
        self.t_start = float(t_start)
        self.t_end = float(t_end)
        self.cfg = cfg or DeformationConfig()
        self.field = DeformationField(self.cfg)

        g = canonical.detach().clone()
        self.means = nn.Parameter(g.means)
        self.log_scales = nn.Parameter(g.log_scales)
        self.quats = nn.Parameter(g.unit_quats())
        self.sh_dc = nn.Parameter(g.sh_dc)
        self.sh_rest = nn.Parameter(g.sh_rest) if g.sh_rest is not None else None
        self.logit_opacities = nn.Parameter(g.logit_opacities)
        self.register_buffer("layer", g.layer)
        if learn_time_window:
            n = g.n
            centre = 0.5 * (self.t_start + self.t_end)
            span = max(self.t_end - self.t_start, 1.0)
            self.t_center = nn.Parameter(
                g.t_center if g.t_center is not None else torch.full((n,), centre)
            )
            self.t_log_scale = nn.Parameter(
                g.t_log_scale if g.t_log_scale is not None else torch.full((n,), math.log(span))
            )
        else:
            self.t_center = None
            self.t_log_scale = None

    # ------------------------------------------------------------------ time
    def t_norm(self, t: float) -> float:
        span = max(self.t_end - self.t_start, 1e-6)
        return 2.0 * (float(t) - self.t_start) / span - 1.0

    def canonical_gaussians(self) -> Gaussians:
        return Gaussians(
            means=self.means,
            log_scales=self.log_scales,
            quats=self.quats,
            sh_dc=self.sh_dc,
            sh_rest=self.sh_rest,
            logit_opacities=self.logit_opacities,
            layer=self.layer,
            t_center=self.t_center,
            t_log_scale=self.t_log_scale,
        )

    def at(self, t: float) -> Gaussians:
        """The deformed gaussian set at project time ``t``."""
        d = self.field(self.means, self.t_norm(t))
        quats = self.quats
        if "dquat" in d:
            quats = quats + d["dquat"]
        log_scales = self.log_scales
        if "dlog_scale" in d:
            log_scales = log_scales + d["dlog_scale"]
        logit_op = self.logit_opacities
        dop = d.get("dlogit_opacity")
        if isinstance(dop, Tensor):
            logit_op = logit_op + dop
        return Gaussians(
            means=self.means + d["dmean"],
            log_scales=log_scales,
            quats=quats,
            sh_dc=self.sh_dc,
            sh_rest=self.sh_rest,
            logit_opacities=logit_op,
            layer=self.layer,
            t_center=self.t_center,
            t_log_scale=self.t_log_scale,
        )

    def render(self, t: float, c2w, intrinsics, width: int, height: int, **kwargs):
        """Render the scene at time ``t`` (same signature as the static backends)."""
        use_time = self.t_center is not None
        return render_backend(
            self.at(t), c2w, intrinsics, width, height, t=t if use_time else None, **kwargs
        )

    def export_timestep(self, t: float) -> Gaussians:
        """A plain static :class:`Gaussians` snapshot for the viewer / ``.ply``."""
        with torch.no_grad():
            return self.at(t).detach()


class DynamicTrainConfig(BaseModel):
    """Configuration for the 4D prototype."""

    window_id: str | None = None
    t_start: float | None = None
    t_end: float | None = None
    device: str = "cpu"
    backend: str | None = None
    iterations: int = 2_000
    lr_field: float = 1e-3
    lr_means: float = 1e-3
    lr_log_scales: float = 2e-3
    lr_quats: float = 1e-3
    lr_sh_dc: float = 2.5e-3
    lr_opacity: float = 2e-2
    lr_time_window: float = 1e-2
    l1_weight: float = 0.8
    ssim_weight: float = 0.2
    time_smooth_weight: float = 0.0
    rigidity_weight: float = 0.0
    learn_time_window: bool = False
    freeze_canonical_iters: int = 0
    background_color: tuple[float, float, float] = (0.55, 0.68, 0.85)
    appearance: AppearanceConfig = Field(default_factory=lambda: AppearanceConfig(mode="none"))
    deformation: DeformationConfig = Field(default_factory=DeformationConfig)
    out_dir: str = "runs/recon"
    name: str = "dynamic"
    log_interval: int = 50
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> DynamicTrainConfig:
        import yaml

        return cls.model_validate(yaml.safe_load(Path(path).read_text()) or {})


@dataclass
class DynamicResult:
    model: DynamicGaussians
    stats: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    stats_path: Path | None = None


def train_dynamic(
    cfg: DynamicTrainConfig,
    dataset,
    canonical: Gaussians,
    val_dataset=None,
    iterations: int | None = None,
) -> DynamicResult:
    """Fit a canonical set + deformation field to a time-stamped dataset.

    ``dataset`` yields :class:`~wtc4d.recon.data.FrameSample` objects carrying
    ``t`` (project seconds).  Frames without ``t`` are skipped: a frame we
    cannot place in time is worse than no frame at all in a 4D window.
    """
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    backend = resolve_backend(cfg.backend)

    ts = [s.t for s in dataset if s.t is not None]
    if not ts:
        raise ValueError("dynamic training needs frames with absolute times (wtc4d.sync)")
    t_start = cfg.t_start if cfg.t_start is not None else min(ts)
    t_end = cfg.t_end if cfg.t_end is not None else max(ts)

    model = DynamicGaussians(
        canonical.to(device=device),
        t_start,
        t_end,
        cfg.deformation,
        learn_time_window=cfg.learn_time_window,
    ).to(device)

    appearance = AppearanceModel(cfg.appearance, _appearance_keys(dataset, per_shot=True)).to(
        device
    )
    bg = torch.tensor(cfg.background_color, device=device)

    groups = [
        {"params": list(model.field.parameters()), "lr": cfg.lr_field},
        {"params": [model.means], "lr": cfg.lr_means},
        {"params": [model.log_scales], "lr": cfg.lr_log_scales},
        {"params": [model.quats], "lr": cfg.lr_quats},
        {"params": [model.sh_dc], "lr": cfg.lr_sh_dc},
        {"params": [model.logit_opacities], "lr": cfg.lr_opacity},
    ]
    if model.sh_rest is not None:
        groups.append({"params": [model.sh_rest], "lr": cfg.lr_sh_dc * 0.05})
    if model.t_center is not None:
        groups.append({"params": [model.t_center, model.t_log_scale], "lr": cfg.lr_time_window})
    if cfg.appearance.mode != "none":
        groups.append({"params": list(appearance.parameters()), "lr": cfg.appearance.lr})
    optimizer = torch.optim.Adam(groups, eps=1e-15)

    frames = [s for s in dataset if s.t is not None and s.image is not None]
    rng = np.random.default_rng(cfg.seed)
    history: list[dict] = []
    total = iterations if iterations is not None else cfg.iterations
    started = time.time()

    for it in range(1, total + 1):
        sample = frames[int(rng.integers(len(frames)))]
        out = model.render(
            sample.t,
            sample.c2w,
            sample.intrinsics,
            sample.width,
            sample.height,
            bg_color=bg,
            backend=backend,
        )
        rgb = appearance(out.rgb, appearance.key_for(sample, per_shot=True))
        gt = sample.image
        weight = sample.static_weight if sample.dynamic_mask is not None else None
        if weight is None:
            l1 = (rgb - gt).abs().mean()
        else:
            l1 = ((rgb - gt).abs() * weight).sum() / (weight.sum() * 3).clamp_min(1e-6)
        loss = cfg.l1_weight * l1
        if cfg.ssim_weight > 0:
            loss = loss + cfg.ssim_weight * (1.0 - ssim(rgb.clamp(0, 1), gt, weight))
        if cfg.time_smooth_weight > 0:
            loss = loss + cfg.time_smooth_weight * _time_smoothness(model, sample.t)
        if cfg.rigidity_weight > 0:
            d = model.field(model.means, model.t_norm(sample.t))["dmean"]
            loss = loss + cfg.rigidity_weight * d.pow(2).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if it <= cfg.freeze_canonical_iters:
            for p in (
                model.means,
                model.log_scales,
                model.quats,
                model.sh_dc,
                model.logit_opacities,
            ):
                p.grad = None
        optimizer.step()

        if cfg.log_interval and (it % cfg.log_interval == 0 or it == 1):
            with torch.no_grad():
                history.append(
                    {
                        "iter": it,
                        "loss": float(loss.detach()),
                        "psnr": psnr(rgb.clamp(0, 1), gt, weight),
                        "t": sample.t,
                    }
                )

    stats = {
        "name": cfg.name,
        "window_id": cfg.window_id,
        "backend": backend,
        "iterations": total,
        "n_frames": len(frames),
        "n_gaussians": model.means.shape[0],
        "t_start": t_start,
        "t_end": t_end,
        "wall_seconds": round(time.time() - started, 2),
        "history": history[-200:],
        "config": json.loads(cfg.model_dump_json()),
    }
    if val_dataset is not None and len(val_dataset):
        stats["val"] = _evaluate_dynamic(model, val_dataset, appearance, bg, backend)

    out_dir = Path(cfg.out_dir) / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / f"{cfg.name}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    return DynamicResult(model=model, stats=stats, history=history, stats_path=stats_path)


def _time_smoothness(model: DynamicGaussians, t: float, dt: float = 0.25) -> Tensor:
    """Penalise acceleration of the deformation: a finite-difference prior.

    With ~0.5-1 s of timing uncertainty per camera, motion that is not smooth
    in time is almost certainly the field absorbing a sync error.
    """
    span = max(model.t_end - model.t_start, 1e-6)
    step = dt / span * 2.0
    tn = model.t_norm(t)
    d0 = model.field(model.means, tn - step)["dmean"]
    d1 = model.field(model.means, tn)["dmean"]
    d2 = model.field(model.means, tn + step)["dmean"]
    return (d2 - 2 * d1 + d0).pow(2).mean()


def _evaluate_dynamic(model, dataset, appearance, bg, backend) -> dict:
    vals = []
    with torch.no_grad():
        for sample in dataset:
            if sample.t is None or sample.image is None:
                continue
            out = model.render(
                sample.t,
                sample.c2w,
                sample.intrinsics,
                sample.width,
                sample.height,
                bg_color=bg,
                backend=backend,
            )
            rgb = appearance(out.rgb, appearance.key_for(sample, per_shot=True)).clamp(0, 1)
            vals.append(psnr(rgb, sample.image))
    return {"psnr_mean": float(np.mean(vals)) if vals else None, "n": len(vals)}
