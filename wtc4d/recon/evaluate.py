"""Metrics, residual images and the per-gaussian coverage proxy.

PSNR and SSIM are implemented in torch (no scikit-image dependency) and accept
a per-pixel weight, because most of our pixels are *not* valid supervision:
smoke, fire, dust and broadcast graphics are masked out, and a metric computed
over them would mostly measure the mask.

``coverage`` counts, for every gaussian, how many frames actually supported it
(the gaussian contributed non-negligible weight to a rendered pixel).  That is
the seed of per-gaussian provenance and uncertainty in the viewer: a facade
seen by eleven broadcasts is a very different claim from one seen by one
helicopter shot through smoke.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from .backends import render as render_backend
from .gaussians import Gaussians

__all__ = [
    "EvalReport",
    "FrameMetrics",
    "coverage",
    "evaluate_dataset",
    "lpips_metric",
    "psnr",
    "save_residual_png",
    "ssim",
]


def _to_chw(img: Tensor) -> Tensor:
    if img.dim() == 3 and img.shape[-1] in (1, 3):
        return img.permute(2, 0, 1)
    return img


def psnr(pred: Tensor, target: Tensor, weight: Tensor | None = None, max_val: float = 1.0) -> float:
    """PSNR in dB over the (optionally weighted) pixels."""
    err = (pred.detach() - target.detach()) ** 2
    if weight is not None:
        w = weight.detach()
        if w.dim() == err.dim() - 1:
            w = w.unsqueeze(-1)
        denom = w.sum() * err.shape[-1]
        if float(denom) <= 0:
            return float("nan")
        mse = (err * w).sum() / denom
    else:
        mse = err.mean()
    mse = float(mse)
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((max_val**2) / mse))


def _gaussian_window(size: int, sigma: float, device, dtype) -> Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return g.outer(g)


def ssim(
    pred: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    window_size: int = 11,
    sigma: float = 1.5,
    max_val: float = 1.0,
) -> Tensor:
    """Mean SSIM of two ``(H, W, C)`` images, differentiable.

    The window shrinks automatically on small images (the toy scenes and
    heavily downscaled SD frames), and ``weight`` averages the SSIM map over
    valid pixels only.
    """
    x = _to_chw(pred).unsqueeze(0)
    y = _to_chw(target).unsqueeze(0)
    c = x.shape[1]
    h, w = x.shape[-2:]
    size = int(min(window_size, h, w))
    if size % 2 == 0:
        size -= 1
    size = max(size, 3)
    win = _gaussian_window(size, sigma * size / window_size, x.device, x.dtype)
    win = win.expand(c, 1, size, size).contiguous()
    pad = size // 2

    def filt(z: Tensor) -> Tensor:
        return F.conv2d(F.pad(z, (pad, pad, pad, pad), mode="reflect"), win, groups=c)

    mu_x, mu_y = filt(x), filt(y)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x = filt(x * x) - mu_x2
    sigma_y = filt(y * y) - mu_y2
    sigma_xy = filt(x * y) - mu_xy
    c1, c2 = (0.01 * max_val) ** 2, (0.03 * max_val) ** 2
    smap = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    )
    if weight is None:
        return smap.mean()
    wt = _to_chw(weight if weight.dim() == 3 else weight.unsqueeze(-1)).unsqueeze(0)
    return (smap * wt).sum() / (wt.sum() * c).clamp_min(1e-8)


def lpips_metric(net: str = "alex"):
    """Return an LPIPS callable, or ``None`` when ``lpips`` is not installed.

    LPIPS is an optional extra: it pulls torchvision weights over the network,
    which we do not want in CI.
    """
    try:
        import lpips as lpips_pkg
    except Exception:  # noqa: BLE001
        return None
    try:
        model = lpips_pkg.LPIPS(net=net)
    except Exception:  # noqa: BLE001 - weight download failed
        return None

    def fn(pred: Tensor, target: Tensor) -> float:
        with torch.no_grad():
            a = _to_chw(pred).unsqueeze(0) * 2 - 1
            b = _to_chw(target).unsqueeze(0) * 2 - 1
            return float(model(a, b).reshape(()))

    return fn


@dataclass
class FrameMetrics:
    index: int
    shot_id: str
    frame_idx: int
    psnr: float
    ssim: float
    lpips: float | None = None
    masked_fraction: float = 0.0
    residual_path: str | None = None


@dataclass
class EvalReport:
    frames: list[FrameMetrics] = field(default_factory=list)
    psnr_mean: float = 0.0
    ssim_mean: float = 0.0
    lpips_mean: float | None = None
    per_shot_psnr: dict[str, float] = field(default_factory=dict)
    n_gaussians: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path


def evaluate_dataset(
    gaussians: Gaussians,
    dataset,
    *,
    appearance=None,
    bg_color=None,
    backend: str | None = None,
    use_masks: bool = True,
    residual_dir: str | Path | None = None,
    with_lpips: bool = False,
    t_from_sample: bool = False,
) -> EvalReport:
    """PSNR / SSIM (/ LPIPS) over every frame of ``dataset``.

    ``appearance`` is an optional callable ``(rgb, sample) -> rgb`` applied to
    the render before comparison -- the per-image exposure/white-balance model
    learned during training (evaluating without it would punish the model for
    a broadcast's colour grade, which is not geometry).
    """
    lpips_fn = lpips_metric() if with_lpips else None
    frames: list[FrameMetrics] = []
    residual_dir = Path(residual_dir) if residual_dir else None
    if residual_dir:
        residual_dir.mkdir(parents=True, exist_ok=True)

    for sample in dataset:
        if sample.image is None:
            continue
        with torch.no_grad():
            out = render_backend(
                gaussians,
                sample.c2w,
                sample.intrinsics,
                sample.width,
                sample.height,
                t=sample.t if t_from_sample else None,
                bg_color=bg_color,
                backend=backend,
            )
            rgb = out.rgb
            if appearance is not None:
                rgb = appearance(rgb, sample)
            rgb = rgb.clamp(0.0, 1.0)
            weight = (
                sample.static_weight if (use_masks and sample.dynamic_mask is not None) else None
            )
            value_psnr = psnr(rgb, sample.image, weight)
            value_ssim = float(ssim(rgb, sample.image, weight))
            value_lpips = lpips_fn(rgb, sample.image) if lpips_fn else None
            path = None
            if residual_dir is not None:
                path = str(
                    save_residual_png(
                        residual_dir / f"{sample.shot_id}_{sample.frame_idx:06d}.png",
                        rgb,
                        sample.image,
                    )
                )
        frames.append(
            FrameMetrics(
                index=sample.index,
                shot_id=sample.shot_id,
                frame_idx=sample.frame_idx,
                psnr=value_psnr,
                ssim=value_ssim,
                lpips=value_lpips,
                masked_fraction=(
                    float(sample.dynamic_mask.mean()) if sample.dynamic_mask is not None else 0.0
                ),
                residual_path=path,
            )
        )

    report = EvalReport(frames=frames, n_gaussians=gaussians.n)
    if frames:
        report.psnr_mean = float(np.mean([f.psnr for f in frames]))
        report.ssim_mean = float(np.mean([f.ssim for f in frames]))
        lp = [f.lpips for f in frames if f.lpips is not None]
        report.lpips_mean = float(np.mean(lp)) if lp else None
        by_shot: dict[str, list[float]] = {}
        for f in frames:
            by_shot.setdefault(f.shot_id, []).append(f.psnr)
        report.per_shot_psnr = {k: float(np.mean(v)) for k, v in by_shot.items()}
    return report


def save_residual_png(path: str | Path, pred: Tensor, target: Tensor, gain: float = 3.0) -> Path:
    """Write a side-by-side ``[prediction | target | |residual| * gain]`` PNG."""
    from PIL import Image

    p = pred.detach().cpu().clamp(0, 1)
    t = target.detach().cpu().clamp(0, 1)
    r = ((p - t).abs() * gain).clamp(0, 1)
    strip = torch.cat([p, t, r], dim=1)
    arr = (strip.numpy() * 255.0).astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    return path


def coverage(
    gaussians: Gaussians,
    dataset,
    *,
    backend: str | None = None,
    weight_threshold: float = 1e-3,
    bg_color=None,
    t_from_sample: bool = False,
) -> dict[str, np.ndarray]:
    """Per-gaussian support across the dataset.

    Returns ``n_frames`` (how many frames the gaussian contributed to),
    ``n_shots`` (how many distinct broadcasts -- the number that matters, since
    50 frames of one shot are one viewpoint) and ``weight`` (total accumulated
    alpha).  A gaussian supported by one shot is geometry asserted by a single
    source and must be presented that way.
    """
    n = gaussians.n
    n_frames = np.zeros(n, dtype=np.int64)
    total_weight = np.zeros(n, dtype=np.float64)
    shots: list[set[str]] = [set() for _ in range(n)]

    for sample in dataset:
        with torch.no_grad():
            out = render_backend(
                gaussians,
                sample.c2w,
                sample.intrinsics,
                sample.width,
                sample.height,
                t=sample.t if t_from_sample else None,
                bg_color=bg_color,
                backend=backend,
            )
        w = out.meta.get("weight_sum")
        if w is None:  # gsplat does not return it; fall back to visibility
            vis = out.meta.get("radii")
            w = (vis > 0).to(torch.float32) if vis is not None else torch.zeros(n)
        w_np = w.detach().cpu().numpy().reshape(-1)
        if w_np.shape[0] != n:  # packed backend layout
            continue
        hit = w_np > weight_threshold
        n_frames += hit.astype(np.int64)
        total_weight += w_np
        for idx in np.nonzero(hit)[0]:
            shots[idx].add(sample.shot_id)

    return {
        "n_frames": n_frames,
        "n_shots": np.array([len(s) for s in shots], dtype=np.int64),
        "weight": total_weight,
    }
