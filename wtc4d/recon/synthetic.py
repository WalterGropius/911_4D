"""Synthetic toy scenes: the CPU stand-in for a registered epoch or window.

Everything here is generated from ``wtc4d.world`` geometry at the real scale
(towers 63 m square, 417 m tall, cameras a kilometre away) so that the
training loop is exercised with the same conditioning the real problem has:
long, narrow view frusta, tiny parallax, and per-camera colour differences.

Two scenes:

``toy_static_scene``
    Box towers plus ground, textured with a deterministic pattern, seen by a
    ring of cameras.  Used to test that :mod:`~wtc4d.recon.train_static`
    actually improves PSNR on CPU in a few hundred steps.

``toy_dynamic_scene``
    The same, plus a "collapse": the upper half of one tower descends and
    spreads over ten timesteps, and a blob (a dust puff) drifts across.  Used
    to test the 4D prototype in :mod:`~wtc4d.recon.dynamic`.

The rendered images are produced by the reference rasteriser, so these tests
check the training logic, not the renderer's physical fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from wtc4d.schema.camera import CameraIntrinsics, CameraPose
from wtc4d.schema.time import TimeEstimate, TimeMethod

from . import cpu_raster
from .conventions import look_at_c2w
from .data import FrameSample
from .gaussians import Gaussians, Layer, rgb_to_sh_dc

__all__ = [
    "InMemoryDataset",
    "toy_cameras",
    "toy_dynamic_scene",
    "toy_static_scene",
]


class InMemoryDataset:
    """A list of :class:`FrameSample` with the slice of the
    :class:`~wtc4d.recon.data.ReconDataset` API that training uses."""

    def __init__(self, samples: list[FrameSample], times: dict | None = None) -> None:
        self.samples = list(samples)
        self.times = dict(times or {})
        self.frames_dir = None
        self.masks_dir = None

    @property
    def poses(self) -> list[CameraPose]:
        return [
            CameraPose.from_matrix(
                s.c2w.detach().cpu().numpy(),
                shot_id=s.shot_id,
                frame_idx=s.frame_idx,
                intrinsics=s.intrinsics,
                method="synthetic",
            )
            for s in self.samples
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, index: int) -> FrameSample:
        return self.samples[index]

    def subset(self, indices) -> InMemoryDataset:
        out = [self.samples[i] for i in indices]
        for j, s in enumerate(out):
            out[j] = FrameSample(**{**s.__dict__, "index": j})
        return InMemoryDataset(out, self.times)

    def split(self, holdout_every: int = 8) -> tuple[InMemoryDataset, InMemoryDataset]:
        if holdout_every <= 1:
            return self, self.subset([])
        train = [i for i in range(len(self)) if i % holdout_every != holdout_every - 1]
        val = [i for i in range(len(self)) if i % holdout_every == holdout_every - 1]
        return self.subset(train), self.subset(val)

    def shots(self) -> list[str]:
        return sorted({s.shot_id for s in self.samples})

    def time_range(self) -> tuple[float, float] | None:
        ts = [s.t for s in self.samples if s.t is not None]
        return (min(ts), max(ts)) if ts else None


@dataclass
class ToyScene:
    """A toy scene and the ground truth it was rendered from."""

    dataset: InMemoryDataset
    truth: Gaussians
    init: Gaussians
    times: list[float]
    center: np.ndarray


def toy_cameras(
    n: int = 6,
    *,
    width: int = 48,
    height: int = 36,
    radius_m: float = 1400.0,
    height_m: float = 260.0,
    target=None,
    hfov_deg: float = 40.0,
    jitter_m: float = 0.0,
    seed: int = 0,
) -> list[tuple[np.ndarray, CameraIntrinsics]]:
    """A ring of cameras around the WTC site, roughly the real geometry.

    Real vantage points (New Jersey, Brooklyn, Midtown, helicopters) are
    1-6 km out and near or above the impact floors; the ring is the same idea
    at a scale where the CPU rasteriser is fast.
    """
    target = np.array([0.0, 0.0, 200.0]) if target is None else np.asarray(target, dtype=np.float64)
    rng = np.random.default_rng(seed)
    fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    out = []
    for i in range(n):
        ang = 2.0 * np.pi * i / n
        eye = np.array(
            [radius_m * np.cos(ang), radius_m * np.sin(ang), height_m + 80.0 * np.sin(2.3 * ang)]
        )
        if jitter_m:
            eye = eye + rng.normal(scale=jitter_m, size=3)
        intr = CameraIntrinsics(
            width=width,
            height=height,
            fx=float(fx),
            fy=float(fx),
            cx=width / 2.0,
            cy=height / 2.0,
            model="PINHOLE",
        )
        out.append((look_at_c2w(eye, target), intr))
    return out


def _textured_scene_gaussians(n_points: int, seed: int, device: str | torch.device) -> Gaussians:
    """Box towers + ground, sampled and given a deterministic texture."""
    import trimesh

    from .init import estimate_spacing, prior_boxes_scene

    scene = prior_boxes_scene()
    rng = np.random.default_rng(seed)
    xyz_parts, layer_parts = [], []
    meshes = {str(k): v for k, v in scene.geometry.items()}
    total = sum(m.area for m in meshes.values())
    for name, mesh in meshes.items():
        count = max(24, int(round(n_points * mesh.area / total)))
        pts, _ = trimesh.sample.sample_surface(mesh, count, seed=int(rng.integers(1 << 30)))
        xyz_parts.append(np.asarray(pts, dtype=np.float64))
        layer_parts.append(
            np.full(
                len(pts), int(Layer.GROUND if "ground" in name else Layer.TOWERS), dtype=np.int64
            )
        )
    xyz = np.concatenate(xyz_parts)
    layers = np.concatenate(layer_parts)

    # A deterministic texture so images carry real signal (a flat grey scene
    # would let a degenerate solution reach high PSNR).
    phase = xyz / np.array([37.0, 41.0, 23.0])
    rgb = 0.5 + 0.28 * np.stack(
        [np.sin(phase[:, 0]), np.sin(phase[:, 1] + 1.7), np.sin(phase[:, 2] + 3.1)], axis=-1
    )
    scales = estimate_spacing(xyz) * 1.4
    g = Gaussians.from_points(
        xyz, np.clip(rgb, 0.0, 1.0), scale_m=scales, opacity=0.92, layer=layers, device=device
    )
    return g


def toy_static_scene(
    *,
    n_cameras: int = 6,
    width: int = 48,
    height: int = 36,
    n_points: int = 220,
    seed: int = 0,
    device: str | torch.device = "cpu",
    appearance_jitter: float = 0.0,
    with_masks: bool = False,
    t: float | None = None,
) -> ToyScene:
    """Render a textured toy scene from a ring of cameras.

    ``appearance_jitter`` multiplies each camera's image by a per-camera gain
    and adds a bias, imitating the exposure/white-balance spread between 2001
    broadcasts; the appearance embedding in training has to absorb it.
    ``with_masks`` marks the lower-left quadrant of each image as "dynamic",
    standing in for a smoke mask.
    """
    truth = _textured_scene_gaussians(n_points, seed, device)
    cams = toy_cameras(n_cameras, width=width, height=height, seed=seed)
    rng = np.random.default_rng(seed + 1)
    samples: list[FrameSample] = []
    times: dict[tuple[str, int], TimeEstimate] = {}
    bg = torch.tensor([0.55, 0.68, 0.85], device=device)  # a September sky

    for i, (c2w, intr) in enumerate(cams):
        c2w_t = torch.as_tensor(c2w, device=device, dtype=torch.float32)
        with torch.no_grad():
            out = cpu_raster.render(truth, c2w_t, intr, intr.width, intr.height, bg_color=bg)
        img = out.rgb
        if appearance_jitter:
            gain = 1.0 + appearance_jitter * float(rng.normal())
            bias = appearance_jitter * 0.2 * float(rng.normal())
            img = (img * gain + bias).clamp(0.0, 1.0)
        mask = None
        if with_masks:
            mask = torch.zeros(intr.height, intr.width, 1, device=device)
            mask[intr.height // 2 :, : intr.width // 2] = 1.0
        shot_id = f"toy_cam{i:02d}"
        if t is not None:
            times[(shot_id, 0)] = TimeEstimate(t=t, sigma=0.5, method=TimeMethod.MANUAL)
        samples.append(
            FrameSample(
                index=i,
                shot_id=shot_id,
                frame_idx=0,
                c2w=c2w_t,
                intrinsics=intr,
                width=intr.width,
                height=intr.height,
                image=img,
                dynamic_mask=mask,
                t=t,
                t_sigma=0.5 if t is not None else None,
            )
        )

    init = _perturbed_init(truth, seed=seed + 2)
    return ToyScene(
        dataset=InMemoryDataset(samples, times),
        truth=truth,
        init=init,
        times=[t] if t is not None else [],
        center=np.zeros(3),
    )


def _perturbed_init(truth: Gaussians, seed: int) -> Gaussians:
    """A deliberately wrong starting point: jittered, grey, half transparent."""
    g = truth.detach().clone()
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn(g.means.shape, generator=gen) * 3.0
    g.means = g.means + noise.to(g.means.device)
    g.sh_dc = rgb_to_sh_dc(torch.full_like(g.sh_dc, 0.5))
    g.logit_opacities = torch.full_like(g.logit_opacities, 0.0)
    g.log_scales = g.log_scales + 0.2
    return g


def toy_dynamic_scene(
    *,
    n_cameras: int = 6,
    n_times: int = 10,
    width: int = 40,
    height: int = 30,
    n_points: int = 140,
    seed: int = 0,
    device: str | torch.device = "cpu",
    t0: float = 0.0,
    dt: float = 1.0,
) -> ToyScene:
    """A toy "collapse": a box whose top half descends, plus a drifting blob.

    Returns ``n_cameras * n_times`` frames, each carrying its own ``t`` (project
    seconds).  ``truth`` is the canonical (``t = t0``) state; the motion is not
    recoverable from it alone, which is the point -- the deformation field has
    to learn it.
    """
    canonical = _textured_scene_gaussians(n_points, seed, device)
    # Keep the scene small and centred so the toy trains fast.
    keep = canonical.means[:, 2] > -5.0
    canonical = canonical.select(keep)

    blob_center0 = torch.tensor([120.0, -60.0, 320.0], device=device)
    blob = Gaussians.from_points(
        (blob_center0.cpu().numpy() + np.random.default_rng(seed).normal(scale=18.0, size=(12, 3))),
        np.tile([0.85, 0.83, 0.8], (12, 1)),
        scale_m=22.0,
        opacity=0.6,
        layer=int(Layer.SMOKE),
        device=device,
    )
    canonical = Gaussians.cat([canonical, blob])
    n_blob = len(blob)

    cams = toy_cameras(n_cameras, width=width, height=height, seed=seed)
    bg = torch.tensor([0.55, 0.68, 0.85], device=device)
    samples: list[FrameSample] = []
    times: dict[tuple[str, int], TimeEstimate] = {}
    ts = [t0 + i * dt for i in range(n_times)]

    for ti, t in enumerate(ts):
        frac = (t - t0) / max(dt * (n_times - 1), 1e-6)
        state = _deform_truth(canonical, frac, n_blob)
        for ci, (c2w, intr) in enumerate(cams):
            c2w_t = torch.as_tensor(c2w, device=device, dtype=torch.float32)
            with torch.no_grad():
                out = cpu_raster.render(state, c2w_t, intr, intr.width, intr.height, bg_color=bg)
            shot_id = f"toy_cam{ci:02d}"
            times[(shot_id, ti)] = TimeEstimate(t=t, sigma=0.5, method=TimeMethod.EVENT_ANCHOR)
            samples.append(
                FrameSample(
                    index=len(samples),
                    shot_id=shot_id,
                    frame_idx=ti,
                    c2w=c2w_t,
                    intrinsics=intr,
                    width=intr.width,
                    height=intr.height,
                    image=out.rgb,
                    t=t,
                    t_sigma=0.5,
                )
            )

    # The init is the canonical state with the geometry slightly wrong: the
    # deformation field has to learn the motion *and* the optimiser has to fix
    # the canonical set, which is the real situation after a static epoch
    # splat is handed to a 4D window.
    init = canonical.detach().clone()
    gen = torch.Generator().manual_seed(seed + 7)
    init.means = init.means + (torch.randn(init.means.shape, generator=gen) * 2.5).to(device)
    init.sh_dc = init.sh_dc * 0.7
    return ToyScene(
        dataset=InMemoryDataset(samples, times),
        truth=canonical,
        init=init,
        times=ts,
        center=np.zeros(3),
    )


def _deform_truth(g: Gaussians, frac: float, n_blob: int) -> Gaussians:
    """Ground-truth motion: upper material falls and spreads; the blob drifts."""
    out = g.detach().clone()
    means = out.means.clone()
    static = means[:, 2:3]
    # Everything above 200 m falls, faster the higher it starts (a crude
    # progressive collapse), and spreads outward as it goes.
    above = (static > 200.0).squeeze(-1)
    drop = (static.squeeze(-1) - 200.0).clamp_min(0.0) * frac * 0.85
    means[above, 2] = means[above, 2] - drop[above]
    spread = 1.0 + 0.35 * frac
    means[above, 0] = means[above, 0] * spread
    means[above, 1] = means[above, 1] * spread
    if n_blob:
        means[-n_blob:, 0] = means[-n_blob:, 0] + 90.0 * frac
        means[-n_blob:, 2] = means[-n_blob:, 2] - 40.0 * frac
    out.means = means
    if out.t_center is None and n_blob:
        # the blob fades in over the window
        opac = out.logit_opacities.clone()
        opac[-n_blob:] = opac[-n_blob:] + 2.0 * (frac - 0.5)
        out.logit_opacities = opac
    return out


def toy_frames_as_tensors(scene: ToyScene) -> tuple[Tensor, Tensor]:
    """Stack a toy scene's images and c2w matrices (handy in assertions)."""
    imgs = torch.stack([s.image for s in scene.dataset])
    c2ws = torch.stack([s.c2w for s in scene.dataset])
    return imgs, c2ws
