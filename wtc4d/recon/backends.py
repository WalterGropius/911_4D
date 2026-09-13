"""Rasteriser backends: ``cpu`` (reference, always available) and ``gsplat`` (CUDA).

Both expose the same call::

    render(gaussians, c2w, intrinsics, width, height, t=None, bg_color=None) -> RenderOutput

so training code is backend-agnostic and CI exercises the real logic on CPU.
``gsplat`` needs CUDA and compiles kernels on first use, so it is an optional
dependency: it is **not** in the ``recon`` extra and is installed only in the
GPU image (see ``wtc4d/recon/README.md``).  ``resolve()`` picks ``gsplat``
when it imports *and* a CUDA device exists, otherwise the CPU reference.
"""

from __future__ import annotations

import functools
import os

import torch
from torch import Tensor

from wtc4d.schema.camera import CameraIntrinsics

from . import cpu_raster
from .cpu_raster import RenderOutput, intrinsics_to_K
from .gaussians import Gaussians

__all__ = ["RenderOutput", "available_backends", "gsplat_available", "render", "resolve"]


@functools.lru_cache(maxsize=1)
def gsplat_available() -> bool:
    """True when ``gsplat`` imports and a CUDA device is present."""
    if not torch.cuda.is_available():
        return False
    try:
        import gsplat  # noqa: F401
    except Exception:  # noqa: BLE001 - any import/compile failure means "not available"
        return False
    return True


def available_backends() -> list[str]:
    return ["cpu"] + (["gsplat"] if gsplat_available() else [])


def resolve(backend: str | None = None) -> str:
    """Resolve a backend name.

    ``None`` or ``"auto"`` picks ``gsplat`` when usable, else ``"cpu"``.  The
    environment variable ``WTC4D_RECON_BACKEND`` overrides ``auto`` (useful to
    force the reference path on a GPU box when validating the kernels).
    """
    backend = backend or os.environ.get("WTC4D_RECON_BACKEND") or "auto"
    if backend == "auto":
        return "gsplat" if gsplat_available() else "cpu"
    if backend not in ("cpu", "gsplat"):
        raise ValueError(f"unknown backend {backend!r}; expected 'cpu', 'gsplat' or 'auto'")
    if backend == "gsplat" and not gsplat_available():
        raise RuntimeError(
            "gsplat backend requested but unavailable (needs CUDA + `pip install gsplat`)"
        )
    return backend


def render(
    gaussians: Gaussians,
    c2w: Tensor,
    intrinsics: CameraIntrinsics | Tensor,
    width: int,
    height: int,
    *,
    t: float | None = None,
    bg_color=None,
    near: float = 0.05,
    far: float | None = None,
    backend: str | None = None,
    use_sh_dirs: bool = True,
    **kwargs,
) -> RenderOutput:
    """Render one view with the selected backend (see the module docstring)."""
    name = resolve(backend)
    if name == "cpu":
        return cpu_raster.render(
            gaussians,
            c2w,
            intrinsics,
            width,
            height,
            t=t,
            bg_color=bg_color,
            near=near,
            far=far,
            use_sh_dirs=use_sh_dirs,
            **kwargs,
        )
    return _render_gsplat(
        gaussians,
        c2w,
        intrinsics,
        width,
        height,
        t=t,
        bg_color=bg_color,
        near=near,
        far=far,
        use_sh_dirs=use_sh_dirs,
        **kwargs,
    )


def _render_gsplat(
    gaussians: Gaussians,
    c2w: Tensor,
    intrinsics: CameraIntrinsics | Tensor,
    width: int,
    height: int,
    *,
    t: float | None,
    bg_color,
    near: float,
    far: float | None,
    use_sh_dirs: bool,
    **kwargs,
) -> RenderOutput:
    import gsplat

    device = gaussians.means.device
    dtype = gaussians.means.dtype
    c2w = torch.as_tensor(c2w, device=device, dtype=dtype).reshape(4, 4)
    k = intrinsics_to_K(intrinsics, device=device, dtype=dtype)
    viewmats = torch.linalg.inv(c2w).unsqueeze(0)

    if use_sh_dirs and gaussians.sh_rest is not None:
        colors = torch.cat([gaussians.sh_dc.unsqueeze(1), gaussians.sh_rest], dim=1)
        sh_degree = gaussians.sh_degree
    else:
        colors = gaussians.colors(None)
        sh_degree = None

    backgrounds = None
    if bg_color is not None:
        backgrounds = torch.as_tensor(bg_color, device=device, dtype=dtype).reshape(1, 3)

    out, alphas, meta = gsplat.rasterization(
        means=gaussians.means,
        quats=gaussians.unit_quats(),
        scales=gaussians.scales(),
        opacities=gaussians.opacities_at(t),
        colors=colors,
        viewmats=viewmats,
        Ks=k.unsqueeze(0),
        width=width,
        height=height,
        near_plane=near,
        far_plane=far if far is not None else 1e10,
        render_mode="RGB+ED",
        sh_degree=sh_degree,
        backgrounds=backgrounds,
        **kwargs,
    )
    rgb = out[0, ..., :3]
    depth = out[0, ..., 3:4]
    meta = dict(meta)
    meta.setdefault("backend", "gsplat")
    meta.setdefault("width", width)
    meta.setdefault("height", height)
    # gsplat names differ slightly; expose the reference names too so callers
    # (densification, coverage) do not branch on the backend.
    if "means2d" in meta and meta["means2d"].dim() == 3:
        meta["means2d_packed"] = meta["means2d"]
    if "radii" in meta:
        meta["radii"] = meta["radii"].reshape(-1) if meta["radii"].dim() > 1 else meta["radii"]
    return RenderOutput(rgb=rgb, alpha=alphas[0], depth=depth, meta=meta)
