"""Pure-torch differentiable reference rasteriser (EWA splatting).

This is the CPU twin of the ``gsplat`` CUDA backend: same maths, same
interface, no kernels.  It exists so the training logic, the losses, the
densification schedule and the 4D deformation field can all be unit-tested
on a laptop or in CI, and so that a numeric discrepancy in the GPU path can
be localised against a readable implementation.

Method (Zwicker et al. 2001, as used by Kerbl et al. 2023):

1. transform gaussian centres to camera space with ``w2c``;
2. project to pixels with the pinhole intrinsics;
3. push the 3D covariance through the local affine approximation of the
   projection, ``Sigma_2d = J W Sigma W^T J^T``, and add a low-pass filter of
   ``low_pass`` px^2 on the diagonal so a gaussian is never thinner than a pixel;
4. sort front-to-back by camera-space depth (globally -- the CUDA backend
   sorts per 16x16 tile, which differs only where splats interleave in depth);
5. alpha-composite ``alpha_i = o_i * exp(-1/2 d^T Sigma_2d^-1 d)``.

Limitations (deliberate -- this is a reference, not a production renderer):
distortion is ignored (undistort the frames instead), the global depth sort
is an approximation, and cost is ``O(N * pixels)`` so it is meant for a few
thousand gaussians at low resolution.  Pixels are processed in chunks to
bound peak memory.

Conventions: ``c2w`` is camera-to-world, OpenCV axes, world ENU metres; the
image is row-major ``(H, W, C)`` with pixel centres at integer coordinates
offset by 0.5 (the COLMAP/OpenCV convention: pixel ``(0, 0)`` spans
``[0, 1) x [0, 1)`` and has centre ``(0.5, 0.5)``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from wtc4d.schema.camera import CameraIntrinsics

from .gaussians import Gaussians, quat_to_rotmat_torch

__all__ = ["RenderOutput", "intrinsics_to_K", "render"]

_ALPHA_EPS = 1.0 / 255.0


@dataclass
class RenderOutput:
    """What every backend returns.

    ``rgb`` (H, W, 3), ``alpha`` (H, W, 1), ``depth`` (H, W, 1) expected depth
    in metres (0 where nothing was hit).  ``meta`` carries the bookkeeping the
    densification schedule and the coverage statistics need:

    ``means2d``
        (N, 2) projected centres in pixels, differentiable -- call
        ``.retain_grad()`` on it before ``backward()`` to get the screen-space
        position gradient 3DGS uses as its densification signal.
    ``radii``
        (N,) 3-sigma screen-space radius in pixels (0 if culled).
    ``visible``
        (N,) bool, gaussians that survived frustum/size culling.
    ``weight_sum``
        (N,) total ``T * alpha`` this gaussian contributed to the image; the
        per-frame support used by ``evaluate.coverage``.
    """

    rgb: Tensor
    alpha: Tensor
    depth: Tensor
    meta: dict

    def as_chw(self) -> Tensor:
        return self.rgb.permute(2, 0, 1)


def intrinsics_to_K(  # noqa: N802
    intrinsics: CameraIntrinsics | Tensor | np.ndarray,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Accept a :class:`CameraIntrinsics`, a 3x3 matrix or ``(fx, fy, cx, cy)``."""
    if isinstance(intrinsics, CameraIntrinsics):
        k = torch.zeros(3, 3, device=device, dtype=dtype)
        k[0, 0] = intrinsics.fx
        k[1, 1] = intrinsics.fy
        k[0, 2] = intrinsics.cx
        k[1, 2] = intrinsics.cy
        k[2, 2] = 1.0
        return k
    if isinstance(intrinsics, Tensor):
        if intrinsics.shape == (3, 3):
            return intrinsics.to(device=device, dtype=dtype) if device or dtype else intrinsics
        if intrinsics.numel() == 4:
            fx, fy, cx, cy = intrinsics.reshape(4).unbind()
            zero = torch.zeros((), device=fx.device, dtype=fx.dtype)
            one = torch.ones((), device=fx.device, dtype=fx.dtype)
            return torch.stack(
                [
                    torch.stack([fx, zero, cx]),
                    torch.stack([zero, fy, cy]),
                    torch.stack([zero, zero, one]),
                ]
            )
    arr = np.asarray(intrinsics, dtype=np.float64)
    if arr.shape == (3, 3):
        return torch.as_tensor(arr, device=device, dtype=dtype)
    if arr.size == 4:
        fx, fy, cx, cy = arr.reshape(4)
        return torch.as_tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=device, dtype=dtype
        )
    raise ValueError(f"cannot interpret intrinsics of shape {arr.shape}")


def _cov2d(
    means_cam: Tensor, cov_world: Tensor, r_w2c: Tensor, fx: Tensor, fy: Tensor, low_pass: float
) -> Tensor:
    """Screen-space 2x2 covariances via the local affine approximation."""
    x, y, z = means_cam.unbind(-1)
    z = z.clamp_min(1e-6)
    zero = torch.zeros_like(z)
    j = torch.stack(
        [
            torch.stack([fx / z, zero, -fx * x / (z * z)], dim=-1),
            torch.stack([zero, fy / z, -fy * y / (z * z)], dim=-1),
        ],
        dim=-2,
    )  # (N, 2, 3)
    t = j @ r_w2c.unsqueeze(0)  # (N, 2, 3)
    cov = t @ cov_world @ t.transpose(-1, -2)
    eye = torch.eye(2, device=cov.device, dtype=cov.dtype) * low_pass
    return cov + eye


def render(
    gaussians: Gaussians,
    c2w: Tensor,
    intrinsics: CameraIntrinsics | Tensor,
    width: int,
    height: int,
    *,
    t: float | None = None,
    bg_color: Tensor | tuple[float, float, float] | None = None,
    near: float = 0.05,
    far: float | None = None,
    low_pass: float = 0.3,
    use_sh_dirs: bool = True,
    pixel_chunk: int | None = None,
    max_alpha: float = 0.999,
) -> RenderOutput:
    """Render ``gaussians`` from the camera ``c2w`` / ``intrinsics``.

    ``t`` (project seconds) enables the temporal opacity window of gaussians
    that carry ``t_center`` / ``t_log_scale``; pass ``None`` for a purely
    static render.  ``bg_color`` composites behind the splats (a learned sky
    colour, typically).
    """
    device = gaussians.means.device
    dtype = gaussians.means.dtype
    c2w = torch.as_tensor(c2w, device=device, dtype=dtype).reshape(4, 4)
    k = intrinsics_to_K(intrinsics, device=device, dtype=dtype)
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]

    w2c = torch.linalg.inv(c2w)
    r_w2c, t_w2c = w2c[:3, :3], w2c[:3, 3]
    cam_center = c2w[:3, 3]

    means_cam = gaussians.means @ r_w2c.transpose(0, 1) + t_w2c
    depth = means_cam[:, 2]
    safe_z = depth.clamp_min(1e-6)
    means2d = torch.stack(
        [fx * means_cam[:, 0] / safe_z + cx, fy * means_cam[:, 1] / safe_z + cy], -1
    )

    rot = quat_to_rotmat_torch(gaussians.unit_quats())
    scaled = rot * gaussians.scales().unsqueeze(-2)
    cov_world = scaled @ scaled.transpose(-1, -2)
    cov = _cov2d(means_cam, cov_world, r_w2c, fx, fy, low_pass)

    det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] * cov[:, 1, 0]
    det_safe = det.clamp_min(1e-12)
    # inverse of the 2x2 covariance ("conic") -> (a, b, c) with b the off-diagonal
    conic = torch.stack([cov[:, 1, 1], -cov[:, 0, 1], cov[:, 0, 0]], dim=-1) / det_safe.unsqueeze(
        -1
    )

    opacity = gaussians.opacities_at(t)

    with torch.no_grad():
        trace = cov[:, 0, 0] + cov[:, 1, 1]
        disc = torch.sqrt((trace * 0.5) ** 2 - det_safe).clamp_min(0.0)
        lam = (trace * 0.5 + disc).clamp_min(1e-12)
        radii = 3.0 * torch.sqrt(lam)
        in_front = depth > near
        if far is not None:
            in_front = in_front & (depth < far)
        on_screen = (
            (means2d[:, 0] + radii > 0)
            & (means2d[:, 0] - radii < width)
            & (means2d[:, 1] + radii > 0)
            & (means2d[:, 1] - radii < height)
        )
        visible = in_front & on_screen & (opacity > _ALPHA_EPS) & (det > 1e-12)
        order = torch.argsort(torch.where(visible, depth, torch.full_like(depth, float("inf"))))
        order = order[visible[order]]

    n_vis = int(order.numel())
    rgb = torch.zeros(height, width, 3, device=device, dtype=dtype)
    acc_alpha = torch.zeros(height, width, 1, device=device, dtype=dtype)
    acc_depth = torch.zeros(height, width, 1, device=device, dtype=dtype)
    weight_sum = torch.zeros(gaussians.n, device=device, dtype=dtype)

    if n_vis:
        if use_sh_dirs and gaussians.sh_rest is not None:
            dirs = gaussians.means[order] - cam_center
            dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            colors = _eval_sh_subset(gaussians, order, dirs)
        else:
            colors = gaussians.colors(None)[order]

        mu = means2d[order]
        cn = conic[order]
        op = opacity[order]
        z = depth[order]

        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        px = torch.stack([xx.reshape(-1) + 0.5, yy.reshape(-1) + 0.5], dim=-1)  # (P, 2)
        n_px = px.shape[0]
        chunk = pixel_chunk or max(1024, min(n_px, int(4_000_000 // max(n_vis, 1)) or 1024))

        rgb_parts, alpha_parts, depth_parts = [], [], []
        for start in range(0, n_px, chunk):
            sl = slice(start, min(start + chunk, n_px))
            d = px[sl].unsqueeze(0) - mu.unsqueeze(1)  # (Nv, p, 2)
            dx, dy = d[..., 0], d[..., 1]
            power = -0.5 * (
                cn[:, 0:1] * dx * dx + 2.0 * cn[:, 1:2] * dx * dy + cn[:, 2:3] * dy * dy
            )
            alpha = (op.unsqueeze(1) * torch.exp(power.clamp(max=0.0))).clamp(0.0, max_alpha)
            one_minus = 1.0 - alpha
            # exclusive cumulative product = transmittance in front of each splat
            trans = torch.cat(
                [torch.ones_like(one_minus[:1]), torch.cumprod(one_minus, dim=0)[:-1]], dim=0
            )
            w = alpha * trans  # (Nv, p)
            rgb_parts.append(torch.einsum("np,nc->pc", w, colors))
            alpha_parts.append(w.sum(dim=0))
            depth_parts.append(torch.einsum("np,n->p", w, z))
            weight_sum = weight_sum.index_add(0, order, w.sum(dim=1).detach())

        rgb = torch.cat(rgb_parts, dim=0).reshape(height, width, 3)
        acc_alpha = torch.cat(alpha_parts, dim=0).reshape(height, width, 1)
        acc_depth = torch.cat(depth_parts, dim=0).reshape(height, width, 1)

    acc_depth = acc_depth / acc_alpha.clamp_min(1e-6)
    if bg_color is not None:
        bg = torch.as_tensor(bg_color, device=device, dtype=dtype).reshape(-1)
        if bg.numel() == 3:
            bg = bg.reshape(1, 1, 3)
        rgb = rgb + (1.0 - acc_alpha) * bg

    return RenderOutput(
        rgb=rgb,
        alpha=acc_alpha,
        depth=acc_depth,
        meta={
            "means2d": means2d,
            "radii": radii,
            "visible": visible,
            "weight_sum": weight_sum,
            "n_visible": n_vis,
            "backend": "cpu_raster",
            "width": width,
            "height": height,
        },
    )


def _eval_sh_subset(gaussians: Gaussians, index: Tensor, dirs: Tensor) -> Tensor:
    """View-dependent colour for a subset of gaussians (keeps autograd intact)."""
    sub = Gaussians(
        means=gaussians.means[index],
        log_scales=gaussians.log_scales[index],
        quats=gaussians.quats[index],
        sh_dc=gaussians.sh_dc[index],
        sh_rest=None if gaussians.sh_rest is None else gaussians.sh_rest[index],
        logit_opacities=gaussians.logit_opacities[index],
        layer=None if gaussians.layer is None else gaussians.layer[index],
    )
    return sub.colors(dirs)
