"""Reference rasteriser: projection, occlusion, gradients, backend selection."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wtc4d.recon import backends, cpu_raster
from wtc4d.recon.conventions import look_at_c2w
from wtc4d.recon.gaussians import Gaussians
from wtc4d.schema.camera import CameraIntrinsics

W, H = 24, 18
INTR = CameraIntrinsics(width=W, height=H, fx=30.0, fy=30.0, cx=W / 2, cy=H / 2)
C2W = torch.as_tensor(look_at_c2w([600.0, 0.0, 200.0], [0.0, 0.0, 200.0]), dtype=torch.float32)


def _single(xyz=(0.0, 0.0, 200.0), rgb=(1.0, 0.0, 0.0), scale=6.0, opacity=0.99) -> Gaussians:
    return Gaussians.from_points([list(xyz)], [list(rgb)], scale_m=scale, opacity=opacity)


def test_render_shapes_and_background():
    out = cpu_raster.render(_single(), C2W, INTR, W, H, bg_color=(0.0, 0.0, 1.0))
    assert out.rgb.shape == (H, W, 3)
    assert out.alpha.shape == (H, W, 1)
    assert out.depth.shape == (H, W, 1)
    corner = out.rgb[0, 0]
    assert torch.allclose(corner, torch.tensor([0.0, 0.0, 1.0]), atol=1e-5)  # sky
    assert out.meta["backend"] == "cpu_raster"


def test_gaussian_lands_at_the_projected_pixel():
    g = _single(xyz=(0.0, 0.0, 200.0), scale=2.0)
    out = cpu_raster.render(g, C2W, INTR, W, H)
    peak = torch.argmax(out.alpha.reshape(-1))
    py, px = int(peak) // W, int(peak) % W
    mu = out.meta["means2d"][0]
    assert abs(px + 0.5 - float(mu[0])) <= 1.0
    assert abs(py + 0.5 - float(mu[1])) <= 1.0
    assert float(out.depth[py, px]) == pytest.approx(600.0, rel=0.02)


def test_nearer_gaussian_occludes_farther_one():
    near = _single(xyz=(300.0, 0.0, 200.0), rgb=(1.0, 0.0, 0.0), scale=8.0, opacity=0.99)
    far = _single(xyz=(-300.0, 0.0, 200.0), rgb=(0.0, 1.0, 0.0), scale=8.0, opacity=0.99)
    both = Gaussians.cat([near, far])
    out = cpu_raster.render(both, C2W, INTR, W, H)
    centre = out.rgb[H // 2, W // 2]
    assert float(centre[0]) > float(centre[1])  # red (near) wins
    # swapping colours must swap the result: ordering is by depth, not by index
    near2 = _single(xyz=(300.0, 0.0, 200.0), rgb=(0.0, 1.0, 0.0), scale=8.0, opacity=0.99)
    far2 = _single(xyz=(-300.0, 0.0, 200.0), rgb=(1.0, 0.0, 0.0), scale=8.0, opacity=0.99)
    out2 = cpu_raster.render(Gaussians.cat([near2, far2]), C2W, INTR, W, H)
    assert float(out2.rgb[H // 2, W // 2][1]) > float(out2.rgb[H // 2, W // 2][0])


def test_behind_camera_is_culled():
    behind = _single(xyz=(1200.0, 0.0, 200.0))  # camera sits at x=600 looking to -x
    out = cpu_raster.render(behind, C2W, INTR, W, H)
    assert not bool(out.meta["visible"].any())
    assert float(out.alpha.sum()) == 0.0


def test_opacity_controls_alpha():
    low = cpu_raster.render(_single(opacity=0.1), C2W, INTR, W, H)
    high = cpu_raster.render(_single(opacity=0.9), C2W, INTR, W, H)
    assert float(high.alpha.max()) > float(low.alpha.max()) * 3


def test_weight_sum_tracks_contribution():
    out = cpu_raster.render(_single(), C2W, INTR, W, H)
    assert out.meta["weight_sum"].shape == (1,)
    assert float(out.meta["weight_sum"][0]) == pytest.approx(float(out.alpha.sum()), rel=1e-4)


def test_pixel_chunking_does_not_change_the_image():
    g = Gaussians.cat([_single(xyz=(x, 0.0, 200.0), scale=9.0) for x in (-100.0, 0.0, 120.0)])
    a = cpu_raster.render(g, C2W, INTR, W, H, bg_color=(0.2, 0.2, 0.2))
    b = cpu_raster.render(g, C2W, INTR, W, H, bg_color=(0.2, 0.2, 0.2), pixel_chunk=7)
    assert torch.allclose(a.rgb, b.rgb, atol=1e-6)


def test_temporal_opacity_is_applied_at_render_time():
    g = _single(scale=6.0, opacity=0.99)
    g.t_center = torch.tensor([100.0])
    g.t_log_scale = torch.tensor([float(np.log(1.0))])
    on = cpu_raster.render(g, C2W, INTR, W, H, t=100.0)
    off = cpu_raster.render(g, C2W, INTR, W, H, t=110.0)
    assert float(on.alpha.max()) > 0.5
    assert float(off.alpha.max()) < 1e-3


@pytest.mark.parametrize(
    "attr,index",
    [
        ("means", (0, 2)),
        ("log_scales", (0, 1)),
        ("logit_opacities", (0,)),
        ("sh_dc", (0, 0)),
        ("quats", (0, 1)),
    ],
)
def test_gradients_match_finite_differences(attr, index):
    """The reference rasteriser must be differentiable *correctly*, not just runnable."""
    rng = np.random.default_rng(3)
    xyz = rng.normal(scale=20.0, size=(5, 3)) + np.array([0.0, 0.0, 200.0])
    g = Gaussians.from_points(xyz, rng.random((5, 3)), scale_m=10.0, opacity=0.6)
    g = g.to(dtype=torch.float64)
    g.requires_grad_(True)
    c2w = C2W.double()

    def loss() -> torch.Tensor:
        out = cpu_raster.render(g, c2w, INTR, W, H, bg_color=(0.3, 0.3, 0.3))
        return (out.rgb**2).sum() + out.alpha.sum()

    loss().backward()
    analytic = float(getattr(g, attr).grad[index])

    eps = 1e-6
    tensor = getattr(g, attr)
    with torch.no_grad():
        tensor[index] += eps
    plus = float(loss().detach())
    with torch.no_grad():
        tensor[index] -= 2 * eps
    minus = float(loss().detach())
    with torch.no_grad():
        tensor[index] += eps
    numeric = (plus - minus) / (2 * eps)

    assert numeric == pytest.approx(analytic, rel=2e-4, abs=1e-7)


def test_intrinsics_accept_tensor_and_matrix():
    k = torch.tensor([[30.0, 0.0, W / 2], [0.0, 30.0, H / 2], [0.0, 0.0, 1.0]])
    a = cpu_raster.render(_single(), C2W, INTR, W, H)
    b = cpu_raster.render(_single(), C2W, k, W, H)
    assert torch.allclose(a.rgb, b.rgb, atol=1e-6)


def test_focal_length_is_differentiable_for_refinement():
    fx = torch.tensor(30.0, requires_grad=True)
    k = torch.stack(
        [
            torch.stack([fx, torch.tensor(0.0), torch.tensor(W / 2)]),
            torch.stack([torch.tensor(0.0), fx, torch.tensor(H / 2)]),
            torch.tensor([0.0, 0.0, 1.0]),
        ]
    )
    out = cpu_raster.render(_single(scale=8.0), C2W, k, W, H)
    out.alpha.sum().backward()
    assert fx.grad is not None and float(fx.grad.abs()) > 0


def test_pose_is_differentiable_for_refinement():
    c2w = C2W.clone().requires_grad_(True)
    out = cpu_raster.render(_single(scale=8.0), c2w, INTR, W, H)
    out.rgb.sum().backward()
    assert c2w.grad is not None and float(c2w.grad.abs().sum()) > 0


def test_backend_resolution_without_cuda():
    assert "cpu" in backends.available_backends()
    assert backends.resolve("cpu") == "cpu"
    if not backends.gsplat_available():
        assert backends.resolve(None) == "cpu"
        assert backends.resolve("auto") == "cpu"
        with pytest.raises(RuntimeError, match="gsplat"):
            backends.resolve("gsplat")
    with pytest.raises(ValueError, match="unknown backend"):
        backends.resolve("nerf")


def test_backend_render_matches_cpu_raster_directly():
    a = backends.render(_single(), C2W, INTR, W, H, backend="cpu", bg_color=(0.1, 0.2, 0.3))
    b = cpu_raster.render(_single(), C2W, INTR, W, H, bg_color=(0.1, 0.2, 0.3))
    assert torch.allclose(a.rgb, b.rgb)
