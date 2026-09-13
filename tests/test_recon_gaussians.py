"""Gaussian container: serialisation round trips, SH, concatenation, layers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wtc4d.recon.gaussians import SH_C0, Gaussians, Layer, rgb_to_sh_dc, sh_dc_to_rgb


def _random(n: int = 24, sh_degree: int = 0, seed: int = 0) -> Gaussians:
    rng = np.random.default_rng(seed)
    xyz = rng.normal(scale=30.0, size=(n, 3)) + np.array([0.0, 0.0, 200.0])
    rgb = rng.random((n, 3))
    g = Gaussians.from_points(
        xyz, rgb, scale_m=5.0, opacity=0.3, layer=int(Layer.TOWERS), sh_degree=sh_degree
    )
    if sh_degree:
        g.sh_rest = torch.as_tensor(
            rng.normal(scale=0.1, size=tuple(g.sh_rest.shape)), dtype=torch.float32
        )
    gen = torch.Generator().manual_seed(seed)
    g.quats = torch.randn(n, 4, generator=gen)
    g.log_scales = torch.randn(n, 3, generator=gen) * 0.3
    return g


def test_from_points_shapes_and_activations():
    g = _random(10)
    assert g.n == 10
    assert g.means.shape == (10, 3)
    assert torch.allclose(g.opacities(), torch.sigmoid(g.logit_opacities))
    assert torch.allclose(g.scales(), torch.exp(g.log_scales))
    assert torch.allclose(g.unit_quats().norm(dim=-1), torch.ones(10), atol=1e-6)
    assert (g.layer == int(Layer.TOWERS)).all()


def test_sh_dc_matches_rgb_round_trip():
    rgb = torch.rand(17, 3)
    assert torch.allclose(sh_dc_to_rgb(rgb_to_sh_dc(rgb)), rgb, atol=1e-6)
    assert SH_C0 == pytest.approx(0.5 / np.sqrt(np.pi), rel=1e-12)


def test_colors_use_view_direction_only_with_higher_sh():
    g = _random(8, sh_degree=2)
    dirs = torch.nn.functional.normalize(torch.randn(8, 3), dim=-1)
    dc_only = g.colors(None)
    view = g.colors(dirs)
    assert view.shape == dc_only.shape
    assert not torch.allclose(view, dc_only)
    # a degree-0 set must be view-independent
    g0 = _random(8, sh_degree=0)
    assert torch.allclose(g0.colors(dirs), g0.colors(None))


@pytest.mark.parametrize("sh_degree", [0, 1, 3])
def test_ply_round_trip(tmp_path, sh_degree):
    g = _random(19, sh_degree=sh_degree)
    g.t_center = torch.full((g.n,), 35939.0)
    g.t_log_scale = torch.full((g.n,), 1.5)
    path = g.save_ply(tmp_path / "splat.ply")
    back = Gaussians.load_ply(path)

    assert back.n == g.n
    assert back.sh_degree == sh_degree
    assert torch.allclose(back.means, g.means, atol=1e-6)
    assert torch.allclose(back.log_scales, g.log_scales, atol=1e-6)
    assert torch.allclose(back.sh_dc, g.sh_dc, atol=1e-6)
    assert torch.allclose(back.logit_opacities, g.logit_opacities, atol=1e-6)
    assert torch.allclose(back.unit_quats(), g.unit_quats(), atol=1e-6)
    assert (back.layer == g.layer).all()
    assert torch.allclose(back.t_center, g.t_center)
    if sh_degree:
        assert torch.allclose(back.sh_rest, g.sh_rest, atol=1e-6)


def test_ply_uses_reference_field_names(tmp_path):
    from plyfile import PlyData

    g = _random(5, sh_degree=1)
    path = g.save_ply(tmp_path / "splat.ply")
    names = {p.name for p in PlyData.read(str(path))["vertex"].properties}
    required = {"x", "y", "z", "nx", "ny", "nz", "opacity"}
    required |= {f"f_dc_{i}" for i in range(3)}
    required |= {f"scale_{i}" for i in range(3)}
    required |= {f"rot_{i}" for i in range(4)}
    required |= {f"f_rest_{i}" for i in range(9)}  # 3 channels x (4 - 1) bands
    assert required <= names


def test_npz_round_trip(tmp_path):
    g = _random(11, sh_degree=1)
    back = Gaussians.load_npz(g.save_npz(tmp_path / "g.npz"))
    assert torch.allclose(back.means, g.means)
    assert torch.allclose(back.sh_rest, g.sh_rest)
    assert back.layer.dtype == torch.int64


def test_cat_raises_sh_degree_and_concatenates_layers():
    a = _random(6, sh_degree=0, seed=1)
    b = _random(4, sh_degree=2, seed=2)
    b.layer = torch.full((b.n,), int(Layer.SMOKE))
    merged = Gaussians.cat([a, b])
    assert merged.n == 10
    assert merged.sh_degree == 2
    assert torch.allclose(merged.sh_rest[:6], torch.zeros(6, 8, 3))
    assert (merged.layer[6:] == int(Layer.SMOKE)).all()


def test_select_and_layer_mask():
    g = _random(10)
    g.layer[:4] = int(Layer.SMOKE)
    smoke = g.select(g.layer_mask(Layer.SMOKE))
    assert smoke.n == 4
    assert g.select(g.layer_mask(Layer.TOWERS, Layer.GROUND)).n == 6


def test_temporal_opacity_window():
    g = _random(5)
    assert torch.allclose(g.temporal_weight(0.0), torch.ones(5))  # no window -> always on
    g.t_center = torch.full((5,), 100.0)
    g.t_log_scale = torch.full((5,), float(np.log(2.0)))
    assert torch.allclose(g.opacities_at(100.0), g.opacities(), atol=1e-6)
    far = g.opacities_at(120.0)
    assert float(far.max()) < 1e-6  # ten sigma away: gone
    assert float(g.temporal_weight(102.0)[0]) == pytest.approx(np.exp(-0.5), rel=1e-4)


def test_covariance_is_symmetric_positive_definite():
    g = _random(6)
    cov = g.covariances()
    assert torch.allclose(cov, cov.transpose(-1, -2), atol=1e-5)
    eig = torch.linalg.eigvalsh(cov.double())
    assert float(eig.min()) > 0
