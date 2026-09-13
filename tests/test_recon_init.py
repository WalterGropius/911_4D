"""Initialisation: mesh sampling, dedup, layer assignment, merging fallbacks."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("trimesh")

from wtc4d.recon.gaussians import Gaussians, Layer
from wtc4d.recon.init import (
    assign_layers,
    color_for_name,
    deduplicate,
    estimate_spacing,
    from_mesh,
    init_gaussians,
    layer_for_name,
    prior_boxes_scene,
)


def test_prior_boxes_scene_has_towers_and_ground():
    scene = prior_boxes_scene()
    names = set(scene.geometry.keys())
    assert {"WTC1", "WTC2", "ground"} <= names


def test_layer_for_name_keywords():
    assert layer_for_name("WTC1") == Layer.TOWERS
    assert layer_for_name("ground") == Layer.GROUND
    assert layer_for_name("smoke_plume_03") == Layer.SMOKE
    assert layer_for_name("rubble_pile") == Layer.DEBRIS
    assert layer_for_name("random_building_42") == Layer.BUILDINGS


def test_color_for_name_is_deterministic_and_bounded():
    a = color_for_name("140_west_st")
    b = color_for_name("140_west_st")
    c = color_for_name("something_else")
    assert a == b
    assert a != c
    assert all(0.0 <= v <= 1.0 for v in a)


def test_from_mesh_samples_towers_and_ground_with_layers():
    g = from_mesh(prior_boxes_scene(), n_points=400, seed=0)
    assert g.n >= 300  # sampling counts are rounded, not exact
    layers = set(int(x) for x in g.layer.unique())
    assert layers <= {int(Layer.TOWERS), int(Layer.GROUND)}
    assert (g.means[:, 2] >= -1.0).all()  # nothing below the fallback ground


def test_estimate_spacing_scales_with_density():
    rng = np.random.default_rng(0)
    dense = rng.normal(scale=1.0, size=(200, 3))
    sparse = dense * 10.0
    d_dense = estimate_spacing(dense)
    d_sparse = estimate_spacing(sparse)
    assert d_sparse.mean() > d_dense.mean() * 5


def test_estimate_spacing_handles_degenerate_input():
    assert estimate_spacing(np.zeros((0, 3))).shape == (0,)
    assert estimate_spacing(np.zeros((1, 3))) == pytest.approx([1.0])


def test_deduplicate_keeps_one_per_voxel_preferring_higher_opacity():
    xyz = np.array([[0.0, 0.0, 0.0], [0.05, 0.05, 0.05], [10.0, 10.0, 10.0]])
    g = Gaussians.from_points(xyz, scale_m=1.0, opacity=0.1)
    # make the second point (same voxel as the first) much more opaque
    g.logit_opacities = torch.tensor([-2.0, 5.0, -2.0])
    out = deduplicate(g, radius=0.5)
    assert out.n == 2
    # the surviving duplicate must be the more opaque one
    assert float(out.logit_opacities.max()) == pytest.approx(5.0)


def test_deduplicate_respects_layer_boundaries():
    xyz = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    g = Gaussians.from_points(xyz, scale_m=1.0, layer=[int(Layer.TOWERS), int(Layer.SMOKE)])
    out = deduplicate(g, radius=0.5)
    assert out.n == 2  # same voxel, different layer -> both kept


def test_deduplicate_noop_on_empty_or_zero_radius():
    g = Gaussians.from_points(np.zeros((0, 3)))
    assert deduplicate(g, radius=1.0).n == 0
    g2 = Gaussians.from_points(np.array([[0.0, 0.0, 0.0]] * 3))
    assert deduplicate(g2, radius=0.0).n == 3


def test_assign_layers_from_geometry():
    from wtc4d import world

    c = world.WTC1.enu_center()
    xyz = np.array(
        [
            [c[0], c[1], 100.0],  # inside WTC1 footprint, above ground
            [c[0], c[1], -5.0],  # below ground_z -> GROUND even though inside footprint x/y
            [5000.0, 5000.0, 50.0],  # far away -> BUILDINGS
        ]
    )
    g = Gaussians.from_points(xyz)
    g = assign_layers(g)
    assert int(g.layer[0]) == int(Layer.TOWERS)
    assert int(g.layer[1]) == int(Layer.GROUND)
    assert int(g.layer[2]) == int(Layer.BUILDINGS)


def test_init_gaussians_falls_back_to_mesh_when_other_priors_fail():
    g = init_gaussians(mesh_points=300, procedural_t=123.0, colmap_path="/nonexistent/path")
    assert g.n > 0
    assert g.means.shape[1] == 3
