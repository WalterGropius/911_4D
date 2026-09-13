from pathlib import Path

import numpy as np
import pytest

from wtc4d import timeline
from wtc4d.procedural import gaussians
from wtc4d.schema.scene import SceneManifest, SplatAsset

_REPO_ROOT = Path(__file__).resolve().parents[1]

_SAMPLE_TIMES = [
    timeline.hms(8, 0, 0),  # E0: intact
    timeline.hms(9, 30, 0),  # E2: both burning
    timeline.WTC2_COLLAPSE.t + 30.0,  # E3: WTC2 just gone
    timeline.hms(11, 0, 0),  # E4: both gone, dust
    timeline.hms(14, 0, 0),  # E5: rubble / aftermath
]


@pytest.mark.parametrize("t", _SAMPLE_TIMES)
def test_gaussian_count_within_bounds(t):
    cloud = gaussians.sample(t)
    assert 200 < len(cloud) < 200_000


@pytest.mark.parametrize("t", _SAMPLE_TIMES)
def test_gaussian_fields_are_well_formed(t):
    cloud = gaussians.sample(t)
    n = len(cloud)
    assert cloud.xyz.shape == (n, 3)
    assert cloud.scale.shape == (n, 3)
    assert cloud.quat.shape == (n, 4)
    assert cloud.rgb.shape == (n, 3)
    assert cloud.opacity.shape == (n,)
    assert cloud.layer.shape == (n,)
    assert np.all(cloud.scale > 0.0)
    assert np.all((cloud.opacity >= 0.0) & (cloud.opacity <= 1.0))
    assert np.all((cloud.rgb >= 0.0) & (cloud.rgb <= 1.0))
    assert np.all(np.isfinite(cloud.xyz))
    assert set(np.unique(cloud.layer)).issubset(set(range(len(gaussians.LAYER_NAMES))))


def test_ground_and_skyline_always_present():
    cloud = gaussians.sample(timeline.hms(8, 0, 0))
    assert cloud.layer_mask("ground").sum() > 0
    assert cloud.layer_mask("skyline").sum() > 0


def test_towers_present_only_before_they_are_gone():
    burning = gaussians.sample(timeline.hms(9, 30, 0))
    assert burning.layer_mask("towers").sum() > 0

    long_after = gaussians.sample(timeline.hms(14, 0, 0))
    assert long_after.layer_mask("towers").sum() == 0
    assert long_after.layer_mask("rubble").sum() > 0


def test_smoke_present_while_burning_and_dust_after_collapse():
    burning = gaussians.sample(timeline.hms(9, 30, 0))
    assert burning.layer_mask("smoke").sum() > 0
    assert burning.layer_mask("dust").sum() == 0

    after_collapse = gaussians.sample(timeline.WTC2_COLLAPSE.t + 120.0)
    assert after_collapse.layer_mask("dust").sum() > 0


def test_determinism_with_seed():
    a = gaussians.sample(timeline.hms(9, 30, 0), seed=7)
    b = gaussians.sample(timeline.hms(9, 30, 0), seed=7)
    assert len(a) == len(b)
    assert np.array_equal(a.xyz, b.xyz)
    assert np.array_equal(a.rgb, b.rgb)

    c = gaussians.sample(timeline.hms(9, 30, 0), seed=8)
    assert not np.array_equal(a.xyz, c.xyz) or not np.array_equal(a.rgb, c.rgb)


def test_ply_round_trip(tmp_path):
    cloud = gaussians.sample(timeline.hms(9, 30, 0), seed=3)
    path = tmp_path / "scene.ply"
    cloud.save_ply(path)
    loaded = gaussians.GaussianCloud.load_ply(path)

    assert len(loaded) == len(cloud)
    assert np.allclose(loaded.xyz, cloud.xyz, atol=1e-2)
    assert np.allclose(loaded.rgb, cloud.rgb, atol=1e-2)
    assert np.allclose(loaded.opacity, cloud.opacity, atol=1e-2)
    assert np.allclose(loaded.scale, cloud.scale, atol=1e-2, rtol=1e-3)
    assert np.allclose(loaded.quat, cloud.quat, atol=1e-4)


def test_npz_round_trip_is_exact(tmp_path):
    cloud = gaussians.sample(timeline.hms(9, 30, 0), seed=3)
    path = tmp_path / "scene.npz"
    cloud.save_npz(path)
    loaded = gaussians.GaussianCloud.load_npz(path)

    assert len(loaded) == len(cloud)
    assert np.array_equal(loaded.xyz, cloud.xyz)
    assert np.array_equal(loaded.rgb, cloud.rgb)
    assert np.array_equal(loaded.opacity, cloud.opacity)
    assert np.array_equal(loaded.scale, cloud.scale)
    assert np.array_equal(loaded.quat, cloud.quat)
    assert np.array_equal(loaded.layer, cloud.layer)


def test_concat_empty_is_identity():
    cloud = gaussians.sample(timeline.hms(9, 30, 0))
    empty = gaussians.GaussianCloud.empty()
    combined = cloud.concat(empty)
    assert len(combined) == len(cloud)


def test_manifest_from_sequence_validates(tmp_path):
    from typer.testing import CliRunner

    from wtc4d.procedural.cli import app

    runner = CliRunner()
    out_dir = tmp_path / "seq"
    result = runner.invoke(
        app,
        [
            "export",
            "--sequence",
            "08:40:00",
            "08:41:00",
            "--step",
            "30",
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output

    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = SceneManifest.model_validate_json(manifest_path.read_text())
    assert len(manifest.assets) == 3
    assert all(isinstance(a, SplatAsset) for a in manifest.assets)
    assert all((out_dir / a.url).exists() for a in manifest.assets)
    assert all(a.kind == "procedural" for a in manifest.assets)


def test_committed_example_manifest_validates():
    path = _REPO_ROOT / "data" / "scenes" / "procedural_manifest.example.json"
    manifest = SceneManifest.model_validate_json(path.read_text())
    assert manifest.assets
    assert all(a.kind == "procedural" for a in manifest.assets)
