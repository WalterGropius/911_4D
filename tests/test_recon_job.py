"""``job.run`` entrypoint: the toy jobs (what the infra smoke test calls) and
the scene manifest writer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wtc4d import timeline
from wtc4d.recon.job import JOBS_DIR, load_spec, run, write_scene_manifest


def test_run_env_reports_backend_and_torch():
    out = run({"kind": "env"})
    assert out["ok"] is True
    assert "torch" in out["env"]
    assert "cpu" in out["env"]["backends"]


def test_run_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown recon job kind"):
        run({"kind": "not_a_real_job"})


def test_toy_static_job_runs_end_to_end(tmp_path):
    out = run(
        {
            "kind": "toy_static",
            "iterations": 60,
            "n_cameras": 4,
            "width": 20,
            "height": 16,
            "n_points": 60,
            "out_dir": str(tmp_path),
            "holdout_every": 4,
            "config": {
                "device": "cpu",
                "backend": "cpu",
                "densify": {"enable": False},
                "output": {"ckpt_interval": 0},
            },
        }
    )
    assert out["ok"] is True
    assert out["n_gaussians"] > 0
    assert out["psnr_mean"] is not None


def test_toy_dynamic_job_runs_end_to_end(tmp_path):
    out = run(
        {
            "kind": "toy_dynamic",
            "iterations": 40,
            "n_cameras": 4,
            "n_times": 5,
            "width": 20,
            "height": 16,
            "n_points": 50,
            "out_dir": str(tmp_path),
            "config": {"device": "cpu", "backend": "cpu"},
        }
    )
    assert out["ok"] is True
    assert out["n_gaussians"] > 0


def test_example_job_specs_are_valid_json():
    specs = sorted(JOBS_DIR.glob("*.json"))
    assert len(specs) >= 4
    for path in specs:
        spec = load_spec(path)
        assert "kind" in spec
        assert "entrypoint" in spec
        assert "description" in spec


def test_toy_job_specs_run_as_shipped(tmp_path):
    """The CPU toy specs in wtc4d/recon/jobs/ must run as-is (no data needed) --
    that is their entire purpose as an infra smoke test."""
    for name in ("toy_static_cpu.json", "toy_dynamic_cpu.json"):
        spec = load_spec(JOBS_DIR / name)
        spec["out_dir"] = str(tmp_path / name)
        spec.setdefault("config", {})["device"] = "cpu"
        spec["config"]["backend"] = "cpu"
        out = run(spec)
        assert out["ok"] is True


def test_write_scene_manifest_defaults_events_to_timeline_anchors(tmp_path):
    path = write_scene_manifest(
        tmp_path / "scene.json",
        assets=[
            {
                "id": "e2",
                "url": "e2.ply",
                "t_start": timeline.WTC2_IMPACT.t,
                "t_end": timeline.WTC2_COLLAPSE.t,
            }
        ],
    )
    doc = json.loads(path.read_text())
    assert doc["assets"][0]["id"] == "e2"
    event_ids = {e["id"] for e in doc["events"]}
    assert {"wtc2_impact", "wtc2_collapse"} <= event_ids
    assert doc["world_origin"]["lat"] == pytest.approx(40.71120)


def test_write_scene_manifest_round_trips_through_schema(tmp_path):
    from wtc4d.schema.scene import SceneManifest

    path = write_scene_manifest(
        tmp_path / "scene.json",
        assets=[{"id": "a", "url": "a.ply", "t_start": 0.0, "t_end": 10.0}],
        cameras=[
            {
                "id": "cam0",
                "shot_id": "s0",
                "frame_idx": 0,
                "t": 5.0,
                "c2w": list(map(float, range(16))),
                "intrinsics": {"width": 10, "height": 10, "fx": 5, "fy": 5, "cx": 5, "cy": 5},
            }
        ],
    )
    manifest = SceneManifest.model_validate_json(path.read_text())
    assert manifest.assets[0].id == "a"
    assert manifest.cameras[0].shot_id == "s0"


def test_example_scene_manifest_is_valid_and_small():
    from wtc4d.schema.scene import SceneManifest

    path = Path("data/scenes/example_e2.json")
    assert path.exists()
    assert path.stat().st_size < 5_000
    manifest = SceneManifest.model_validate_json(path.read_text())
    assert manifest.assets and manifest.assets[0].epoch_id == "E2"
