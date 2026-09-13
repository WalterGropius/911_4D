"""CLI smoke tests: the app mounts, --help works, and a couple of pure-math
subcommands (no video/OCR/network) produce sane output."""

import json

from typer.testing import CliRunner

from wtc4d.sync.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "broadcast" in result.stdout
    assert "clock" in result.stdout
    assert "events" in result.stdout
    assert "audio" in result.stdout
    assert "solar" in result.stdout
    assert "fuse" in result.stdout


def test_mounted_on_root_cli():
    from wtc4d.cli import app as root_app

    result = runner.invoke(root_app, ["sync", "--help"])
    assert result.exit_code == 0


def test_solar_table_runs():
    result = runner.invoke(app, ["solar", "table"])
    assert result.exit_code == 0
    assert "E0 start" in result.stdout


def test_solar_time_shadow_azimuth():
    result = runner.invoke(
        app, ["solar", "time", "--shadow-azimuth", "286.0", "--sigma-deg", "1.0"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["method"] == "solar_shadow"


def test_solar_time_no_args_errors():
    result = runner.invoke(app, ["solar", "time"])
    assert result.exit_code == 2


def test_fuse_shot(tmp_path):
    candidates = [
        {"t": 100.0, "sigma": 1.0, "method": "broadcast_metadata", "evidence": "a"},
        {"t": 100.5, "sigma": 0.5, "method": "onscreen_clock", "evidence": "b"},
    ]
    p = tmp_path / "cands.json"
    p.write_text(json.dumps(candidates))
    result = runner.invoke(app, ["fuse", "shot", str(p)])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert 99.5 <= payload["estimate"]["t"] <= 101.0


def test_fuse_graph_and_report(tmp_path):
    priors_path = tmp_path / "priors.jsonl"
    priors_path.write_text(
        json.dumps(
            {"shot_id": "A", "time": {"t": 1000.0, "sigma": 1.0, "method": "broadcast_metadata"}}
        )
        + "\n"
    )
    offsets_path = tmp_path / "offsets.jsonl"
    offsets_path.write_text(
        json.dumps({"a": "A", "b": "B", "dt": 5.0, "sigma": 0.5, "method": "audio_xcorr"}) + "\n"
    )
    out_path = tmp_path / "time_estimates.jsonl"

    result = runner.invoke(
        app, ["fuse", "graph", str(priors_path), str(offsets_path), "--out", str(out_path)]
    )
    assert result.exit_code == 0
    assert out_path.exists()

    report_result = runner.invoke(app, ["fuse", "report", "--path", str(out_path)])
    assert report_result.exit_code == 0
    assert "shots timed" in report_result.stdout
