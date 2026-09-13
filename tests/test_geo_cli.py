"""Smoke tests for the ``wtc4d geo`` CLI. No network, small synthetic camera."""

from __future__ import annotations

import pytest

trimesh = pytest.importorskip("trimesh")
typer_testing = pytest.importorskip("typer.testing")

CliRunner = typer_testing.CliRunner


def test_geo_cli_mounts_on_root_app():
    from wtc4d.cli import app as root_app

    # importable and mounted -> "geo" appears as a registered sub-typer group
    names = [g.name for g in root_app.registered_groups]
    assert "geo" in names


def test_info_and_cache_commands_run():
    from wtc4d.geo.cli import app

    runner = CliRunner()
    for cmd in (["info"], ["cache"]):
        result = runner.invoke(app, cmd)
        assert result.exit_code == 0, result.output


def test_preview_writes_png(tmp_path):
    from wtc4d.geo.cli import app

    out = tmp_path / "preview.png"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "preview",
            "--from",
            "40.7100,-74.0200,300",
            "--look-at",
            "0,0,300",
            "--width",
            "64",
            "--height",
            "48",
            "--out",
            str(out),
            "--backend",
            "raster",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists() and out.stat().st_size > 0
