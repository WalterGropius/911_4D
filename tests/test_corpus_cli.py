from typer.testing import CliRunner

from wtc4d.corpus.cli import app
from wtc4d.schema import Source, SourceKind

runner = CliRunner()


def test_registry_list_runs():
    result = runner.invoke(app, ["registry-list"])
    assert result.exit_code == 0
    assert "ia_911_tv_archive" in result.stdout


def test_harvest_unknown_collection_errors():
    result = runner.invoke(app, ["harvest", "--collection", "not-a-real-collection"])
    assert result.exit_code != 0


def test_harvest_dispatches_and_saves(tmp_path, monkeypatch):
    from wtc4d.corpus import cli, store

    monkeypatch.setattr(store, "MANIFEST_DIR", tmp_path)

    def fake_dispatch(collection_id, limit):
        return [Source(id="x", kind=SourceKind.VIDEO, url="https://x", archive="other")]

    monkeypatch.setattr(cli, "_harvest_dispatch", fake_dispatch)
    result = runner.invoke(app, ["harvest", "--collection", "documentary_pointers"])
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "sources.jsonl").exists()


def test_dedup_requires_matching_counts():
    result = runner.invoke(
        app,
        ["dedup", "--videos", "a.mp4", "--videos", "b.mp4", "--source-ids", "only-one"],
    )
    assert result.exit_code != 0
