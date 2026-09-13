from wtc4d.corpus import store
from wtc4d.schema import Shot, Source, SourceKind


def _src(id_, **kw) -> Source:
    kw.setdefault("archive", "other")
    return Source(id=id_, kind=SourceKind.VIDEO, url=f"https://example.org/{id_}", **kw)


def test_write_read_roundtrip(tmp_path):
    path = tmp_path / "sources.jsonl"
    store._write_jsonl(path, [_src("a"), _src("b")])
    got = store.read_sources(path)
    assert [s.id for s in got] == ["a", "b"]


def test_merge_sources_new_wins():
    existing = {"a": _src("a", title="old")}
    merged = store.merge_sources(existing, [_src("a", title="new"), _src("b")])
    assert merged["a"].title == "new"
    assert set(merged) == {"a", "b"}


def test_save_sources_monolithic_under_limit(tmp_path):
    by_id = {s.id: s for s in [_src("b"), _src("a"), _src("c")]}
    written = store.save_sources(by_id, manifest_dir=tmp_path, max_bytes=10_000_000)
    assert written == [tmp_path / "sources.jsonl"]
    loaded = store.load_all_sources(tmp_path)
    assert set(loaded) == {"a", "b", "c"}
    # stable ordering: sorted by id
    ids_in_file = [
        line and Source.model_validate_json(line).id
        for line in (tmp_path / "sources.jsonl").read_text().splitlines()
    ]
    assert ids_in_file == ["a", "b", "c"]


def test_save_sources_splits_when_over_limit(tmp_path):
    by_id = {
        "x": _src("x", archive="archive.org"),
        "y": _src("y", archive="wikimedia"),
    }
    written = store.save_sources(by_id, manifest_dir=tmp_path, max_bytes=1)
    names = sorted(p.name for p in written)
    assert names == ["sources-archive.org.jsonl", "sources-wikimedia.jsonl"]
    assert not (tmp_path / "sources.jsonl").exists()
    loaded = store.load_all_sources(tmp_path)
    assert set(loaded) == {"x", "y"}


def test_save_sources_clears_stale_split_files(tmp_path):
    by_id = {"x": _src("x", archive="archive.org"), "y": _src("y", archive="wikimedia")}
    store.save_sources(by_id, manifest_dir=tmp_path, max_bytes=1)
    # now everything fits monolithic again -- split files must be cleaned up
    store.save_sources({"x": by_id["x"]}, manifest_dir=tmp_path, max_bytes=10_000_000)
    assert (tmp_path / "sources.jsonl").exists()
    assert not (tmp_path / "sources-wikimedia.jsonl").exists()


def test_merge_shots(tmp_path):
    path = tmp_path / "shots.jsonl"
    s1 = Shot(id="s1", source_id="src", start_frame=0, end_frame=10, fps=25.0)
    s2 = Shot(id="s2", source_id="src", start_frame=10, end_frame=20, fps=25.0)
    store.merge_shots([s1, s2], path=path)
    s1b = Shot(id="s1", source_id="src", start_frame=0, end_frame=15, fps=25.0)
    merged = store.merge_shots([s1b], path=path)
    assert {s.id: s.end_frame for s in merged} == {"s1": 15, "s2": 20}
