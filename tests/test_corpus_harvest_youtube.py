import json

from wtc4d.corpus.harvest import youtube

FAKE_ENTRY = {
    "id": "abc123",
    "title": "9/11 raw footage",
    "channel": "Some Archive Channel",
    "duration": 120.0,
    "webpage_url": "https://www.youtube.com/watch?v=abc123",
    "view_count": 4242,
}


def test_source_from_entry():
    src = youtube._source_from_entry(FAKE_ENTRY, query="test query")
    assert src.id == "yt-abc123"
    assert src.archive == "youtube"
    assert src.duration_s == 120.0
    assert src.creator == "Some Archive Channel"
    assert "search:test query" in src.tags
    assert "4242" in src.notes


def test_source_from_entry_missing_id_returns_none():
    assert youtube._source_from_entry({"title": "no id"}, query="q") is None


def test_run_yt_dlp_parses_jsonl(monkeypatch):
    class FakeProc:
        returncode = 0
        stdout = "\n".join([json.dumps(FAKE_ENTRY), "", json.dumps({**FAKE_ENTRY, "id": "def456"})])
        stderr = ""

    def fake_run(cmd, capture_output, text, timeout, check):
        assert "ytsearch3:test" in cmd[-1]
        return FakeProc()

    monkeypatch.setattr(youtube.subprocess, "run", fake_run)
    entries = youtube._run_yt_dlp("test", 3)
    assert [e["id"] for e in entries] == ["abc123", "def456"]


def test_run_yt_dlp_handles_missing_binary(monkeypatch):
    def fake_run(*a, **kw):
        raise FileNotFoundError("yt-dlp not found")

    monkeypatch.setattr(youtube.subprocess, "run", fake_run)
    assert youtube._run_yt_dlp("test", 3) == []


def test_harvest_merges_and_tags_across_queries(monkeypatch):
    def fake_run_yt_dlp(query, n):
        return [FAKE_ENTRY]

    monkeypatch.setattr(youtube, "_run_yt_dlp", fake_run_yt_dlp)
    result = youtube.harvest(queries=["q1", "q2"], results_per_query=5)
    assert len(result) == 1
    assert sorted(result[0].tags) == ["search:q1", "search:q2"]
