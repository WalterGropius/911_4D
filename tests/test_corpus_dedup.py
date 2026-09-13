import pytest

pytest.importorskip("cv2")
pytest.importorskip("imagehash")

from wtc4d.corpus import dedup  # noqa: E402
from wtc4d.schema import Source, SourceKind  # noqa: E402


def _src(id_) -> Source:
    return Source(id=id_, kind=SourceKind.VIDEO, url=f"https://example.org/{id_}", archive="other")


def test_hash_frames_same_video_is_self_similar(static_video):
    h1 = dedup.hash_frames(static_video)
    h2 = dedup.hash_frames(static_video)
    assert len(h1) > 0
    assert dedup._min_distance(h1, h2) == 0


def test_hash_frames_different_videos_differ(static_video, pan_video):
    h_static = dedup.hash_frames(static_video)
    h_pan = dedup.hash_frames(pan_video)
    assert dedup._min_distance(h_static, h_pan) > dedup.DEFAULT_HAMMING_THRESHOLD


def test_find_duplicates_links_identical_hashes(static_video, pan_video):
    h_static = dedup.hash_frames(static_video)
    h_pan = dedup.hash_frames(pan_video)
    by_source = {"a": h_static, "b": h_static, "c": h_pan}
    links = dedup.find_duplicates(by_source, threshold=dedup.DEFAULT_HAMMING_THRESHOLD)
    assert len(links) == 1
    assert {links[0].source_id, links[0].other_id} == {"a", "b"}
    assert links[0].distance == 0


def test_find_duplicates_skips_empty_hash_lists():
    links = dedup.find_duplicates({"a": [], "b": []})
    assert links == []


def test_annotate_duplicates_tags_both_sides():
    sources = {"a": _src("a"), "b": _src("b")}
    link = dedup.DuplicateLink(source_id="a", other_id="b", distance=2)
    dedup.annotate_duplicates(sources, [link])
    assert "dup:b" in sources["a"].tags
    assert "dup:a" in sources["b"].tags
    assert "distance 2" in sources["a"].notes


def test_annotate_duplicates_is_idempotent():
    sources = {"a": _src("a"), "b": _src("b")}
    link = dedup.DuplicateLink(source_id="a", other_id="b", distance=2)
    dedup.annotate_duplicates(sources, [link])
    dedup.annotate_duplicates(sources, [link])
    assert sources["a"].tags.count("dup:b") == 1
