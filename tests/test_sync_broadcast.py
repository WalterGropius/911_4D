"""Broadcast/archive.org metadata parsing tests.

Uses the cached real-metadata fixtures in data/time/examples/ (fetched once,
committed, never re-fetched here) so the parsing is checked against actual
archive.org output rather than a hand-rolled fake -- no network in tests.
"""

import json
from pathlib import Path

import pytest

from wtc4d.schema.corpus import Shot, Source, SourceKind
from wtc4d.sync import broadcast as B
from wtc4d.timeline import fmt_local

EXAMPLES = Path(__file__).resolve().parents[1] / "data" / "time" / "examples"


def _load(name: str) -> dict:
    return json.loads((EXAMPLES / name).read_text())


def test_parse_archive_datetime():
    dt = B.parse_archive_datetime("2001-09-11 13:00:00")
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second) == (2001, 9, 11, 13, 0, 0)


def test_parse_runtime_variants():
    assert B.parse_runtime("00:29:56") == 29 * 60 + 56
    assert B.parse_runtime("29:56") == 29 * 60 + 56
    assert B.parse_runtime("1796.26") == pytest.approx(1796.26)


def test_parse_utc_offset():
    assert B.parse_utc_offset("-400") == -240
    assert B.parse_utc_offset("+0100") == 60
    assert B.parse_utc_offset(0) == 0


def test_parse_identifier():
    channel, dt, title = B.parse_identifier("WUSA_20010911_093000_News")
    assert channel == "WUSA"
    assert (dt.hour, dt.minute) == (9, 30)
    assert title == "News"


@pytest.mark.parametrize(
    "fname,expected_channel,expected_local",
    [
        ("CNN_20010911_130000_CNN_Live_This_Morning.metadata.json", "CNN", "09:00:00"),
        ("WUSA_20010911_093000_News.metadata.json", "WUSA", "05:30:00"),
    ],
)
def test_archive_item_timing_from_real_metadata(fname, expected_channel, expected_local):
    md = _load(fname)
    timing = B.ArchiveItemTiming.from_metadata(md)
    assert timing.channel == expected_channel
    assert fmt_local(timing.start_t) == expected_local
    ok, problems = timing.consistent()
    assert ok, problems


def test_cnn_item_is_satellite_reception():
    md = _load("CNN_20010911_130000_CNN_Live_This_Morning.metadata.json")
    timing = B.ArchiveItemTiming.from_metadata(md)
    assert timing.reception == "satellite"


def test_wusa_item_is_antenna_reception():
    md = _load("WUSA_20010911_093000_News.metadata.json")
    timing = B.ArchiveItemTiming.from_metadata(md)
    assert timing.reception == "antenna"


def test_time_at_offset_applies_channel_delay():
    md = _load("CNN_20010911_130000_CNN_Live_This_Morning.metadata.json")
    timing = B.ArchiveItemTiming.from_metadata(md)
    delays = B.ChannelDelays(
        default=B.ChannelDelay(delay_s=0.0, sigma_s=1.0),
        channels={"CNN": B.ChannelDelay(delay_s=5.0, sigma_s=2.0)},
    )
    est = timing.time_at_offset(100.0, delays=delays)
    # start_t (media offset 0) is 09:00:00; +100s offset - 5s delay = 09:01:35
    assert fmt_local(est.t) == "09:01:35"
    assert est.sigma == pytest.approx((B.ARCHIVE_CLOCK_SIGMA_S**2 + 2.0**2) ** 0.5)


def test_channel_delays_lookup_order():
    delays = B.ChannelDelays(
        default=B.ChannelDelay(delay_s=1.0),
        reception={"satellite": B.ChannelDelay(delay_s=2.0)},
        channels={"CNN": B.ChannelDelay(delay_s=3.0)},
    )
    assert delays.for_channel("CNN", reception="satellite").delay_s == 3.0
    assert delays.for_channel("XYZ", reception="satellite").delay_s == 2.0
    assert delays.for_channel("XYZ", reception="cable").delay_s == 1.0


def test_load_channel_delays_from_repo_yaml():
    delays = B.load_channel_delays()
    cnn = delays.for_channel("CNN", reception="satellite")
    assert cnn.delay_s > 0
    assert cnn.sigma_s > 0


def test_timing_for_source_from_identifier():
    src = Source(
        id="ia-WUSA_20010911_093000_News",
        kind=SourceKind.TV_BROADCAST,
        url="https://archive.org/details/WUSA_20010911_093000_News",
        archive="archive.org",
    )
    timing = B.timing_for_source(src)
    assert timing is not None
    assert timing.channel == "WUSA"


def test_estimate_for_shot_prefers_timing_over_hint():
    md = _load("WUSA_20010911_093000_News.metadata.json")
    timing = B.ArchiveItemTiming.from_metadata(md)
    src = Source(
        id="ia-WUSA_20010911_093000_News",
        kind=SourceKind.TV_BROADCAST,
        url="https://archive.org/details/WUSA_20010911_093000_News",
        archive="archive.org",
    )
    shot = Shot(id="s1", source_id=src.id, start_frame=0, end_frame=300, fps=29.97)
    est = B.estimate_for_shot(src, shot, timing=timing)
    assert est is not None
    # item start_t is 05:30:00 local; the WUSA (antenna) channel delay is a
    # fraction of a second, so this should land right at that mark.
    assert abs(est.t - timing.start_t) < 1.0


def test_estimate_for_offset_falls_back_to_time_hint():
    from wtc4d.schema.time import TimeEstimate, TimeMethod

    src = Source(
        id="yt-abc123",
        kind=SourceKind.VIDEO,
        url="https://youtube.com/watch?v=abc123",
        archive="youtube",
        time_hint=TimeEstimate(t=1000.0, sigma=30.0, method=TimeMethod.MANUAL),
    )
    est = B.estimate_for_offset(src, 5.0)
    assert est is not None
    assert est.t == 1005.0
    assert est.sigma == 30.0


def test_estimate_for_offset_none_without_timing_or_hint():
    src = Source(id="x", kind=SourceKind.VIDEO, url="http://x", archive="other")
    assert B.estimate_for_offset(src, 5.0) is None
