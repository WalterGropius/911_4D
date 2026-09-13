from wtc4d.corpus.harvest import archive_org

FAKE_ITEM = {
    "identifier": "ANT1_20010914_010000_Antenna_1_Greece",
    "title": "Antenna 1 Greece : ANT1 : September 13, 2001 9:00pm-9:30pm EDT",
    "date": "2001-09-14T00:00:00Z",
    "start_localtime": "2001-09-13 21:00:00",
    "runtime": "00:30:01",
    "collection": ["TV-ANT1", "911", "tvarchive", "television", "tvnews"],
    "mediatype": "movies",
    "contributor": "ANT1",
    "description": "From T5 Ku 97w",
    "source_pixel_width": "480",
    "source_pixel_height": "480",
    "frames_per_second": "29.97",
}


def test_parse_runtime():
    assert archive_org._parse_runtime("00:30:01") == 1801.0
    assert archive_org._parse_runtime(None) is None
    assert archive_org._parse_runtime("garbage") is None


def test_parse_start_localtime_is_edt():
    from wtc4d.timeline import EDT

    dt = archive_org._parse_start_localtime("2001-09-13 21:00:00")
    assert dt.tzinfo == EDT
    assert (dt.hour, dt.minute) == (21, 0)


def test_channel_tag():
    assert archive_org._channel_tag(["TV-ANT1", "911"]) == "ANT1"
    assert archive_org._channel_tag(["911", "tvarchive"]) is None
    assert archive_org._channel_tag(None) is None


def test_source_from_item():
    src = archive_org.source_from_item(FAKE_ITEM)
    assert src.id == "ia-ANT1_20010914_010000_Antenna_1_Greece"
    assert src.archive == "archive.org"
    assert src.kind == "tv_broadcast"
    assert src.creator == "ANT1"
    assert src.duration_s == 1801.0
    assert src.width == 480 and src.height == 480
    assert abs(src.fps - 29.97) < 1e-6
    assert src.time_hint is not None
    assert src.time_hint.method == "broadcast_metadata"
    # Sept 13 21:00 EDT -> project seconds since Sept 11 00:00 EDT
    from wtc4d.timeline import fmt_local

    assert fmt_local(src.time_hint.t) == "21:00:00 +2d"


def test_source_from_item_missing_identifier_returns_none():
    assert archive_org.source_from_item({"title": "no id"}) is None


def test_source_from_item_missing_optional_fields():
    src = archive_org.source_from_item({"identifier": "bare"})
    assert src is not None
    assert src.time_hint is None
    assert src.duration_s is None
    assert src.width is None
