from wtc4d.corpus.harvest import wikimedia

FAKE_PAGE = {
    "pageid": 199285564,
    "title": "File:Pavel Hlava first plane 9.11.01.png",
    "imageinfo": [
        {
            "size": 1109595,
            "width": 1299,
            "height": 928,
            "url": "https://upload.wikimedia.org/wikipedia/commons/0/0c/x.png",
            "descriptionurl": "https://commons.wikimedia.org/wiki/File:x.png",
            "mime": "image/png",
            "extmetadata": {
                "LicenseShortName": {"value": "CC BY-SA 4.0"},
                "Artist": {"value": '<a href="...">Pavel Hlava</a>'},
                "Categories": {"value": "September 11 attacks|Screenshots"},
                "DateTimeOriginal": {"value": "2001-09-11 08:50:00"},
                "ImageDescription": {"value": "A screenshot."},
            },
        }
    ],
}


def test_source_from_page():
    src = wikimedia._source_from_page(FAKE_PAGE)
    assert src.id == "wm-199285564"
    assert src.kind == "photo"
    assert src.license == "cc-by-sa-4.0"
    assert src.creator == "Pavel Hlava"
    assert src.width == 1299 and src.height == 928
    assert "September 11 attacks" in src.tags
    assert src.time_hint is not None
    assert src.time_hint.t > 0


def test_source_from_page_no_imageinfo_returns_none():
    assert wikimedia._source_from_page({"pageid": 1, "title": "File:x"}) is None


def test_license_slug_public_domain():
    assert (
        wikimedia._license_slug({"LicenseShortName": {"value": "Public domain"}}) == "public-domain"
    )
    assert wikimedia._license_slug({}) == "unknown"


def test_video_mime_gives_video_kind():
    page = {**FAKE_PAGE, "imageinfo": [{**FAKE_PAGE["imageinfo"][0], "mime": "video/ogg"}]}
    src = wikimedia._source_from_page(page)
    assert src.kind == "video"


def test_parse_exif_datetime_bad_value_returns_none():
    assert wikimedia._parse_exif_datetime("not a date") is None
    assert wikimedia._parse_exif_datetime(None) is None


def test_api_get_retries_on_429_then_succeeds(monkeypatch):
    calls = {"n": 0}

    class Resp:
        def __init__(self, status_code, payload=None):
            self.status_code = status_code
            self.headers = {"Retry-After": "0"}
            self._payload = payload or {}

        def json(self):
            return self._payload

        def raise_for_status(self):
            pass

    def fake_get(url, params=None, headers=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return Resp(429)
        return Resp(200, {"query": {"pages": {}}})

    monkeypatch.setattr(wikimedia.requests, "get", fake_get)
    monkeypatch.setattr(wikimedia.time, "sleep", lambda s: None)

    result = wikimedia._api_get({"action": "query"})
    assert result == {"query": {"pages": {}}}
    assert calls["n"] == 2


def test_api_get_gives_up_after_max_retries(monkeypatch):
    class Resp:
        status_code = 429
        headers = {"Retry-After": "0"}

    monkeypatch.setattr(wikimedia.requests, "get", lambda *a, **kw: Resp())
    monkeypatch.setattr(wikimedia.time, "sleep", lambda s: None)

    assert wikimedia._api_get({"action": "query"}) is None
