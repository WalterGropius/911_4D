import pytest

from wtc4d.corpus.harvest import flickr

FAKE_PHOTO = {
    "id": "123456",
    "owner": "someone",
    "title": "WTC photo",
    "ownername": "Someone",
    "license": "4",
    "url_o": "https://live.staticflickr.com/x/123456_o.jpg",
    "width_o": "800",
    "height_o": "600",
    "datetaken": "2001-09-11 09:00:00",
}


def test_source_from_photo():
    src = flickr._source_from_photo(FAKE_PHOTO)
    assert src.id == "flickr-123456"
    assert src.license == "cc-by-2.0"
    assert src.width == 800 and src.height == 600
    assert "date_taken=2001-09-11" in src.notes


def test_harvest_search_requires_api_key(monkeypatch):
    monkeypatch.delenv("FLICKR_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FLICKR_API_KEY"):
        flickr.harvest_search()


def test_harvest_dispatches_to_public_feed_without_key(monkeypatch):
    monkeypatch.delenv("FLICKR_API_KEY", raising=False)
    monkeypatch.setattr(flickr, "harvest_public_feed", lambda: ["fallback"])
    assert flickr.harvest() == ["fallback"]


def test_harvest_dispatches_to_search_with_key(monkeypatch):
    monkeypatch.setenv("FLICKR_API_KEY", "fake-key")
    monkeypatch.setattr(flickr, "harvest_search", lambda: ["searched"])
    assert flickr.harvest() == ["searched"]


def test_public_feed_parses_items(monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "items": [
                    {
                        "title": "<p>New York</p>",
                        "link": "https://www.flickr.com/photos/someone/987654/",
                        "author": "nobody@flickr.com (Someone Else)",
                    }
                ]
            }

    monkeypatch.setattr(flickr.requests, "get", lambda *a, **kw: FakeResp())
    result = flickr.harvest_public_feed(tags=["worldtradecenter"])
    assert len(result) == 1
    assert result[0].id == "flickr-987654"
    assert result[0].title == "New York"
    assert result[0].creator == "Someone Else"
