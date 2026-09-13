from wtc4d.corpus.harvest import nist_foia

FAKE_METADATA = {
    "item_size": 433152695983,
    "files_count": 149,
    "metadata": {
        "identifier": "nist-foia-09-42-r27",
        "mediatype": "movies",
        "title": "NIST FOIA #09-42_Rel. 27 of 42",
        "creator": "National Institute of Standards and Technology (NIST)",
        "description": "<div>NIST FOIA request #09-42, Release 27 of 42.</div><div>Some <b>raw</b> tapes.</div>",
    },
}


def test_source_from_metadata_release():
    src = nist_foia._source_from_metadata(
        FAKE_METADATA,
        identifier="nist-foia-09-42-r27",
        license_="public-domain",
        extra_tags=["nist_foia_release"],
    )
    assert src.id == "ia-nist-foia-09-42-r27"
    assert src.archive == "nist_foia"
    assert src.kind == "video"
    assert src.bytes == 433152695983
    assert "raw" in src.notes
    assert "<" not in src.notes
    assert "nist_foia_release" in src.tags


def test_source_from_metadata_archive_override():
    src = nist_foia._source_from_metadata(
        FAKE_METADATA,
        identifier="911datasets",
        license_="unknown",
        extra_tags=["911datasets_mirror"],
        archive="archive.org",
    )
    assert src.archive == "archive.org"


def test_mediatype_to_kind_mapping():
    for mediatype, expected in [
        ("movies", "video"),
        ("image", "photo"),
        ("data", "document"),
        ("texts", "document"),
    ]:
        data = {"item_size": None, "metadata": {"mediatype": mediatype, "title": "x"}}
        src = nist_foia._source_from_metadata(
            data, identifier="x", license_="unknown", extra_tags=[]
        )
        assert src.kind == expected


def test_harvest_dispatch_selects_variant(monkeypatch):
    calls = []

    def fake_releases(identifiers=None):
        calls.append("releases")
        return []

    def fake_mirror(identifiers=None):
        calls.append("mirror")
        return []

    monkeypatch.setattr(nist_foia, "harvest_releases", fake_releases)
    monkeypatch.setattr(nist_foia, "harvest_datasets_mirror", fake_mirror)

    nist_foia.harvest("ia_nist_foia_09_42")
    nist_foia.harvest("ia_911datasets_mirror")
    assert calls == ["releases", "mirror"]


def test_fetch_metadata_handles_error(monkeypatch):
    import requests

    def fake_get(*a, **kw):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(requests, "get", fake_get)
    assert nist_foia._fetch_metadata("whatever") is None
