"""Harvester for NIST FOIA #09-42 releases and the 911datasets.org mirror.

NIST's WTC Disaster Investigation (National Construction Safety Team Act,
2002-2008) collected raw videotapes and photographs from broadcasters, NYPD,
and private citizens. They were digitized and released via FOIA in 42
batches (2009-2010), originally distributed by 911datasets.org over
BitTorrent and now mirrored to archive.org with identifiers matching
``nist-foia-09-42-r*``. This is the highest-quality raw footage available
for the event (camera-original DV/Betacam masters, not broadcast
re-encodes) and is US-government public domain.

We use archive.org's plain ``/metadata/<identifier>`` JSON endpoint (via
``requests``) rather than ``internetarchive.get_item`` -- it returns
``item_size``/``files_count`` in one small response instead of a full file
listing, which matters here since a single release can carry hundreds of
multi-GB files.
"""

from __future__ import annotations

import logging

import requests

from wtc4d.schema import Source, SourceKind

logger = logging.getLogger(__name__)

METADATA_URL = "https://archive.org/metadata/{identifier}"
USER_AGENT = "wtc4d-corpus/0.1 (https://github.com/WalterGropius/911_4D; research cataloguing tool)"
_MAX_NOTES_CHARS = 300

# Verified 2026-09-13 via `ia.search_items('identifier:911datasets OR ...')`.
# These are hand-picked (not a fuzzy search) to avoid pulling in unrelated
# archive.org items that happen to mention "911" or "datasets".
DATASETS_MIRROR_IDENTIFIERS = [
    "911datasets",  # index item, "9/11 dataset - 305GB"
    "nist-r27-missing-vids",  # tapes NIST Release 27 omitted vs. the BitTorrent set
    "wiki-911datasetsorg",  # mirror of the (now defunct) 911datasets.org wiki
]

_KIND_BY_MEDIATYPE = {
    "movies": SourceKind.VIDEO,
    "image": SourceKind.PHOTO,
    "data": SourceKind.DOCUMENT,
    "texts": SourceKind.DOCUMENT,
    "web": SourceKind.DOCUMENT,
}


def _fetch_metadata(identifier: str) -> dict | None:
    try:
        resp = requests.get(
            METADATA_URL.format(identifier=identifier),
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("nist_foia: failed to fetch metadata for %s: %s", identifier, exc)
        return None
    if not data.get("metadata"):
        return None
    return data


def _trim(text: str) -> str:
    text = (text or "").strip()
    if len(text) > _MAX_NOTES_CHARS:
        text = text[: _MAX_NOTES_CHARS - 1].rstrip() + "…"
    return text


def _source_from_metadata(
    data: dict, *, identifier: str, license_: str, extra_tags: list[str], archive: str = "nist_foia"
) -> Source:
    meta = data["metadata"]
    mediatype = meta.get("mediatype", "other")
    kind = _KIND_BY_MEDIATYPE.get(mediatype, SourceKind.OTHER)

    import re

    description = re.sub(r"<[^>]+>", " ", meta.get("description", "") or "")
    description = re.sub(r"\s+", " ", description)

    return Source(
        id=f"ia-{identifier}",
        kind=kind,
        url=f"https://archive.org/details/{identifier}",
        archive=archive,
        title=meta.get("title", "") or identifier,
        creator=meta.get("creator", "") or "",
        license=license_,
        bytes=int(data["item_size"]) if data.get("item_size") else None,
        notes=_trim(description),
        tags=["nist_foia", *extra_tags],
    )


def iter_release_identifiers() -> list[str]:
    """List archive.org identifiers matching ``nist-foia-09-42-r*``."""
    import internetarchive as ia

    items = ia.search_items("identifier:nist-foia-09-42-r*", fields=["identifier"])
    return sorted(item["identifier"] for item in items if item.get("identifier"))


def harvest_releases(identifiers: list[str] | None = None) -> list[Source]:
    """Harvest the 42 (or so) NIST FOIA #09-42 release items."""
    identifiers = identifiers if identifiers is not None else iter_release_identifiers()
    sources = []
    for identifier in identifiers:
        data = _fetch_metadata(identifier)
        if data is None:
            continue
        sources.append(
            _source_from_metadata(
                data,
                identifier=identifier,
                license_="public-domain",
                extra_tags=["nist_foia_release"],
            )
        )
    logger.info(
        "nist_foia harvest_releases: %d of %d identifiers resolved", len(sources), len(identifiers)
    )
    return sources


def harvest_datasets_mirror(identifiers: list[str] | None = None) -> list[Source]:
    """Harvest the handful of known 911datasets.org mirror items."""
    identifiers = identifiers if identifiers is not None else DATASETS_MIRROR_IDENTIFIERS
    sources = []
    for identifier in identifiers:
        data = _fetch_metadata(identifier)
        if data is None:
            continue
        sources.append(
            _source_from_metadata(
                data,
                identifier=identifier,
                license_="unknown",
                extra_tags=["911datasets_mirror"],
                archive="archive.org",
            )
        )
    return sources


def harvest(collection: str = "ia_nist_foia_09_42") -> list[Source]:
    """Dispatch on the registry collection id (``ia_nist_foia_09_42`` or
    ``ia_911datasets_mirror``)."""
    if collection == "ia_911datasets_mirror":
        return harvest_datasets_mirror()
    return harvest_releases()
