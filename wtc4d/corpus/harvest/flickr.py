"""Flickr harvester.

Flickr's full search API (``flickr.photos.search``, filterable by
``license``, ``tags``, and ``min_/max_upload_date``) needs an API key
(https://www.flickr.com/services/apps/create/) exported as ``FLICKR_API_KEY``
-- not available in this environment. :func:`harvest_search` implements it
so a future run only needs the key.

Without a key, Flickr's public *feeds* endpoint
(``api.flickr.com/services/feeds/photos_public.gne``) needs no
authentication at all, so :func:`harvest_public_feed` uses that instead.
Caveat: it is a "recent uploads matching tag" feed, not a historical search
-- verified 2026-09-13 it returns only recent (anniversary-post,
non-original) photos for 9/11-related tags, not 2001-era Creative Commons
photos. It is still real, live data (not a stub), just low-precision; kept
as the best zero-key option and clearly tagged as such.
"""

from __future__ import annotations

import logging
import os
import re

import requests

from wtc4d.schema import Source, SourceKind

logger = logging.getLogger(__name__)

SEARCH_API_URL = "https://api.flickr.com/services/rest/"
PUBLIC_FEED_URL = "https://api.flickr.com/services/feeds/photos_public.gne"
USER_AGENT = "wtc4d-corpus/0.1 (https://github.com/WalterGropius/911_4D; research cataloguing tool)"

DEFAULT_TAGS = ["worldtradecenter", "groundzero", "septemberattacks", "911memorial"]
# Flickr license ids: 4,5,6,7,9,10 = CC BY / BY-SA / BY-ND / BY-NC / CC0 / PDM (no BY-NC-*)
CC_LICENSE_IDS = "4,5,6,7,9,10"
_LICENSE_ID_SLUG = {
    "1": "cc-by-nc-sa-2.0",
    "2": "cc-by-nc-2.0",
    "3": "cc-by-nc-nd-2.0",
    "4": "cc-by-2.0",
    "5": "cc-by-sa-2.0",
    "6": "cc-by-nd-2.0",
    "7": "public-domain",  # "no known copyright restrictions"
    "9": "cc0-1.0",
    "10": "public-domain",  # US Government Work
}


def harvest_search(
    api_key: str | None = None,
    tags: list[str] | None = None,
    min_upload_date: str = "2001-09-11",
    max_upload_date: str = "2001-12-31",
    per_page: int = 250,
) -> list[Source]:
    """Full historical search via ``flickr.photos.search``. Requires an API key."""
    api_key = api_key or os.environ.get("FLICKR_API_KEY")
    if not api_key:
        raise RuntimeError(
            "flickr harvest_search requires FLICKR_API_KEY (see "
            "https://www.flickr.com/services/apps/create/); none set in this environment."
        )
    tags = tags or DEFAULT_TAGS
    resp = requests.get(
        SEARCH_API_URL,
        params={
            "method": "flickr.photos.search",
            "api_key": api_key,
            "tags": ",".join(tags),
            "license": CC_LICENSE_IDS,
            "min_upload_date": min_upload_date,
            "max_upload_date": max_upload_date,
            "extras": "license,date_taken,owner_name,url_o,url_l,dimensions",
            "per_page": per_page,
            "format": "json",
            "nojsoncallback": "1",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("stat") != "ok":
        raise RuntimeError(f"Flickr API error: {data.get('message', data)}")
    return [_source_from_photo(p) for p in data["photos"]["photo"]]


def _source_from_photo(photo: dict) -> Source:
    license_slug = _LICENSE_ID_SLUG.get(str(photo.get("license", "")), "unknown")
    url = (
        photo.get("url_o")
        or photo.get("url_l")
        or f"https://www.flickr.com/photos/{photo.get('owner')}/{photo['id']}/"
    )
    return Source(
        id=f"flickr-{photo['id']}",
        kind=SourceKind.PHOTO,
        url=url,
        archive="flickr",
        title=photo.get("title", "") or "",
        creator=photo.get("ownername", "") or "",
        license=license_slug,
        width=int(photo["width_o"]) if photo.get("width_o") else None,
        height=int(photo["height_o"]) if photo.get("height_o") else None,
        notes=f"date_taken={photo.get('datetaken', '')}",
    )


def harvest_public_feed(tags: list[str] | None = None) -> list[Source]:
    """Zero-key fallback: Flickr's public recent-uploads-by-tag feed.

    Not a historical search (see module docstring) -- results are whatever
    is recently uploaded with these tags, not necessarily 2001-era photos.
    """
    tags = tags or DEFAULT_TAGS
    by_id: dict[str, Source] = {}
    for tag in tags:
        try:
            resp = requests.get(
                PUBLIC_FEED_URL,
                params={"tags": tag, "format": "json", "nojsoncallback": "1"},
                headers={"User-Agent": USER_AGENT},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("flickr: public feed request failed for tag %r: %s", tag, exc)
            continue
        for item in data.get("items", []):
            link = item.get("link", "")
            m = re.search(r"/photos/[^/]+/(\d+)/?", link)
            photo_id = m.group(1) if m else link
            src = Source(
                id=f"flickr-{photo_id}",
                kind=SourceKind.PHOTO,
                url=link,
                archive="flickr",
                title=re.sub(r"<[^>]+>", "", item.get("title", "") or ""),
                creator=re.sub(r"^.*\(([^)]+)\)\s*$", r"\1", item.get("author", "") or ""),
                license="unknown",
                notes="via public recent-uploads feed, tag="
                + tag
                + " (not a historical search, see harvest/flickr.py)",
                tags=["public_feed", f"tag:{tag}"],
            )
            by_id[src.id] = src
    return list(by_id.values())


def harvest(collection: str = "flickr_cc_911") -> list[Source]:
    if os.environ.get("FLICKR_API_KEY"):
        return harvest_search()
    logger.info("flickr: FLICKR_API_KEY not set, falling back to the public recent-uploads feed")
    return harvest_public_feed()
