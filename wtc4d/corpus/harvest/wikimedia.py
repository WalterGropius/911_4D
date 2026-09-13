"""Harvester for Wikimedia Commons categories via the MediaWiki API.

Commons requires a descriptive ``User-Agent`` (unauthenticated requests
without one get a bare 403) and rate-limits aggressively under concurrent
load (429 with no ``Retry-After``); :func:`_api_get` backs off and retries.
"""

from __future__ import annotations

import logging
import re
import time

import requests

from wtc4d.schema import Source, SourceKind, TimeEstimate
from wtc4d.schema.time import TimeMethod
from wtc4d.timeline import EDT, from_datetime

logger = logging.getLogger(__name__)

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "wtc4d-corpus/0.1 (https://github.com/WalterGropius/911_4D; research cataloguing tool)"

# Verified reachable via the MediaWiki API, 2026-09-13 (see registry.yaml).
CATEGORIES = [
    "Category:September 11 attacks",
    "Category:World Trade Center (1973–2001)",
    "Category:Collapse of the World Trade Center",
    "Category:Aftermath of the September 11 attacks",
    "Category:Ground Zero (World Trade Center)",
]

PAGE_LIMIT = 100  # gcmlimit per request
MAX_RETRIES = 6
BASE_BACKOFF_S = 5.0


def _api_get(params: dict) -> dict | None:
    params = {**params, "format": "json"}
    delay = BASE_BACKOFF_S
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                API_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=30
            )
        except requests.RequestException as exc:
            logger.warning("wikimedia: request error (attempt %d): %s", attempt, exc)
            time.sleep(delay)
            delay *= 2
            continue
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", delay))
            logger.info("wikimedia: rate limited, sleeping %.0fs (attempt %d)", wait, attempt)
            time.sleep(wait)
            delay *= 2
            continue
        if resp.status_code >= 500:
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
        return resp.json()
    logger.warning("wikimedia: giving up after %d retries: %s", MAX_RETRIES, params)
    return None


def _strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()


def _license_slug(extmetadata: dict) -> str:
    short = extmetadata.get("LicenseShortName", {}).get("value", "")
    if not short:
        return "unknown"
    if short.lower().startswith("public domain"):
        return "public-domain"
    return re.sub(r"[^a-z0-9.]+", "-", short.lower()).strip("-")


def _parse_exif_datetime(value: str | None):
    if not value:
        return None
    value = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S"):
        try:
            from datetime import datetime

            return datetime.strptime(value, fmt).replace(tzinfo=EDT)
        except ValueError:
            continue
    return None


def _source_from_page(page: dict) -> Source | None:
    imageinfo = (page.get("imageinfo") or [None])[0]
    if imageinfo is None:
        return None
    extmetadata = imageinfo.get("extmetadata", {})
    title = page.get("title", "")
    mime = imageinfo.get("mime", "") or ""
    kind = SourceKind.VIDEO if mime.startswith("video/") else SourceKind.PHOTO

    time_hint = None
    exif_dt = _parse_exif_datetime(extmetadata.get("DateTimeOriginal", {}).get("value"))
    if exif_dt is not None:
        time_hint = TimeEstimate(
            t=from_datetime(exif_dt),
            sigma=300.0,  # camera clock/timezone not verified -- coarse
            method=TimeMethod.BROADCAST_METADATA,
            evidence=f"Wikimedia Commons EXIF DateTimeOriginal on '{title}' (timezone assumed EDT)",
        )

    return Source(
        id=f"wm-{page['pageid']}",
        kind=kind,
        url=imageinfo.get("descriptionurl", f"https://commons.wikimedia.org/wiki/{title}"),
        archive="wikimedia",
        title=title.removeprefix("File:"),
        creator=_strip_html(extmetadata.get("Artist", {}).get("value", "")),
        license=_license_slug(extmetadata),
        bytes=imageinfo.get("size"),
        width=imageinfo.get("width"),
        height=imageinfo.get("height"),
        time_hint=time_hint,
        tags=[
            c.strip()
            for c in _strip_html(extmetadata.get("Categories", {}).get("value", "")).split("|")
            if c
        ],
        notes=_strip_html(extmetadata.get("ImageDescription", {}).get("value", ""))[:300],
    )


def harvest_category(category: str, limit: int | None = None) -> list[Source]:
    sources: list[Source] = []
    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": category,
        "gcmtype": "file",
        "gcmlimit": min(PAGE_LIMIT, limit) if limit else PAGE_LIMIT,
        "prop": "imageinfo",
        "iiprop": "url|size|extmetadata|mime",
    }
    while True:
        data = _api_get(params)
        if data is None:
            break
        pages = (data.get("query") or {}).get("pages", {})
        for page in pages.values():
            src = _source_from_page(page)
            if src is not None:
                sources.append(src)
            if limit is not None and len(sources) >= limit:
                return sources
        cont = data.get("continue")
        if not cont:
            break
        params = {**params, **cont}
    return sources


def harvest(
    categories: list[str] | None = None, limit_per_category: int | None = None
) -> list[Source]:
    categories = categories if categories is not None else CATEGORIES
    by_id: dict[str, Source] = {}
    for category in categories:
        for src in harvest_category(category, limit=limit_per_category):
            by_id[src.id] = src
    logger.info(
        "wikimedia harvest: %d unique files across %d categories", len(by_id), len(categories)
    )
    return list(by_id.values())
