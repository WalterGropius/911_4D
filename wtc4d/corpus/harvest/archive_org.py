"""Harvester for the archive.org "Understanding 9/11" TV News Archive.

Collection identifier ``911`` (~5,180 items as of 2026-09-13, see
``data/manifests/registry.yaml``). Each item is one off-air recording (often
a 30-minute block) from one of ~20 channels, with ``start_localtime`` and
``runtime`` metadata accurate to the second -- the best available temporal
anchor for cross-video sync.

Uses the ``internetarchive`` package's bulk search API (``search_items``
with an explicit ``fields`` list), which returns every requested field in
one paginated scan -- no per-item metadata fetch needed for ~5k items.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import datetime

from wtc4d.schema import Source, SourceKind, TimeEstimate
from wtc4d.schema.time import TimeMethod
from wtc4d.timeline import EDT, from_datetime

logger = logging.getLogger(__name__)

COLLECTION = "911"
SEARCH_FIELDS = [
    "identifier",
    "title",
    "date",
    "start_localtime",
    "runtime",
    "collection",
    "mediatype",
    "contributor",
    "description",
    "source_pixel_width",
    "source_pixel_height",
    "frames_per_second",
]

# start_localtime is a scan-start timestamp, not guaranteed frame-accurate
# (tape ingest/splice rounding); treat it as accurate to a few seconds.
START_LOCALTIME_SIGMA_S = 3.0
_MAX_NOTES_CHARS = 200


def _parse_runtime(runtime: str | None) -> float | None:
    """``"HH:MM:SS"`` (or ``"HH:MM:SS.ff"``) -> seconds."""
    if not runtime:
        return None
    parts = runtime.strip().split(":")
    if len(parts) != 3:
        return None
    try:
        h, m, s = parts
        return int(h) * 3600.0 + int(m) * 60.0 + float(s)
    except ValueError:
        return None


def _parse_start_localtime(value: str | None) -> datetime | None:
    """``"YYYY-MM-DD HH:MM:SS"`` naive local time -> tz-aware EDT datetime.

    archive.org's TV archive metadata carries ``utc_offset: "-400"`` for
    every item in this collection (Sept 2001, EDT year-round for the
    relevant window), matching ``wtc4d.timeline.EDT``.
    """
    if not value:
        return None
    try:
        naive = datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return naive.replace(tzinfo=EDT)


def _parse_float(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_int(value: str | None) -> int | None:
    f = _parse_float(value)
    return int(f) if f is not None else None


def _channel_tag(collection: list[str] | str | None) -> str | None:
    """Pick the ``TV-<CALL LETTERS>`` tag out of an item's collection list."""
    if isinstance(collection, str):
        collection = [collection]
    for tag in collection or []:
        if tag.startswith("TV-"):
            return tag[len("TV-") :]
    return None


def source_from_item(item: dict) -> Source | None:
    identifier = item.get("identifier")
    if not identifier:
        return None

    channel = _channel_tag(item.get("collection")) or item.get("contributor", "") or ""
    start_dt = _parse_start_localtime(item.get("start_localtime"))
    time_hint = None
    if start_dt is not None:
        time_hint = TimeEstimate(
            t=from_datetime(start_dt),
            sigma=START_LOCALTIME_SIGMA_S,
            method=TimeMethod.BROADCAST_METADATA,
            evidence=f"archive.org item '{identifier}' start_localtime={item.get('start_localtime')}",
        )

    description = (item.get("description") or "").strip()
    if len(description) > _MAX_NOTES_CHARS:
        description = description[: _MAX_NOTES_CHARS - 1].rstrip() + "…"

    tags = item.get("collection") or []
    if isinstance(tags, str):
        tags = [tags]

    return Source(
        id=f"ia-{identifier}",
        kind=SourceKind.TV_BROADCAST,
        url=f"https://archive.org/details/{identifier}",
        archive="archive.org",
        title=item.get("title", "") or "",
        creator=channel,
        license="fair-use-research",
        duration_s=_parse_runtime(item.get("runtime")),
        width=_parse_int(item.get("source_pixel_width")),
        height=_parse_int(item.get("source_pixel_height")),
        fps=_parse_float(item.get("frames_per_second")),
        time_hint=time_hint,
        location_hint_text=f"{channel} (off-air TV capture)" if channel else "",
        tags=list(tags),
        notes=description,
    )


def iter_raw_items(collection: str = COLLECTION, max_items: int | None = None) -> Iterator[dict]:
    """Yield raw ``internetarchive`` search result dicts for ``collection``."""
    import internetarchive as ia

    results = ia.search_items(f"collection:{collection}", fields=SEARCH_FIELDS)
    for n, item in enumerate(results):
        if max_items is not None and n >= max_items:
            return
        yield item


def harvest(collection: str = COLLECTION, max_items: int | None = None) -> list[Source]:
    """Harvest ``collection`` (default: the ``911`` TV News Archive) into Sources."""
    sources: list[Source] = []
    skipped = 0
    for item in iter_raw_items(collection=collection, max_items=max_items):
        src = source_from_item(item)
        if src is None:
            skipped += 1
            continue
        sources.append(src)
    logger.info(
        "archive_org harvest: %d sources, %d skipped (no identifier)", len(sources), skipped
    )
    return sources
