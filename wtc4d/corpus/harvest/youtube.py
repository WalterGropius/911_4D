"""Harvester for YouTube via ``yt-dlp`` metadata dumps (no video downloads).

There is no single official "9/11 archive" channel, so this runs a curated
list of search queries through ``yt-dlp --flat-playlist --dump-json
ytsearch<N>:<query>`` and keeps title/channel/duration/view_count. This is
fast (~1-2s per query, no per-video extraction) but ``upload_date`` in flat
mode is unreliable as a footage-date proxy, so ``time_hint`` is left unset
for every YouTube source -- the ``sync`` workstream should date these from
content, not upload metadata.
"""

from __future__ import annotations

import json
import logging
import subprocess

from wtc4d.schema import Source, SourceKind

logger = logging.getLogger(__name__)

RESULTS_PER_QUERY = 20

# Curated 2026-09-13: raw/network footage, documentary re-uploads, and
# retrospectives known to embed archival clips (not staged reenactments).
QUERIES = [
    "World Trade Center September 11 raw footage archive",
    "9/11 WTC2 collapse raw footage uncut",
    "9/11 WTC1 collapse raw footage uncut",
    "Naudet brothers 9/11 documentary footage",
    "CameraPlanet World Trade Center 2001",
    "NIST FOIA World Trade Center video release",
    "9/11 second plane impact multiple angles raw",
    "World Trade Center attack live news broadcast September 11 2001",
]


def _run_yt_dlp(query: str, n: int) -> list[dict]:
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-json",
        "--no-warnings",
        "--playlist-end",
        str(n),
        f"ytsearch{n}:{query}",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.warning("youtube: yt-dlp failed for query %r: %s", query, exc)
        return []
    if proc.returncode != 0:
        logger.warning(
            "youtube: yt-dlp exited %d for query %r: %s", proc.returncode, query, proc.stderr[:300]
        )
    entries = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _source_from_entry(entry: dict, *, query: str) -> Source | None:
    video_id = entry.get("id")
    if not video_id:
        return None
    return Source(
        id=f"yt-{video_id}",
        kind=SourceKind.VIDEO,
        url=entry.get("webpage_url")
        or entry.get("url")
        or f"https://www.youtube.com/watch?v={video_id}",
        archive="youtube",
        title=entry.get("title", "") or "",
        creator=entry.get("channel", "") or entry.get("uploader", "") or "",
        license="unknown",
        duration_s=entry.get("duration"),
        tags=["search:" + query],
        notes=f"view_count={entry.get('view_count')}"
        if entry.get("view_count") is not None
        else "",
    )


def harvest(
    queries: list[str] | None = None, results_per_query: int = RESULTS_PER_QUERY
) -> list[Source]:
    queries = queries if queries is not None else QUERIES
    by_id: dict[str, Source] = {}
    for query in queries:
        for entry in _run_yt_dlp(query, results_per_query):
            src = _source_from_entry(entry, query=query)
            if src is None:
                continue
            if src.id in by_id:
                # seen under an earlier query too -- keep both query tags
                by_id[src.id].tags = sorted(set(by_id[src.id].tags) | set(src.tags))
            else:
                by_id[src.id] = src
    logger.info("youtube harvest: %d unique videos across %d queries", len(by_id), len(queries))
    return list(by_id.values())
