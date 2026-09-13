"""``wtc4d corpus`` CLI: harvest, download, shots, dedup.

Mounted automatically as a subcommand of the root CLI (see ``wtc4d/cli.py``).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from wtc4d.corpus import dedup as dedup_mod
from wtc4d.corpus import registry, store
from wtc4d.corpus import shots as shots_mod
from wtc4d.corpus.harvest import (
    archive_org,
    documentary,
    flickr,
    nist_foia,
    noaa_usgs,
    wikimedia,
    youtube,
)
from wtc4d.schema import Source

app = typer.Typer(
    no_args_is_help=True, help="Footage/photo discovery, cataloguing, download, dedup."
)
console = Console()
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

# Module-level option singletons (ruff B008: no function calls in defaults).
_VIDEO_OPTION = typer.Option(..., exists=True, help="Local video file to run shot detection on.")
_SOURCE_ID_OPTION = typer.Option(..., help="Source.id this video belongs to (see sources.jsonl).")
_IDS_OPTION = typer.Option(..., help="Comma-separated Source ids to download.")
_DEST_OPTION = typer.Option(..., help="Destination directory (never committed to git).")
_VIDEOS_OPTION = typer.Option(..., help="Local video/image files to hash, one per --videos flag.")
_SOURCE_IDS_OPTION = typer.Option(..., help="Source.id for each --videos path, same order.")


def _harvest_dispatch(collection_id: str, limit: int | None) -> list[Source]:
    """Explicit id -> harvester mapping (module ``harvest()`` signatures vary)."""
    if collection_id == "ia_911_tv_archive":
        return archive_org.harvest(max_items=limit)
    if collection_id == "ia_nist_foia_09_42":
        return nist_foia.harvest("ia_nist_foia_09_42")
    if collection_id == "ia_911datasets_mirror":
        return nist_foia.harvest("ia_911datasets_mirror")
    if collection_id == "wikimedia_commons_911":
        return wikimedia.harvest(limit_per_category=limit)
    if collection_id == "youtube_curated":
        return youtube.harvest(results_per_query=limit or youtube.RESULTS_PER_QUERY)
    if collection_id == "flickr_cc_911":
        return flickr.harvest()
    if collection_id == "noaa_wtc_aerial_lidar":
        return noaa_usgs.harvest("noaa_wtc_aerial_lidar")
    if collection_id == "usgs_landsat7_wtc":
        return noaa_usgs.harvest("usgs_landsat7_wtc")
    if collection_id == "commercial_satellite_ikonos_spot":
        return noaa_usgs.harvest("commercial_satellite_ikonos_spot")
    if collection_id == "documentary_pointers":
        return documentary.harvest()
    raise typer.BadParameter(f"no harvester wired up for collection '{collection_id}'")


@app.command()
def registry_list() -> None:
    """List the curated source registry (data/manifests/registry.yaml)."""
    table = Table(title="corpus source registry")
    for col in ("id", "archive", "harvest_method", "est_size"):
        table.add_column(col)
    for entry in registry.load_registry():
        table.add_row(entry.id, entry.archive, entry.harvest_method.value, entry.est_size)
    console.print(table)


@app.command()
def harvest(
    collection: str = typer.Option(
        "all", help="Registry collection id (see `corpus registry-list`), or 'all'."
    ),
    limit: int | None = typer.Option(
        None,
        help="Cap results per collection where the harvester supports it (mainly for smoke-testing).",
    ),
) -> None:
    """Run harvester(s) and merge results into data/manifests/sources*.jsonl."""
    reg = registry.by_id()
    if collection == "all":
        ids = list(reg)
    elif collection in reg:
        ids = [collection]
    else:
        raise typer.BadParameter(f"unknown collection '{collection}'; see `corpus registry-list`")

    existing = store.load_all_sources()
    console.print(f"[bold]{len(existing)}[/bold] sources currently on disk")

    all_new: list[Source] = []
    for cid in ids:
        console.print(f"harvesting [bold]{cid}[/bold] ...")
        try:
            new_sources = _harvest_dispatch(cid, limit)
        except Exception as exc:  # noqa: BLE001 -- keep harvesting other collections
            console.print(f"  [red]failed:[/red] {exc}")
            continue
        console.print(f"  {len(new_sources)} sources")
        all_new.extend(new_sources)

    merged = store.merge_sources(existing, all_new)
    written = store.save_sources(merged)
    console.print(
        f"[bold green]{len(merged)}[/bold green] total sources -> {[str(p) for p in written]}"
    )


@app.command()
def shots(
    video: Path = _VIDEO_OPTION,
    source_id: str = _SOURCE_ID_OPTION,
) -> None:
    """Detect shots in a local video and merge Shot records into shots.jsonl."""
    all_sources = store.load_all_sources()
    src = all_sources.get(source_id)
    time_hint = src.time_hint if src else None
    detected = shots_mod.detect_shots(video, source_id, source_time_hint=time_hint)
    merged = store.merge_shots(detected)
    console.print(f"{len(detected)} shots detected, {len(merged)} total in shots.jsonl")
    for s in detected:
        motion_line = f"  {s.id}: frames [{s.start_frame},{s.end_frame}) motion={s.camera_motion}"
        if s.quality_score is not None:
            motion_line += f" q={s.quality_score:.2f}"
        console.print(motion_line)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_one(src: Source, dest: Path) -> Source:
    import datetime as dt
    import subprocess

    dest.mkdir(parents=True, exist_ok=True)
    if src.archive == "youtube":
        subprocess.run(
            ["yt-dlp", "--continue", "-o", str(dest / f"{src.id}.%(ext)s"), src.url],
            check=True,
            timeout=600,
        )
        matches = list(dest.glob(f"{src.id}.*"))
    elif src.archive in ("archive.org", "nist_foia"):
        import internetarchive as ia

        identifier = src.id.removeprefix("ia-")
        item_dir = dest / identifier
        ia.download(identifier, destdir=str(dest), verbose=True, ignore_existing=True)
        matches = list(item_dir.glob("*")) if item_dir.exists() else []
    else:
        raise ValueError(f"no download method wired up for archive '{src.archive}'")

    matches = [m for m in matches if m.is_file()]
    if not matches:
        raise RuntimeError(f"download produced no files for {src.id}")
    largest = max(matches, key=lambda p: p.stat().st_size)

    return src.model_copy(
        update={
            "checksum_sha256": _sha256_file(largest),
            "bytes": largest.stat().st_size,
            "retrieved_at": dt.datetime.now(dt.UTC),
        }
    )


@app.command()
def download(
    ids: str = _IDS_OPTION,
    dest: Path = _DEST_OPTION,
) -> None:
    """Download 1+ sources (yt-dlp for youtube, `ia` for archive.org/nist_foia).

    Records sha256 + byte count back into the manifest. Intended for small
    smoke-test downloads (2-3 items), not bulk fetches -- run those on Modal.
    """
    all_sources = store.load_all_sources()
    updated = []
    for sid in (s.strip() for s in ids.split(",") if s.strip()):
        src = all_sources.get(sid)
        if src is None:
            console.print(f"[yellow]skip[/yellow] {sid}: not found in manifest")
            continue
        console.print(f"downloading [bold]{sid}[/bold] ...")
        try:
            updated.append(_download_one(src, dest))
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [red]failed:[/red] {exc}")

    if updated:
        merged = store.merge_sources(all_sources, updated)
        store.save_sources(merged)
    console.print(
        f"[bold green]{len(updated)}[/bold green] sources downloaded and updated in manifest"
    )


@app.command()
def dedup(
    videos: list[Path] = _VIDEOS_OPTION,
    source_ids: list[str] = _SOURCE_IDS_OPTION,
    threshold: int = dedup_mod.DEFAULT_HAMMING_THRESHOLD,
) -> None:
    """Perceptual-hash local files and link near-duplicates in the manifest."""
    if len(videos) != len(source_ids):
        raise typer.BadParameter("--videos and --source-ids must have the same count")

    hashes_by_source = {}
    for path, sid in zip(videos, source_ids, strict=True):
        hashes_by_source[sid] = dedup_mod.hash_frames(path)

    links = dedup_mod.find_duplicates(hashes_by_source, threshold=threshold)
    console.print(f"{len(links)} near-duplicate link(s) found")

    all_sources = store.load_all_sources()
    dedup_mod.annotate_duplicates(all_sources, links)
    store.save_sources(all_sources)
    for link in links:
        console.print(f"  {link.source_id} <-> {link.other_id} (distance {link.distance})")
