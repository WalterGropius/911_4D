"""JSONL persistence for corpus manifests (``Source``, ``Shot`` records).

Layout under ``data/manifests/``:

- ``sources.jsonl`` — the canonical catalogue, one JSON object per line,
  sorted by ``id`` for stable diffs.
- ``sources-<archive>.jsonl`` — used instead of the monolithic file once the
  catalogue would exceed :data:`MAX_MANIFEST_BYTES`; one file per ``archive``
  value (``archive_org``, ``wikimedia``, ...).
- ``shots.jsonl`` — ``Shot`` records, same merge-by-id convention.

Harvesters never overwrite blindly: :func:`merge_sources` combines new
records with whatever is already on disk, keyed by ``Source.id``, so running
a harvester twice is idempotent and re-running one collection does not
disturb another.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from wtc4d.schema import Shot, Source

MAX_MANIFEST_BYTES = 30_000_000

_REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = _REPO_ROOT / "data" / "manifests"
SOURCES_PATH = MANIFEST_DIR / "sources.jsonl"
SHOTS_PATH = MANIFEST_DIR / "shots.jsonl"

ModelT = TypeVar("ModelT", bound=BaseModel)


def _read_jsonl(path: Path, model: type[ModelT]) -> list[ModelT]:
    if not path.exists():
        return []
    out: list[ModelT] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(model.model_validate_json(line))
    return out


def _write_jsonl(path: Path, records: Iterable[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [r.model_dump_json(exclude_none=False) for r in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_sources(path: Path) -> list[Source]:
    return _read_jsonl(path, Source)


def load_all_sources(manifest_dir: Path | None = None) -> dict[str, Source]:
    """Load every ``Source`` currently on disk, merged by id.

    Reads the canonical ``sources.jsonl`` if present, plus any per-archive
    ``sources-<archive>.jsonl`` split files (the two layouts are mutually
    exclusive in normal operation but both are read defensively).
    """
    manifest_dir = manifest_dir if manifest_dir is not None else MANIFEST_DIR
    by_id: dict[str, Source] = {}
    for path in sorted(manifest_dir.glob("sources*.jsonl")):
        for src in read_sources(path):
            by_id[src.id] = src
    return by_id


def merge_sources(existing: dict[str, Source], new: Iterable[Source]) -> dict[str, Source]:
    """Merge ``new`` records into ``existing`` (keyed by id), new wins."""
    merged = dict(existing)
    for src in new:
        merged[src.id] = src
    return merged


def save_sources(
    by_id: dict[str, Source], manifest_dir: Path | None = None, max_bytes: int = MAX_MANIFEST_BYTES
) -> list[Path]:
    """Write the merged catalogue to disk, splitting per-archive if large.

    Returns the list of files written. Clears whichever layout (monolithic
    vs. split) is *not* used, so the manifest directory never carries stale
    duplicates of a record.
    """
    manifest_dir = manifest_dir if manifest_dir is not None else MANIFEST_DIR
    ordered = [by_id[k] for k in sorted(by_id)]
    monolithic_size = sum(len(r.model_dump_json()) + 1 for r in ordered)

    manifest_dir.mkdir(parents=True, exist_ok=True)
    for stale in manifest_dir.glob("sources*.jsonl"):
        stale.unlink()

    if monolithic_size <= max_bytes:
        path = manifest_dir / "sources.jsonl"
        _write_jsonl(path, ordered)
        return [path]

    by_archive: dict[str, list[Source]] = {}
    for src in ordered:
        by_archive.setdefault(src.archive, []).append(src)
    written = []
    for archive in sorted(by_archive):
        path = manifest_dir / f"sources-{archive}.jsonl"
        _write_jsonl(path, by_archive[archive])
        written.append(path)
    return written


def read_shots(path: Path | None = None) -> list[Shot]:
    path = path if path is not None else SHOTS_PATH
    return _read_jsonl(path, Shot)


def merge_shots(new: Iterable[Shot], path: Path | None = None) -> list[Shot]:
    path = path if path is not None else SHOTS_PATH
    by_id = {s.id: s for s in read_shots(path)}
    for shot in new:
        by_id[shot.id] = shot
    ordered = [by_id[k] for k in sorted(by_id)]
    _write_jsonl(path, ordered)
    return ordered
