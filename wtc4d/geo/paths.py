"""Filesystem locations used by the geo workstream.

Downloads are cached **outside** the repository (``$WTC4D_GEO_CACHE`` or
``~/.cache/wtc4d/geo``); only small derived products live in ``data/geo/``.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "geo"
"""Committed, small, versioned derived data (registries, corrections, GeoJSON)."""


def cache_dir() -> Path:
    """Directory for raw downloads. Never inside the repo."""
    env = os.environ.get("WTC4D_GEO_CACHE")
    root = Path(env) if env else Path.home() / ".cache" / "wtc4d" / "geo"
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_dir() -> Path:
    """Directory for large build outputs that must not be committed."""
    env = os.environ.get("WTC4D_GEO_BUILD")
    root = Path(env) if env else cache_dir() / "build"
    root.mkdir(parents=True, exist_ok=True)
    return root
