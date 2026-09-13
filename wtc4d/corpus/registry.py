"""Curated registry of source collections (``data/manifests/registry.yaml``).

This is the human-curated map of *where to look*; ``wtc4d/corpus/harvest/``
turns each ``api``-method entry into ``Source`` records. See
``wtc4d/corpus/README.md`` for coverage stats and next steps.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel

_REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = _REPO_ROOT / "data" / "manifests" / "registry.yaml"


class HarvestMethod(StrEnum):
    API = "api"  # a wtc4d/corpus/harvest/*.py module can (re-)run this
    MANUAL = "manual"  # catalogued by hand; no scriptable API found
    DOCUMENTED = "documented"  # pointer only, e.g. licensed commercial data


class RegistryCollection(BaseModel):
    """One entry in the curated source registry."""

    id: str
    name: str
    archive: str
    url: str
    license: str
    harvest_method: HarvestMethod
    harvester: str | None = None  # dotted path to a wtc4d.corpus.harvest module
    est_size: str = ""
    notes: str = ""


def load_registry(path: Path = REGISTRY_PATH) -> list[RegistryCollection]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [RegistryCollection.model_validate(entry) for entry in data]


def by_id(path: Path = REGISTRY_PATH) -> dict[str, RegistryCollection]:
    return {c.id: c for c in load_registry(path)}
