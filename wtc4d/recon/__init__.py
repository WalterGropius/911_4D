"""Gaussian splat reconstruction: static splats per epoch and 4D dynamic windows.

See ``wtc4d/recon/README.md`` for the design and ``docs/recon_4d_design.md``
for the 4D survey and decision.  Conventions are the project ones: world =
ENU metres at ``wtc4d.world.WORLD_ORIGIN``, camera axes OpenCV, poses
camera-to-world, times in project seconds.

Submodules are imported lazily so that ``import wtc4d.recon`` stays cheap and
so that a missing optional dependency (torch, trimesh) surfaces where it is
used rather than at import time.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = [
    "Gaussians",
    "Layer",
    "ReconDataset",
    "backends",
    "cpu_raster",
    "dynamic",
    "evaluate",
    "job",
    "render",
    "train_static",
]

_LAZY_MODULES = {
    "backends",
    "cli",
    "colmap",
    "conventions",
    "cpu_raster",
    "data",
    "dynamic",
    "evaluate",
    "gaussians",
    "init",
    "job",
    "nerfstudio",
    "synthetic",
    "train_static",
}

_LAZY_ATTRS = {
    "Gaussians": ("gaussians", "Gaussians"),
    "Layer": ("gaussians", "Layer"),
    "ReconDataset": ("data", "ReconDataset"),
    "render": ("backends", "render"),
}

if TYPE_CHECKING:  # pragma: no cover
    from .backends import render
    from .data import ReconDataset
    from .gaussians import Gaussians, Layer


def __getattr__(name: str):
    if name in _LAZY_MODULES:
        return importlib.import_module(f".{name}", __name__)
    if name in _LAZY_ATTRS:
        module, attr = _LAZY_ATTRS[name]
        return getattr(importlib.import_module(f".{module}", __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | _LAZY_MODULES | set(_LAZY_ATTRS))
