"""Shared Modal app, volume and container images for 911_4D compute jobs.

Every other module in ``infra/modal`` (and, eventually, ``kind: shell`` job
payloads from other workstreams) builds on the single ``app`` / ``volume``
/ image objects defined here so there is exactly one Modal App ("wtc4d")
and one data volume ("wtc4d-data") for the whole project.

Nothing in this module contacts the Modal API at import time: ``modal.App``,
``modal.Volume.from_name`` and ``modal.Image`` are all lazy client-side
descriptions.  They only talk to Modal when a function actually runs
(``.remote()``/``.spawn()``), or under ``modal run`` / ``modal deploy``.
That is what lets ``tests/test_infra_jobs.py`` import this module with
``MODAL_TOKEN_ID`` unset.

Volume layout (mounted at ``/data``): see ``data/README.md``.
    /data/raw/<source_id>/...        original downloads (immutable)
    /data/frames/<shot_id>/%06d.jpg  extracted frames
    /data/work/<job_id>/...          per-job log.txt + result.json
    /data/splats/<epoch_or_window>/  trained splats
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "wtc4d"
VOLUME_NAME = "wtc4d-data"
DATA_MOUNT = "/data"

DEFAULT_GPU = "A10G"
DEFAULT_TIMEOUT_S = 900  # 15 min; override per-job via JobSpec.timeout_s

# GPU choices offered to callers (GitHub Actions input, dispatch.py, gateway).
GPU_CHOICES = ("none", "T4", "L4", "A10G", "A100", "H100")

app = modal.App(APP_NAME)

# ``create_if_missing`` so the very first deploy / job run provisions the
# volume; afterwards this is just a reference to the existing one.
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Repo root, used only to decide what ``add_local_python_source`` mounts;
# never baked into the image (kept out of the Docker layer cache so editing
# wtc4d/** doesn't invalidate the dependency layers below).
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Shared CPU tooling: ffmpeg for frame extraction, yt-dlp/internetarchive for
# downloads, opencv for basic image ops.  Pin every version; bump deliberately.
_BASE_PIP_PACKAGES = [
    "numpy==1.26.4",
    "pydantic==2.9.2",
    "pyyaml==6.0.2",
    "opencv-python-headless==4.10.0.84",
    "yt-dlp==2024.10.22",
    "internetarchive==5.0.2",
    "requests==2.32.3",
]

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "ca-certificates")
    .pip_install(*_BASE_PIP_PACKAGES)
    # Mounted at container start (copy=False, the default): edits under
    # wtc4d/ do not invalidate the apt/pip layers above.
    .add_local_python_source("wtc4d")
)

# CUDA 12.4 wheel index for torch/torchvision (must match the base image's
# CUDA runtime below).
_TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu124"

# TODO(infra): LightGlue has no PyPI releases; pinned to a commit that
# existed on the ``main`` branch. GitHub access from this session is
# restricted to WalterGropius/911_4d, so this could not be re-verified
# against upstream HEAD — confirm/bump before the first real GPU job.
_LIGHTGLUE_COMMIT = "edb2b83"

gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "ffmpeg",
        "git",
        "ca-certificates",
        "build-essential",
        "libgl1",
        "libglib2.0-0",
        "libegl1",
        "libgles2",
        "libosmesa6",
        "colmap",
    )
    .pip_install(*_BASE_PIP_PACKAGES)
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        extra_index_url=_TORCH_CUDA_INDEX,
    )
    .pip_install(
        "gsplat==1.4.0",
        "pycolmap==0.6.1",
        "kornia==0.7.4",
        "trimesh==4.5.3",
        "pyrender==0.1.45",
        "pyopengl==3.1.7",
    )
    .pip_install(f"git+https://github.com/cvg/LightGlue.git@{_LIGHTGLUE_COMMIT}")
    .env({"PYOPENGL_PLATFORM": "egl"})
    .add_local_python_source("wtc4d")
)
