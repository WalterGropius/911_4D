"""Generic Modal job runner: one entrypoint, many job ``kind``s.

This is how every workstream uses GPU/CPU compute without touching
``infra/``: submit a JSON job spec (see :class:`JobSpec`) with
``kind: "shell"`` and a ``module:function`` entrypoint, and it runs inside
the shared images defined in :mod:`infra.modal.common`.

Dispatch (see ``infra/README.md`` for the full runbook):
    - GitHub Actions UI / API: ``.github/workflows/modal-job.yml``
    - ``python -m infra.dispatch --spec spec.json --gpu A10G``
    - locally (needs a working Modal connection): ``modal run
      infra/modal/jobs.py --spec '{"kind": "smoke_test_gpu"}'``

Every job writes ``/data/work/<job_id>/log.txt`` and
``/data/work/<job_id>/result.json`` on the shared volume, and the modal
function itself returns the same result dict.

Nothing here contacts Modal at import time: building ``JobSpec`` and
importing this module is plain Python.  The only network- or GPU-touching
code paths are inside the ``@app.function``-decorated functions and the
``@app.local_entrypoint()``, which only run under ``modal run``.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import shutil
import subprocess
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from infra.modal.common import (
    DEFAULT_GPU,
    DEFAULT_TIMEOUT_S,
    GPU_CHOICES,
    app,
    base_image,
    gpu_image,
    volume,
)

JobKind = Literal[
    "smoke_test_gpu",
    "download_sources",
    "extract_frames",
    "shell",
    "run_colmap",
]

GPU_KINDS = {"smoke_test_gpu", "run_colmap"}  # kinds that always need the gpu_image

GpuChoice = Literal["none", "T4", "L4", "A10G", "A100", "H100"]
assert set(GpuChoice.__args__) == set(GPU_CHOICES), "GpuChoice must mirror common.GPU_CHOICES"


class JobSpec(BaseModel):
    """Validated job payload. ``params`` is kind-specific; see README."""

    kind: JobKind
    job_id: str | None = None
    gpu: GpuChoice = "none"
    timeout_s: int = Field(default=DEFAULT_TIMEOUT_S, gt=0)
    params: dict[str, Any] = Field(default_factory=dict)

    def resolved_job_id(self) -> str:
        return self.job_id or f"{self.kind}-{uuid.uuid4().hex[:10]}"

    def needs_gpu(self) -> bool:
        if self.kind in GPU_KINDS:
            return True
        if self.kind == "shell":
            return bool(self.params.get("gpu"))
        return False


def data_root() -> Path:
    """Root of the mounted volume. Overridable via ``WTC4D_DATA_ROOT`` so
    tests never touch a real ``/data``."""
    return Path(os.environ.get("WTC4D_DATA_ROOT", "/data"))


def _run_subprocess(cmd: list[str], *, log: logging.Logger, cwd: str | None = None) -> dict:
    """Thin wrapper around subprocess.run so tests can monkeypatch it
    without a real ffmpeg/yt-dlp/ia/nvidia-smi binary on PATH."""
    log.info("$ %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        log.error("command not found: %s", exc)
        return {"returncode": 127, "stdout": "", "stderr": str(exc)}
    if proc.stdout:
        log.info(proc.stdout)
    if proc.stderr:
        log.info(proc.stderr)
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def list_dir(path: str = "") -> list[str]:
    """List a directory relative to the volume root. Used by the
    ``list_volume`` modal function and directly by tests."""
    root = data_root()
    target = (root / path) if path else root
    if not target.exists():
        return []
    return sorted(str(p.relative_to(root)) for p in target.iterdir())


# --------------------------------------------------------------------------
# Job kind handlers. Each takes (spec, work_dir, log) and returns a result
# dict (JSON-serialisable). They must not import Modal or reach outside
# data_root() so they stay unit-testable.
# --------------------------------------------------------------------------


def _handle_smoke_test_gpu(spec: JobSpec, work_dir: Path, log: logging.Logger) -> dict:
    result: dict[str, Any] = {}
    smi = _run_subprocess(["nvidia-smi"], log=log)
    result["nvidia_smi_returncode"] = smi["returncode"]
    result["nvidia_smi_output"] = smi["stdout"] or smi["stderr"]

    try:
        import torch  # noqa: PLC0415

        result["torch_installed"] = True
        result["torch_version"] = torch.__version__
        result["cuda_available"] = bool(torch.cuda.is_available())
        if result["cuda_available"]:
            result["device_name"] = torch.cuda.get_device_name(0)
            x = torch.randn(1024, 1024, device="cuda")
            result["cuda_matmul_ok"] = bool(torch.allclose(x @ x.T, x @ x.T))
    except ImportError:
        result["torch_installed"] = False
        result["cuda_available"] = False

    marker = work_dir / "smoke_test_marker.txt"
    marker.write_text(f"smoke test ok, job_id={spec.resolved_job_id()}\n")
    result["marker_path"] = str(marker)
    result["ok"] = bool(result.get("cuda_available"))
    return result


def _source_kind(url: str) -> str:
    if "archive.org" in url:
        return "archive.org"
    if "youtube.com" in url or "youtu.be" in url:
        return "youtube"
    return "other"


def _handle_download_sources(spec: JobSpec, work_dir: Path, log: logging.Logger) -> dict:
    items = spec.params.get("items", [])
    if not items:
        raise ValueError("download_sources requires params.items: [{id, url}, ...]")

    raw_root = data_root() / "raw"
    downloaded: list[dict] = []

    try:
        from wtc4d.corpus import download as corpus_download  # type: ignore  # noqa: PLC0415

        log.info("wtc4d.corpus importable; delegating to corpus_download()")
        for item in items:
            dest = raw_root / item["id"]
            dest.mkdir(parents=True, exist_ok=True)
            corpus_download(item["url"], dest)
            downloaded.append({"id": item["id"], "dest": str(dest), "method": "wtc4d.corpus"})
    except ImportError:
        log.info("wtc4d.corpus not importable; falling back to yt-dlp / ia directly")
        for item in items:
            dest = raw_root / item["id"]
            dest.mkdir(parents=True, exist_ok=True)
            url = item["url"]
            kind = item.get("kind") or _source_kind(url)
            if kind == "archive.org":
                identifier = item.get("identifier") or url.rstrip("/").rsplit("/", 1)[-1]
                cmd = ["ia", "download", identifier, "--destdir", str(dest)]
            else:
                cmd = ["yt-dlp", "-o", str(dest / "%(id)s.%(ext)s"), url]
            res = _run_subprocess(cmd, log=log)
            downloaded.append(
                {
                    "id": item["id"],
                    "dest": str(dest),
                    "method": kind,
                    "returncode": res["returncode"],
                }
            )

    ok = all(d.get("returncode", 0) == 0 for d in downloaded)
    return {"ok": ok, "items": downloaded}


def _handle_extract_frames(spec: JobSpec, work_dir: Path, log: logging.Logger) -> dict:
    params = spec.params
    shot_id = params.get("shot_id")
    source_path = params.get("source_path")
    if not shot_id or not source_path:
        raise ValueError("extract_frames requires params.shot_id and params.source_path")

    fps = params.get("fps", 5)
    crop = params.get("crop")  # "w:h:x:y" (ffmpeg crop filter syntax) or None
    deinterlace = bool(params.get("deinterlace", True))

    out_dir = data_root() / "frames" / shot_id
    out_dir.mkdir(parents=True, exist_ok=True)

    filters = []
    if deinterlace:
        filters.append("yadif")
    filters.append(f"fps={fps}")
    if crop:
        filters.append(f"crop={crop}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_path),
        "-vf",
        ",".join(filters),
        "-qscale:v",
        "2",
        str(out_dir / "%06d.jpg"),
    ]
    res = _run_subprocess(cmd, log=log)
    frame_count = len(list(out_dir.glob("*.jpg"))) if out_dir.exists() else 0
    return {
        "ok": res["returncode"] == 0,
        "shot_id": shot_id,
        "out_dir": str(out_dir),
        "frame_count": frame_count,
        "returncode": res["returncode"],
    }


def _handle_shell(spec: JobSpec, work_dir: Path, log: logging.Logger) -> dict:
    """Import ``module`` and call ``function(job_spec)`` where ``job_spec``
    is everything in ``params`` except ``entrypoint``/``gpu`` themselves.
    This is the plug-in point other workstreams use, e.g.
    ``{"kind": "shell", "params": {"entrypoint": "wtc4d.recon.job:run",
    "scene_id": "E2", ...}}`` calls ``wtc4d.recon.job.run({"scene_id":
    "E2", ...})``.
    """
    entrypoint = spec.params.get("entrypoint")
    if not entrypoint or ":" not in entrypoint:
        raise ValueError("shell requires params.entrypoint = 'module.path:function_name'")
    module_name, func_name = entrypoint.split(":", 1)
    log.info("shell: importing %s, calling %s", module_name, func_name)
    mod = importlib.import_module(module_name)
    func = getattr(mod, func_name)
    sub_spec = {k: v for k, v in spec.params.items() if k not in ("entrypoint", "gpu")}
    out = func(sub_spec)
    if out is None:
        return {"ok": True}
    if isinstance(out, dict):
        out.setdefault("ok", True)
        return out
    return {"ok": True, "return_value": out}


def _handle_run_colmap(spec: JobSpec, work_dir: Path, log: logging.Logger) -> dict:
    # Placeholder: the `recon` workstream owns the actual COLMAP pipeline and
    # plugs it in via `kind: shell` (entrypoint `wtc4d.recon...:run`). This
    # kind exists so the job-spec surface is stable; it currently only
    # validates inputs and reports what colmap binary is available.
    which = shutil.which("colmap")
    return {
        "ok": False,
        "status": "not_implemented",
        "colmap_binary": which,
        "notes": "run_colmap is a placeholder; use kind=shell with a recon entrypoint instead.",
    }


_HANDLERS = {
    "smoke_test_gpu": _handle_smoke_test_gpu,
    "download_sources": _handle_download_sources,
    "extract_frames": _handle_extract_frames,
    "shell": _handle_shell,
    "run_colmap": _handle_run_colmap,
}


def run_job(spec: dict) -> dict:
    """Validate ``spec``, dispatch on ``kind``, log to and write
    ``result.json`` under ``/data/work/<job_id>/``, and return the result.

    Never raises for a job-level failure: exceptions from the handler are
    caught and returned as ``{"ok": False, "error": ...}`` so a bad job spec
    doesn't crash the whole GitHub Actions step before logs are captured.
    Programmer errors in the *spec itself* (failing JobSpec validation)
    still raise, since there is no job_id to log against yet.
    """
    job_spec = JobSpec.model_validate(spec)
    job_id = job_spec.resolved_job_id()

    work_dir = data_root() / "work" / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "log.txt"

    log = logging.getLogger(f"wtc4d.job.{job_id}")
    log.setLevel(logging.INFO)
    log.propagate = False
    for h in list(log.handlers):
        log.removeHandler(h)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    log.addHandler(stream_handler)

    log.info("starting job kind=%s job_id=%s params=%s", job_spec.kind, job_id, job_spec.params)
    try:
        handler = _HANDLERS[job_spec.kind]
        result = handler(job_spec, work_dir, log)
        result.setdefault("ok", True)
    except Exception as exc:  # noqa: BLE001 - job failures are data, not crashes
        log.error("job failed: %s\n%s", exc, traceback.format_exc())
        result = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
    finally:
        file_handler.close()

    result["job_id"] = job_id
    result["kind"] = job_spec.kind
    (work_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


# --------------------------------------------------------------------------
# Modal functions. Two variants (cpu / gpu) rather than one function with a
# per-call gpu type, since Modal's `gpu=` is set at function-definition time;
# `run_job_gpu.with_options(gpu=..., timeout=...)` (used by the local
# entrypoint and dispatch.py) covers the requested GPU type / timeout without
# duplicating handler code.
# --------------------------------------------------------------------------


@app.function(image=base_image, volumes={"/data": volume}, timeout=DEFAULT_TIMEOUT_S)
def run_job_cpu(spec: dict) -> dict:
    return run_job(spec)


@app.function(
    image=gpu_image,
    gpu=DEFAULT_GPU,
    volumes={"/data": volume},
    timeout=DEFAULT_TIMEOUT_S,
)
def run_job_gpu(spec: dict) -> dict:
    return run_job(spec)


@app.function(image=base_image, volumes={"/data": volume})
def list_volume(path: str = "") -> list[str]:
    return list_dir(path)


def _dispatch_for(job_spec: JobSpec):
    """Pick the right modal Function + apply gpu/timeout overrides.

    ``job_spec.gpu`` is already constrained to :data:`GpuChoice` by pydantic,
    so there's nothing left to validate here.
    """
    gpu = job_spec.gpu if job_spec.gpu != "none" else None
    if job_spec.needs_gpu() or gpu:
        return run_job_gpu.with_options(gpu=gpu or DEFAULT_GPU, timeout=job_spec.timeout_s)
    return run_job_cpu.with_options(timeout=job_spec.timeout_s)


@app.local_entrypoint()
def main(spec: str, gpu: str = "none", timeout_min: int = 15) -> None:
    """``modal run infra/modal/jobs.py --spec '<json>' --gpu A10G --timeout-min 30``

    Prints the result as JSON on the last line, prefixed with
    ``RESULT_JSON:`` so ``.github/workflows/modal-job.yml`` can extract it
    from the raw ``modal run`` output without needing a separate volume-get
    step.
    """
    payload = json.loads(spec)
    payload.setdefault("gpu", gpu)
    payload.setdefault("timeout_s", timeout_min * 60)
    job_spec = JobSpec.model_validate(payload)

    fn = _dispatch_for(job_spec)
    result = fn.remote(payload)
    print("RESULT_JSON:" + json.dumps(result))
