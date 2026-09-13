"""Job entrypoint for the infra runner, plus the viewer manifest writer.

``infra`` executes jobs as ``module:function(spec)`` inside a GPU image; this
module is that function for everything ``recon`` does::

    run({"kind": "train_static", "config": {...}}) -> {"ok": True, ...}

Every job returns a JSON-serialisable dict: paths written, headline metrics,
and the resolved backend, so a GitHub Actions log is enough to tell whether a
run used the GPU kernels or silently fell back to the CPU reference.

Example specs live in ``wtc4d/recon/jobs/``.  ``kind: "toy_static"`` and
``kind: "toy_dynamic"`` need no data at all and run in under a minute on CPU:
use them as the smoke test that a new image can train anything before
launching a real epoch.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import torch

from wtc4d import timeline, world
from wtc4d.schema.scene import CameraRef, SceneManifest, SplatAsset, TimelineEvent

__all__ = ["load_spec", "run", "write_scene_manifest"]

JOBS_DIR = Path(__file__).parent / "jobs"


def load_spec(path: str | Path) -> dict:
    """Load a job spec from JSON (or YAML if it ends in ``.yaml``/``.yml``)."""
    path = Path(path)
    if path.suffix in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(path.read_text()) or {}
    return json.loads(path.read_text())


def run(spec: dict) -> dict:
    """Run one recon job.  ``spec["kind"]`` selects what happens."""
    kind = spec.get("kind", "train_static")
    handlers = {
        "train_static": _run_train_static,
        "train_dynamic": _run_train_dynamic,
        "toy_static": _run_toy_static,
        "toy_dynamic": _run_toy_dynamic,
        "export_manifest": _run_export_manifest,
        "env": _run_env,
    }
    if kind not in handlers:
        raise ValueError(f"unknown recon job kind {kind!r}; expected one of {sorted(handlers)}")
    result = handlers[kind](spec)
    result.setdefault("kind", kind)
    result.setdefault("ok", True)
    result.setdefault("env", _env_summary())
    return result


def _env_summary() -> dict:
    from .backends import available_backends, resolve

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "backends": available_backends(),
        "selected_backend": resolve(None),
    }


def _run_env(spec: dict) -> dict:
    return {}


def _config_from(spec: dict, key: str = "config"):
    """A spec may inline its config or point at a YAML file."""
    value = spec.get(key)
    if isinstance(value, str):
        import yaml

        return yaml.safe_load(Path(value).read_text()) or {}
    return dict(value or {})


def _run_train_static(spec: dict) -> dict:
    from .train_static import TrainStaticConfig, train_static

    cfg = TrainStaticConfig.model_validate(_config_from(spec))
    result = train_static(cfg, iterations=spec.get("iterations"))
    out = {
        "stats_path": str(result.stats_path) if result.stats_path else None,
        "ply_path": str(result.ply_path) if result.ply_path else None,
        "npz_path": str(result.npz_path) if result.npz_path else None,
        "n_gaussians": result.gaussians.n,
        "psnr_mean": result.stats.get("psnr_mean"),
        "wall_seconds": result.stats.get("wall_seconds"),
    }
    if spec.get("manifest"):
        out["manifest_path"] = str(_manifest_for(spec, result))
    return out


def _run_train_dynamic(spec: dict) -> dict:
    from .data import ReconDataset
    from .dynamic import DynamicTrainConfig, train_dynamic
    from .gaussians import Gaussians
    from .init import init_gaussians

    cfg = DynamicTrainConfig.model_validate(_config_from(spec))
    data_spec = spec.get("data", {})
    window = None
    if cfg.window_id:
        window = next((w for w in timeline.DYNAMIC_WINDOWS if w.id == cfg.window_id), None)
        if window is None:
            raise ValueError(f"unknown dynamic window {cfg.window_id!r}")
        cfg.t_start = cfg.t_start if cfg.t_start is not None else window.t_start
        cfg.t_end = cfg.t_end if cfg.t_end is not None else window.t_end

    ds = ReconDataset.from_paths(
        data_spec["poses_path"],
        data_spec.get("frames_dir"),
        data_spec.get("masks_dir"),
        data_spec.get("times_path"),
        downscale=data_spec.get("downscale", 1.0),
        device=cfg.device,
    )
    if cfg.t_start is not None and cfg.t_end is not None:
        ds = ds.filter_time(cfg.t_start, cfg.t_end)
    train_ds, val_ds = ds.split(data_spec.get("holdout_every", 8))

    init_spec = spec.get("init", {})
    if init_spec.get("ply_path"):
        canonical = Gaussians.load_ply(init_spec["ply_path"], device=cfg.device)
    else:
        canonical = init_gaussians(
            mesh_points=init_spec.get("mesh_points", 100_000),
            epoch_id=init_spec.get("epoch_id"),
            procedural_t=init_spec.get("procedural_t", cfg.t_start),
            procedural_ply=init_spec.get("procedural_ply"),
            dedup_radius=init_spec.get("dedup_radius", 0.5),
            device=cfg.device,
        )

    result = train_dynamic(cfg, train_ds, canonical, val_dataset=val_ds)
    out_dir = Path(cfg.out_dir) / cfg.name
    exported = []
    for t in spec.get("export_times", []):
        g = result.model.export_timestep(float(t))
        exported.append(str(g.save_ply(out_dir / f"{cfg.name}_t{float(t):.2f}.ply")))
    return {
        "stats_path": str(result.stats_path) if result.stats_path else None,
        "n_frames": result.stats.get("n_frames"),
        "n_gaussians": result.stats.get("n_gaussians"),
        "val": result.stats.get("val"),
        "exported_ply": exported,
        "wall_seconds": result.stats.get("wall_seconds"),
    }


def _run_toy_static(spec: dict) -> dict:
    from .synthetic import toy_static_scene
    from .train_static import TrainStaticConfig, train_static

    scene = toy_static_scene(
        n_cameras=spec.get("n_cameras", 6),
        width=spec.get("width", 48),
        height=spec.get("height", 36),
        n_points=spec.get("n_points", 220),
        appearance_jitter=spec.get("appearance_jitter", 0.0),
        with_masks=spec.get("with_masks", False),
    )
    cfg = TrainStaticConfig.model_validate(_config_from(spec))
    cfg.output.out_dir = spec.get("out_dir", cfg.output.out_dir)
    cfg.output.name = spec.get("name", "toy_static")
    train_ds, val_ds = scene.dataset.split(spec.get("holdout_every", 6))
    result = train_static(
        cfg,
        dataset=train_ds,
        val_dataset=val_ds,
        gaussians=scene.init,
        iterations=spec.get("iterations", 200),
    )
    return {
        "psnr_mean": result.stats.get("psnr_mean"),
        "n_gaussians": result.gaussians.n,
        "stats_path": str(result.stats_path) if result.stats_path else None,
        "first_psnr": result.history[0]["psnr"] if result.history else None,
        "last_psnr": result.history[-1]["psnr"] if result.history else None,
    }


def _run_toy_dynamic(spec: dict) -> dict:
    from .dynamic import DynamicTrainConfig, train_dynamic
    from .synthetic import toy_dynamic_scene

    scene = toy_dynamic_scene(
        n_cameras=spec.get("n_cameras", 6),
        n_times=spec.get("n_times", 10),
        width=spec.get("width", 40),
        height=spec.get("height", 30),
        n_points=spec.get("n_points", 140),
    )
    cfg = DynamicTrainConfig.model_validate(_config_from(spec))
    cfg.out_dir = spec.get("out_dir", cfg.out_dir)
    cfg.name = spec.get("name", "toy_dynamic")
    result = train_dynamic(
        cfg,
        scene.dataset,
        scene.init,
        iterations=spec.get("iterations", 150),
    )
    return {
        "stats_path": str(result.stats_path) if result.stats_path else None,
        "first_psnr": result.history[0]["psnr"] if result.history else None,
        "last_psnr": result.history[-1]["psnr"] if result.history else None,
        "n_gaussians": result.stats.get("n_gaussians"),
    }


def _run_export_manifest(spec: dict) -> dict:
    path = write_scene_manifest(
        spec["out_path"],
        assets=spec.get("assets", []),
        cameras=spec.get("cameras", []),
        name=spec.get("name", "911_4D"),
        t_min=spec.get("t_min"),
        t_max=spec.get("t_max"),
        notes=spec.get("notes", ""),
    )
    return {"manifest_path": str(path)}


def _manifest_for(spec: dict, result) -> Path:
    """Write a one-asset manifest next to a finished static run."""
    m = dict(spec.get("manifest", {}))
    epoch_id = result.stats.get("epoch_id")
    epoch = timeline.EPOCHS_BY_ID.get(epoch_id) if epoch_id else None
    asset = {
        "id": m.get("asset_id", result.stats.get("name", "static")),
        "url": m.get("url") or (Path(result.ply_path).name if result.ply_path else ""),
        "format": "ply",
        "t_start": m.get("t_start", epoch.t_start if epoch else 0.0),
        "t_end": m.get("t_end", epoch.t_end if epoch else 0.0),
        "kind": "static",
        "layer": "scene",
        "epoch_id": epoch_id,
        "notes": f"{result.stats.get('n_gaussians')} gaussians, PSNR {result.stats.get('psnr_mean')}",
    }
    out_path = m.get("out_path") or str(Path(result.stats_path).with_name("scene.json"))
    return write_scene_manifest(out_path, assets=[asset], cameras=m.get("cameras", []))


def write_scene_manifest(
    out_path: str | Path,
    assets: list[dict[str, Any]],
    cameras: list[dict[str, Any]] | None = None,
    *,
    name: str = "911_4D",
    t_min: float | None = None,
    t_max: float | None = None,
    events: list[dict] | None = None,
    notes: str = "",
) -> Path:
    """Write a :class:`~wtc4d.schema.scene.SceneManifest` for the web viewer.

    Times default to the span of the assets; events default to the canonical
    anchors from :mod:`wtc4d.timeline` (impacts and collapses), which is what
    the viewer's scrubber marks.
    """
    asset_models = [SplatAsset.model_validate(a) for a in assets]
    camera_models = [CameraRef.model_validate(c) for c in (cameras or [])]
    if t_min is None:
        t_min = min((a.t_start for a in asset_models), default=timeline.WTC1_IMPACT.t - 600)
    if t_max is None:
        t_max = max((a.t_end for a in asset_models), default=timeline.WTC1_COLLAPSE.t + 600)
    if events is None:
        event_models = [
            TimelineEvent(id=e.id, name=e.name, t=e.t, sigma=e.sigma)
            for e in timeline.EVENTS
            if t_min <= e.t <= t_max
        ]
    else:
        event_models = [TimelineEvent.model_validate(e) for e in events]

    manifest = SceneManifest(
        name=name,
        world_origin=world.WORLD_ORIGIN,
        t_min=float(t_min),
        t_max=float(t_max),
        events=event_models,
        assets=asset_models,
        cameras=camera_models,
        notes=notes,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    return out_path
