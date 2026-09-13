"""``wtc4d recon ...`` -- training, conversion and inspection commands.

Mounted automatically by :mod:`wtc4d.cli` when this package imports.
"""

# ruff: noqa: B008 -- `typer.Option(...)`/`typer.Argument(...)` as a parameter
# default is typer's own required idiom (it introspects the call to build the
# CLI), not the mutable-default footgun B008 exists for.

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(
    no_args_is_help=True, help="Gaussian splat reconstruction (static per epoch, 4D windows)"
)


@app.command()
def info() -> None:
    """Show the available rasteriser backends and the device situation."""
    import torch
    from rich import print as rprint

    from .backends import available_backends, resolve

    rprint(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    rprint(f"backends: {', '.join(available_backends())}")
    rprint(f"selected: [bold]{resolve(None)}[/bold]")
    if not torch.cuda.is_available():
        rprint(
            "[yellow]No CUDA device: the pure-torch reference rasteriser will be used. "
            "It is correct but O(N x pixels) -- fine for tests, not for an epoch.[/yellow]"
        )


@app.command("train-static")
def train_static_cmd(
    config: Path = typer.Option(..., "--config", "-c", help="YAML TrainStaticConfig"),
    iterations: int | None = typer.Option(None, help="override optim.iterations"),
) -> None:
    """Train one static per-epoch splat."""
    from rich import print as rprint

    from .train_static import TrainStaticConfig, train_static

    cfg = TrainStaticConfig.from_yaml(config)
    result = train_static(cfg, iterations=iterations)
    rprint(
        f"[green]done[/green] {result.gaussians.n} gaussians, "
        f"PSNR {result.stats.get('psnr_mean')}, stats -> {result.stats_path}"
    )


@app.command("train-dynamic")
def train_dynamic_cmd(
    spec: Path = typer.Option(..., "--spec", "-s", help="job spec JSON (see wtc4d/recon/jobs/)"),
) -> None:
    """Train a 4D window from a job spec (the same path the GPU runner takes)."""
    from rich import print as rprint

    from .job import load_spec, run

    out = run({**load_spec(spec), "kind": "train_dynamic"})
    rprint(json.dumps(out, indent=2))


@app.command("run-job")
def run_job_cmd(
    spec: Path = typer.Argument(..., help="job spec JSON/YAML"),
) -> None:
    """Run any recon job spec locally (what ``infra`` runs in the GPU image)."""
    from rich import print as rprint

    from .job import load_spec, run

    rprint(json.dumps(run(load_spec(spec)), indent=2))


@app.command("export-colmap")
def export_colmap_cmd(
    poses: Path = typer.Option(..., "--poses", help="poses.jsonl"),
    out: Path = typer.Option(..., "--out", help="output directory for the text model"),
) -> None:
    """Export registered poses as a COLMAP text model."""
    from rich import print as rprint

    from .data import load_poses_jsonl, write_colmap_model

    p = load_poses_jsonl(poses)
    write_colmap_model(out, p)
    rprint(f"[green]wrote[/green] {len(p)} images -> {out}")


@app.command("export-transforms")
def export_transforms_cmd(
    poses: Path = typer.Option(..., "--poses", help="poses.jsonl"),
    out: Path = typer.Option(..., "--out", help="output transforms.json"),
    times: Path | None = typer.Option(None, "--times", help="time_estimates.jsonl"),
) -> None:
    """Export registered poses as a nerfstudio ``transforms.json``."""
    from rich import print as rprint

    from .data import load_poses_jsonl, load_time_estimates, write_transforms_json

    p = load_poses_jsonl(poses)
    t = load_time_estimates(times) if times else None
    write_transforms_json(out, p, times=t)
    rprint(f"[green]wrote[/green] {len(p)} frames -> {out}")


@app.command("import-colmap")
def import_colmap_cmd(
    model: Path = typer.Argument(..., help="COLMAP text model directory"),
    out: Path = typer.Option(..., "--out", help="output poses.jsonl"),
) -> None:
    """Import a COLMAP text model into ``poses.jsonl`` (world frame assumed ENU)."""
    from rich import print as rprint

    from .data import read_colmap_model, save_poses_jsonl

    poses = read_colmap_model(model).to_poses()
    save_poses_jsonl(out, poses)
    rprint(
        f"[green]wrote[/green] {len(poses)} poses -> {out}  "
        "[yellow](verify the model really is in the wtc4d ENU frame)[/yellow]"
    )


@app.command("ply-info")
def ply_info_cmd(path: Path = typer.Argument(..., help="3DGS .ply")) -> None:
    """Summarise a splat file: count, extent, opacity and layer histogram."""
    from rich import print as rprint

    from .gaussians import Gaussians, Layer

    g = Gaussians.load_ply(path)
    means = g.means
    rprint(f"{g.n} gaussians, SH degree {g.sh_degree}")
    rprint(f"extent (m): min={means.min(0).values.tolist()} max={means.max(0).values.tolist()}")
    rprint(f"opacity: mean={float(g.opacities().mean()):.3f}")
    rprint(f"scale (m): median={float(g.scales().median()):.3f} max={float(g.scales().max()):.3f}")
    if g.layer is not None:
        counts = {Layer(int(k)).name: int((g.layer == k).sum()) for k in g.layer.unique()}
        rprint(f"layers: {counts}")
    if g.t_center is not None:
        rprint(
            f"time window: t in [{float(g.t_center.min()):.1f}, {float(g.t_center.max()):.1f}] s"
        )


@app.command("toy")
def toy_cmd(
    kind: str = typer.Argument("static", help="static | dynamic"),
    iterations: int = typer.Option(200, help="training iterations"),
    out_dir: Path = typer.Option(Path("runs/recon"), "--out-dir"),
) -> None:
    """Run the CPU toy training end to end (no data required)."""
    from rich import print as rprint

    from .job import run

    spec = {"kind": f"toy_{kind}", "iterations": iterations, "out_dir": str(out_dir)}
    rprint(json.dumps(run(spec), indent=2))
