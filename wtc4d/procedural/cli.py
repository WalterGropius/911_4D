"""``wtc4d procedural`` CLI: export gaussians for a time (or a time series
plus a viewer manifest), and render quick CPU previews."""

from __future__ import annotations

from pathlib import Path

import typer

from wtc4d import timeline
from wtc4d.procedural import gaussians, render
from wtc4d.procedural.params import DEFAULT_PARAMS, load_params
from wtc4d.schema.scene import SceneManifest, SplatAsset, TimelineEvent
from wtc4d.world import WORLD_ORIGIN

app = typer.Typer(
    no_args_is_help=True,
    help="Procedural 4D baseline: towers, smoke, collapse -> gaussians, for any project time t.",
)


def _parse_time(s: str) -> float:
    """Accepts 'HH:MM:SS' (local, 2001-09-11) or a raw seconds value."""
    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = parts
        return timeline.hms(int(h), int(m), float(sec))
    return float(s)


def _save(cloud: gaussians.GaussianCloud, path: Path) -> None:
    if path.suffix.lower() == ".npz":
        cloud.save_npz(path)
    else:
        cloud.save_ply(path)


@app.command()
def export(
    t: str | None = typer.Option(None, "--t", help="local time HH:MM:SS (single-frame export)"),  # noqa: B008
    out: Path | None = typer.Option(  # noqa: B008
        None, "--out", help="output .ply or .npz (single-frame export)"
    ),
    sequence: tuple[str, str] | None = typer.Option(  # noqa: B008
        None, "--sequence", help="T_START T_END, local time HH:MM:SS (series export)"
    ),
    step: float = typer.Option(30.0, "--step", help="seconds between frames for --sequence"),  # noqa: B008
    out_dir: Path | None = typer.Option(  # noqa: B008
        None, "--out-dir", help="output directory for --sequence"
    ),
    fmt: str = typer.Option("ply", "--format", help="ply | npz, for --sequence"),  # noqa: B008
    seed: int = typer.Option(0, "--seed", help="RNG seed for jittered detail (damage, rubble)"),  # noqa: B008
    params_path: Path | None = typer.Option(None, "--params", help="override params.yaml"),  # noqa: B008
) -> None:
    """Export gaussians for one instant (--t/--out), or a time series plus a
    SceneManifest (--sequence/--step/--out-dir)."""
    params = load_params(params_path) if params_path else DEFAULT_PARAMS

    if sequence is not None:
        if out_dir is None:
            raise typer.BadParameter("--out-dir is required with --sequence")
        if fmt not in ("ply", "npz"):
            raise typer.BadParameter("--format must be 'ply' or 'npz'")
        t0, t1 = _parse_time(sequence[0]), _parse_time(sequence[1])
        if t1 <= t0:
            raise typer.BadParameter("--sequence end time must be after the start time")
        out_dir.mkdir(parents=True, exist_ok=True)

        times = []
        i = 0
        while t0 + i * step <= t1:
            times.append(t0 + i * step)
            i += 1

        assets = []
        for i, ti in enumerate(times):
            cloud = gaussians.sample(ti, params=params, seed=seed)
            frame_path = out_dir / f"procedural_{i:04d}.{fmt}"
            _save(cloud, frame_path)
            t_end = times[i + 1] if i + 1 < len(times) else ti + step
            assets.append(
                SplatAsset(
                    id=f"procedural_{i:04d}",
                    url=frame_path.name,
                    format=fmt,
                    t_start=ti,
                    t_end=t_end,
                    kind="procedural",
                    layer="scene",
                    notes=f"wtc4d.procedural, seed={seed}",
                )
            )

        events = [
            TimelineEvent(id=e.id, name=e.name, t=e.t, sigma=e.sigma) for e in timeline.EVENTS
        ]
        manifest = SceneManifest(
            world_origin=WORLD_ORIGIN,
            t_min=t0,
            t_max=t1,
            events=events,
            assets=assets,
            notes="Procedural baseline (wtc4d.procedural): a kinematic sketch, not a measurement.",
        )
        (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))
        typer.echo(f"wrote {len(assets)} frame(s) + manifest.json to {out_dir}")
        return

    if t is None or out is None:
        raise typer.BadParameter("provide either --t/--out or --sequence/--out-dir")
    tv = _parse_time(t)
    cloud = gaussians.sample(tv, params=params, seed=seed)
    _save(cloud, out)
    typer.echo(f"wrote {len(cloud)} gaussians to {out} (t={timeline.fmt_local(tv)})")


@app.command()
def preview(
    t: str = typer.Option(..., "--t", help="local time HH:MM:SS"),  # noqa: B008
    out: Path = typer.Option(..., "--out", help="output .png"),  # noqa: B008
    viewpoint: str = typer.Option(  # noqa: B008
        "jersey_city", "--viewpoint", help="jersey_city | brooklyn_promenade | helicopter"
    ),
    seed: int = typer.Option(0, "--seed"),  # noqa: B008
    params_path: Path | None = typer.Option(None, "--params"),  # noqa: B008
) -> None:
    """Render a quick CPU preview from a canonical viewpoint."""
    if viewpoint not in render.VIEWPOINTS:
        raise typer.BadParameter(
            f"unknown viewpoint {viewpoint!r}; choices: {sorted(render.VIEWPOINTS)}"
        )
    params = load_params(params_path) if params_path else DEFAULT_PARAMS
    tv = _parse_time(t)
    cloud = gaussians.sample(tv, params=params, seed=seed)
    cam = render.VIEWPOINTS[viewpoint]()
    image = render.render(cloud, cam)
    out.parent.mkdir(parents=True, exist_ok=True)
    render.save_png(out, image)
    typer.echo(f"wrote {cam.width}x{cam.height} preview to {out} (t={timeline.fmt_local(tv)})")
