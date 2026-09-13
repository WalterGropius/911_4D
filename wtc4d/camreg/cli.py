"""``wtc4d camreg ...`` -- camera registration commands."""

# ruff: noqa: B008  (typer's own idiom: typer.Option(...)/typer.Argument(...) as
# a parameter default is how the library reads help text and defaults; it is
# not the mutable-default footgun B008 exists to catch.)

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print as rprint
from rich.table import Table

app = typer.Typer(no_args_is_help=True, help="Camera registration: intrinsics + world-frame pose.")


@app.command()
def priors(
    region: str | None = typer.Option(None, help="filter by region (manhattan, brooklyn, ...)"),
) -> None:
    """List the known camera vantage points."""
    from wtc4d.camreg.priors import bearing_to_wtc_deg, load_priors, range_to_wtc_m

    table = Table(title="Camera priors")
    for col in ("id", "region", "kind", "conf", "range (km)", "bearing", "sigma (m)", "moving"):
        table.add_column(col)
    for p in load_priors():
        if region and p.region != region:
            continue
        table.add_row(
            p.id,
            p.region,
            p.kind,
            p.confidence,
            f"{range_to_wtc_m(p) / 1000:.2f}",
            f"{bearing_to_wtc_deg(p):.0f}°",
            f"{p.position_sigma_m:.0f}",
            "yes" if p.moving else "no",
        )
    rprint(table)


@app.command()
def pnp(
    annotation: Path = typer.Argument(..., help="path to a FrameAnnotation JSON"),
    prior_id: str | None = typer.Option(None, help="override the annotation's camera_prior_id"),
    estimate_k1: bool = typer.Option(False, help="also solve one radial distortion coefficient"),
    fix_position: bool = typer.Option(False, help="hold the camera at the prior's location"),
    out: Path | None = typer.Option(None, help="append the resulting CameraPose to this .jsonl"),
) -> None:
    """Solve pose + focal length from a saved annotation."""
    from wtc4d.camreg.annotations import load_annotation
    from wtc4d.camreg.pnp import PnPConfig, solve_annotation
    from wtc4d.camreg.priors import get_prior

    ann = load_annotation(annotation)
    problems = ann.check()
    if problems:
        rprint("[yellow]annotation warnings:[/yellow]")
        for p in problems:
            rprint(f"  - {p}")
    prior = get_prior(prior_id) if prior_id else None
    cfg = PnPConfig(estimate_k1=estimate_k1, fix_position=fix_position)
    result = solve_annotation(ann, prior=prior, config=cfg)
    rprint(f"[bold]{ann.shot_id}#{ann.frame_idx}[/bold]  {result.summary()}")
    for w in result.warnings:
        rprint(f"  [yellow]![/yellow] {w}")
    if result.outlier_ids:
        rprint(f"  outliers dropped: {result.outlier_ids}")
    if out:
        pose = result.to_camera_pose(ann.shot_id, ann.frame_idx)
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(pose.model_dump_json() + "\n")
        rprint(f"appended pose to {out}")


@app.command()
def annotate(
    image: Path = typer.Argument(..., help="path to a frame image (never commit this)"),
    shot: str = typer.Option(..., help="shot id"),
    frame: int = typer.Option(0, help="frame index within the shot"),
    prior: str | None = typer.Option(None, "--camera-prior", help="camera prior id for this shot"),
    backend: str = typer.Option("matplotlib", help="matplotlib | web"),
    out_dir: Path | None = typer.Option(None, help="override data/cameras/annotations"),
) -> None:
    """Click landmarks on a frame and save the observations."""
    from wtc4d.camreg import annotate as annotate_mod

    fn = annotate_mod.annotate_web if backend == "web" else annotate_mod.annotate_matplotlib
    fn(image, shot_id=shot, frame_idx=frame, camera_prior_id=prior, out_dir=out_dir)


@app.command()
def auto(
    image: Path = typer.Argument(..., help="path to a frame image"),
    prior_id: str = typer.Option(..., "--camera-prior", help="camera prior id to search around"),
    out: Path | None = typer.Option(None, help="append the resulting CameraPose to this .jsonl"),
    shot: str = typer.Option("unknown", help="shot id for the saved pose"),
    frame: int = typer.Option(0, help="frame index for the saved pose"),
) -> None:
    """Automatic render-and-match registration (best-effort; see the README)."""
    from wtc4d.camreg.annotate import load_image_gray
    from wtc4d.camreg.priors import get_prior
    from wtc4d.camreg.render_match import auto_register

    img = load_image_gray(image)
    prior = get_prior(prior_id)
    result = auto_register(img, prior)
    rprint(result.summary())
    for w in result.warnings:
        rprint(f"  [yellow]![/yellow] {w}")
    if out and result.pose:
        pose = result.pose.to_camera_pose(shot, frame, method="render_match")
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(pose.model_dump_json() + "\n")
        rprint(f"appended pose to {out}")


@app.command()
def track(
    frames_dir: Path = typer.Argument(..., help="directory of frame images, named sortably"),
    keyframe_annotation: Path = typer.Option(..., help="annotation for the keyframe"),
    keyframe_index: int = typer.Option(
        ..., help="index of the keyframe within frames_dir's sorted list"
    ),
    out: Path = typer.Option(..., help="write CameraPose per frame to this .jsonl"),
) -> None:
    """Propagate one registered frame's pose through the rest of its shot."""
    from wtc4d.camreg.annotate import load_image_gray
    from wtc4d.camreg.annotations import load_annotation
    from wtc4d.camreg.pnp import solve_annotation
    from wtc4d.camreg.track import TrackConfig, track_shot

    paths = sorted(frames_dir.iterdir())
    frames = [load_image_gray(p) for p in paths]
    ann = load_annotation(keyframe_annotation)
    kf_result = solve_annotation(ann)
    kf_pose = kf_result.to_camera_pose(ann.shot_id, keyframe_index)
    rprint(f"keyframe: {kf_result.summary()}")

    tracked = track_shot(frames, keyframe_index, kf_pose, config=TrackConfig())
    with open(out, "w", encoding="utf-8") as fh:
        for t in tracked:
            if t.pose is None:
                continue
            pose = t.pose.to_camera_pose(ann.shot_id, t.frame_idx, method="tracking")
            fh.write(pose.model_dump_json() + "\n")
    n_ok = sum(1 for t in tracked if t.pose is not None)
    rprint(f"tracked {n_ok}/{len(tracked)} frames -> {out}")


@app.command()
def report(
    poses: Path = typer.Argument(..., help="a poses.jsonl file (CameraPose per line)"),
    out: Path = typer.Option(Path("docs/img/camreg_coverage.png"), help="output image path"),
) -> None:
    """Render a small SVG/PNG coverage map of registered camera positions."""
    from wtc4d.schema.camera import CameraPose
    from wtc4d.world import TOWERS

    records = []
    with open(poses, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(CameraPose.model_validate(json.loads(line)))
    if not records:
        rprint("[yellow]no poses found[/yellow]")
        raise typer.Exit(1)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    for tower in TOWERS:
        c = tower.enu_center()
        ax.add_patch(
            plt.Circle((c[0], c[1]), tower.footprint_m / 2, color="black", alpha=0.6, zorder=3)
        )
        ax.annotate(tower.id, (c[0], c[1]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    xs = [p.position()[0] for p in records]
    ys = [p.position()[1] for p in records]
    sig = [max(p.position_sigma_m or 20.0, 5.0) for p in records]
    ax.scatter(
        xs, ys, s=[min(s, 200) for s in sig], alpha=0.5, c="tab:blue", label="registered cameras"
    )
    ax.set_aspect("equal")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.set_title(f"camreg coverage: {len(records)} poses")
    ax.legend(loc="lower right", fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    rprint(f"wrote {out}")


@app.command()
def synthetic_report() -> None:
    """Run the synthetic PnP validation and print a summary table (see README)."""
    from wtc4d.camreg.validate import run_synthetic_validation

    rows = run_synthetic_validation()
    table = Table(title="Synthetic PnP validation")
    for col in ("prior", "hfov", "n_pts", "pos_err_m", "sigma_m", "rot_err_deg", "rmse_px"):
        table.add_column(col)
    for r in rows:
        table.add_row(
            r["prior"],
            f"{r['hfov']:.0f}",
            str(r["n_pts"]),
            f"{r['pos_err_m']:.2f}",
            f"{r['sigma_m']:.2f}",
            f"{r['rot_err_deg']:.3f}",
            f"{r['rmse_px']:.2f}",
        )
    rprint(table)
