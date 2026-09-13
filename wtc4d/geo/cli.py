"""``wtc4d geo`` -- build and inspect the 2001 static scene prior."""

from __future__ import annotations

import json
import math
from pathlib import Path

import typer

from wtc4d.geo.paths import DATA_DIR, build_dir, cache_dir

app = typer.Typer(no_args_is_help=True, help="Lower Manhattan as of 2001-09-11")

GLB_COMMIT_LIMIT_BYTES = 15 * 1024 * 1024


@app.command()
def info() -> None:
    """Summarise the committed scene: counts, sources, tallest structures."""
    from rich import print as rprint

    from wtc4d.geo.landmarks import load_landmarks
    from wtc4d.geo.scene import scene_metadata
    from wtc4d.geo.wtc import PLAZA_ELEVATION_M, SITE_GRID_AZIMUTH_DEG

    meta = scene_metadata()
    lms = load_landmarks()
    rprint(
        f"[bold]geo[/bold]  buildings={meta['n_buildings']}  landmarks={len(lms)} "
        f"(2001: {sum(1 for lm in lms if lm.existed_on_2001_09_11)})"
    )
    rprint(f"site grid azimuth {SITE_GRID_AZIMUTH_DEG} deg, plaza datum {PLAZA_ELEVATION_M} m MSL")
    rprint("sources:")
    for k, v in sorted(meta["sources"].items(), key=lambda kv: -kv[1]):
        rprint(f"  {v:6d}  {k or '(none)'}")
    rprint("tallest:")
    for t in meta["tallest"]:
        rprint(f"  {t['roof_elev_m']:8.1f} m  {t['id']:<28} {t['name'][:48]}")


@app.command()
def build(
    out: Path = typer.Option(DATA_DIR, "--out", help="directory for the derived files"),  # noqa: B008
    radius_m: float = typer.Option(2600.0, help="half-size of the reconstruction box"),
    simplify_m: float = typer.Option(0.5, help="footprint simplification tolerance"),
    osm: bool = typer.Option(True, help="include OSM footprints outside NYC planimetrics"),
    glb: bool = typer.Option(True, help="also export lower_manhattan_2001.glb"),
    refresh: bool = typer.Option(False, help="ignore the download cache"),
) -> None:
    """Fetch, correct to 2001 and write buildings.geojson / landmarks.json / .glb."""
    from rich import print as rprint

    from wtc4d.geo.build import build_geojson, export_glb
    from wtc4d.geo.scene import load_buildings, load_scene

    out.mkdir(parents=True, exist_ok=True)
    doc, records = build_geojson(
        radius_m=radius_m, include_osm=osm, simplify_tol_m=simplify_m, refresh=refresh
    )
    gj = out / "buildings_2001.geojson"
    gj.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    rprint(f"wrote {gj}  ({len(doc['features'])} features, {gj.stat().st_size / 1e6:.2f} MB)")

    if glb:
        load_buildings.cache_clear()
        scene = load_scene()
        target = out / "lower_manhattan_2001.glb"
        size = export_glb(scene, target)
        if size > GLB_COMMIT_LIMIT_BYTES:
            fallback = build_dir() / "lower_manhattan_2001.glb"
            target.replace(fallback)
            rprint(
                f"[yellow]glb is {size / 1e6:.1f} MB (> 15 MB): moved to {fallback}, "
                "not committed[/yellow]"
            )
        else:
            rprint(f"wrote {target}  ({size / 1e6:.2f} MB)")


@app.command()
def landmarks(
    out: Path = typer.Option(DATA_DIR / "landmarks.json", "--out"),  # noqa: B008
    limit: int = typer.Option(45, help="how many auto-derived DoITT roof landmarks to keep"),
    radius_m: float = typer.Option(7000.0, help="search radius for tall buildings"),
) -> None:
    """Rebuild data/geo/landmarks.json from DoITT + the WTC model + manual entries."""
    from rich import print as rprint

    from wtc4d.geo.build import build_landmarks
    from wtc4d.geo.landmarks import save_landmarks

    lms = build_landmarks(limit_doitt=limit, radius_m=radius_m)
    n = save_landmarks(lms, out)
    rprint(
        f"wrote {out} ({n} landmarks, "
        f"{sum(1 for lm in lms if lm.existed_on_2001_09_11)} present on 2001-09-11)"
    )


@app.command()
def terrain(
    out: Path = typer.Option(DATA_DIR / "terrain_grid.json", "--out"),  # noqa: B008
    half_size_m: float = typer.Option(2600.0),
    cell_m: float = typer.Option(200.0),
) -> None:
    """Sample USGS 3DEP onto the coarse ENU terrain grid."""
    from rich import print as rprint
    from rich.progress import Progress

    from wtc4d.geo.sources import write_json
    from wtc4d.geo.terrain import build_terrain_grid

    with Progress() as bar:
        task = bar.add_task("sampling 3DEP", total=None)

        def tick(done: int, total: int) -> None:
            bar.update(task, completed=done, total=total)

        doc = build_terrain_grid(half_size_m=half_size_m, cell_m=cell_m, progress=tick)
    write_json(out, doc)
    rprint(f"wrote {out} ({doc['shape'][0]}x{doc['shape'][1]} cells)")


@app.command()
def preview(
    c2w: str = typer.Option(
        "", "--c2w", help="16 comma-separated floats (row-major camera-to-world)"
    ),
    from_: str = typer.Option(
        "", "--from", help="camera position as lat,lon,alt_m (alternative to --c2w)"
    ),
    look_at: str = typer.Option("0,0,250", "--look-at", help="ENU target x,y,z for --from"),
    width: int = typer.Option(640),
    height: int = typer.Option(480),
    hfov_deg: float = typer.Option(50.0),
    out: Path = typer.Option(Path("preview.png"), "--out"),  # noqa: B008
    backend: str = typer.Option("auto"),
) -> None:
    """Render a depth/instance preview PNG of the 2001 scene."""
    import numpy as np
    from rich import print as rprint

    from wtc4d.geo.render import render_view
    from wtc4d.geo.scene import load_scene
    from wtc4d.schema.camera import CameraIntrinsics
    from wtc4d.schema.geometry import LatLonAlt
    from wtc4d.world import latlon_to_enu

    if c2w:
        m = np.array([float(x) for x in c2w.split(",")], dtype=np.float64).reshape(4, 4)
    elif from_:
        lat, lon, alt = (float(x) for x in from_.split(","))
        eye = latlon_to_enu(LatLonAlt(lat=lat, lon=lon, alt_m=alt))
        target = np.array([float(x) for x in look_at.split(",")], dtype=np.float64)
        m = look_at_c2w(eye, target)
    else:
        raise typer.BadParameter("pass --c2w or --from")
    fx = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    intr = CameraIntrinsics(
        width=width, height=height, fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0
    )
    res = render_view(m, intr, load_scene(), backend=backend)
    write_preview_png(res, out)
    hit = int((res["instance_id"] >= 0).sum())
    rprint(f"wrote {out} (backend={res['backend']}, {hit}/{width * height} px on geometry)")


def look_at_c2w(eye, target, up=(0.0, 0.0, 1.0)):
    """Camera-to-world matrix in OpenCV axes looking from ``eye`` at ``target``."""
    import numpy as np

    eye = np.asarray(eye, dtype=np.float64)
    fwd = np.asarray(target, dtype=np.float64) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(up, dtype=np.float64))
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, down, fwd, eye
    return m


def write_preview_png(res, path: Path) -> Path:
    """Depth + instance id as a single false-colour PNG."""
    import numpy as np
    from PIL import Image

    depth = np.asarray(res["depth"], dtype=np.float64)
    inst = np.asarray(res["instance_id"])
    hit = np.isfinite(depth)
    img = np.zeros(depth.shape + (3,), dtype=np.uint8)
    img[..., 2] = 40  # background
    if hit.any():
        d = depth.copy()
        lo, hi = np.percentile(d[hit], [2, 98])
        shade = np.clip((hi - d) / max(hi - lo, 1e-6), 0.0, 1.0)
        tint = ((inst.astype(np.int64) * 2654435761) % 96).astype(np.float64) / 96.0
        img[..., 0] = np.where(hit, (60 + 195 * shade) * (0.55 + 0.45 * tint), 0)
        img[..., 1] = np.where(hit, (60 + 195 * shade) * (0.75 + 0.25 * (1 - tint)), 0)
        img[..., 2] = np.where(hit, 50 + 150 * shade, 40)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, optimize=True)
    return path


@app.command()
def cache() -> None:
    """Show where downloads and large build outputs are kept."""
    from rich import print as rprint

    rprint(f"download cache : {cache_dir()}")
    rprint(f"build outputs  : {build_dir()}")
    rprint(f"committed data : {DATA_DIR}")
